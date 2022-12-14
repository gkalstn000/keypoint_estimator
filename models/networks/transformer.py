import math
import torch
import numpy as np
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, h_grid, w_grid, embedding_dim):
        super(Embedding, self).__init__()
        input_size = embedding_dim * 2 +1
        self.max_len = 18

        self.h_embedding = nn.Embedding(h_grid+1, embedding_dim)
        self.w_embedding = nn.Embedding(w_grid+1, embedding_dim)
        self.pos_embed = nn.Embedding(self.max_len, input_size)  # 18 position embedding

        self.norm = nn.LayerNorm(input_size)
        self.pos = torch.arange(self.max_len)

    def forward(self, x, grid_size_tensor):
        # key point embedding
        unknown_index = (x > 1)[:, :, 0]
        grid_embedding_index = torch.div(x + 1, grid_size_tensor[None, None, :], rounding_mode='trunc').int()
        h_embedding = self.h_embedding(grid_embedding_index[:, :, 0])
        w_embedding = self.w_embedding(grid_embedding_index[:, :, 1])
        point_embedding = torch.cat([h_embedding, w_embedding, unknown_index.unsqueeze(2)], dim = 2)
        # pos embedding

        pos = self.pos.unsqueeze(0).repeat(x.size(0), 1)  # [seq_len] -> [batch_size, seq_len]
        pos_embedding = self.pos_embed(pos.to(x.device.type))
        embedding = point_embedding + pos_embedding
        return self.norm(embedding)

def gelu(x):
    """ Implementation of the gelu activation function. For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) Also see https://arxiv.org/abs/1606.08415 """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, n_heads, seq_len, seq_len]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
    def forward(self, Q, K, V):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, seq_len, n_heads, d_v]
        output = self.fc(context)
        return self.norm(output + residual) # output: [batch_size, seq_len, d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model = d_model, d_k = d_k, d_v = d_v, n_heads = n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model = d_model, d_ff = d_ff)

    def forward(self, enc_inputs):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs



from options.train_options import TrainOptions
# =========== Import Dataset modules ===========
import data
# =========== Import Models ===========
from models import create_model
# =========== Import util modules ===========
from util import util
from tools.train_tools import Trainer
# =========== Import etc... ===========
import sys
if __name__ == '__main__':
    h_grid, w_grid, embedding_dim = 100, 100, 7
    embedding = Embedding(h_grid, w_grid, embedding_dim)
    opt = TrainOptions().parse()
    # print options to help debugging
    print(' '.join(sys.argv))

    dataloader = data.create_dataloader(opt)

    # cuda??? model????????? ??????????????????
    for i, data_i in enumerate(dataloader):
        print(data_i)
        src_keypoint = data_i['source_keypoint']
        embedding(data_i)
