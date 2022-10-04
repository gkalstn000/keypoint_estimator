import torch.nn as nn
import torch

class Embedding(nn.Module):
    def __init__(self, grid_num, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.grid_num = grid_num

        input_size = embedding_dim * 2 +1
        self.max_len = 18
        self.h_embedding = nn.Embedding(grid_num+1, embedding_dim) # grid embedding + unknown_token
        self.w_embedding = nn.Embedding(grid_num+1, embedding_dim) # grid embedding + unknown_token
        self.pos_embed = nn.Embedding(self.max_len, input_size)  # 18 position embedding

        self.norm = nn.LayerNorm(input_size)
        self.pos = torch.arange(self.max_len)

    def forward(self, x):
        # key point embedding
        unknown_index = x.isnan().any(-1)
        token = self.coord_mapping_to_token(x, self.grid_num)
        h_embedding = self.h_embedding(token[:, :, 0])
        w_embedding = self.w_embedding(token[:, :, 1])
        point_embedding = torch.cat([h_embedding, w_embedding, unknown_index.unsqueeze(2)], dim = 2)
        # pos embedding
        pos = self.pos.unsqueeze(0).repeat(x.size(0), 1)  # [seq_len] -> [batch_size, seq_len]
        pos_embedding = self.pos_embed(pos)
        embedding = point_embedding + pos_embedding
        return self.norm(embedding)

    def coord_mapping_to_token(self, x, grid_num):
        # float [0, 1] -> integer [0, grid] mapping
        token = torch.div(x, 1 / grid_num, rounding_mode='trunc')
        token = torch.where(token == grid_num, grid_num - 1, token) # h 좌표가 max_h 인경우 unknown_token으로 mapping 되는거 방지)
        token[token.isnan()] = grid_num
        return token.int()

