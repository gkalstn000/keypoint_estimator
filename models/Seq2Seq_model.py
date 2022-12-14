import torch.nn as nn
import torch
import torch.nn.functional as F

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, device, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Linear(self.output_size, self.hidden_size)
        self.d_hidden = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.device = device

    def forward(self, input, hidden, encoder_outputs):
        # input size : (B, 1, 2)
        # hidden size : (1, B, hidden)
        # embedded size : (B, 1, hidden)

        embedded = self.embedding(input) # (1, 1, 5)
        # embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded, hidden)
        # 잘못썼음. GRU cell을 써야함. 하나씩 넣을 때는 cell을 써야함.
        # output으로 attention 구하는게 맞는지 확인
        # 사실은 RNN 한칸으로 계속 쓰고있엇던거임
        # 같은값나오는거보면 이거뿐만이아니고 다른것도 잘못되었을꺼임
        attention_scores = encoder_outputs @ output.transpose(2, 1)
        attention_weight = F.softmax(attention_scores, dim=1)
        attention_value = encoder_outputs * attention_weight
        attention_vector = attention_value.sum(dim=1, keepdim = True)

        output = torch.cat((output, attention_vector), 2)
        output = self.attn_combine(output)
        # output = F.relu(output)
        output = self.out(output)

        return output, hidden, 0

    def initHidden(self, batch_size = 1):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

