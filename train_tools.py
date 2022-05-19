import torch
import random
import time
import math

import torch.nn as nn
from torch import optim

import utils
import eval_tools
from tqdm import tqdm, trange
class Trainer :
    def __init__(self,
                 encoder,
                 decoder,
                 max_length,
                 device,
                 teacher_forcing_ratio):
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
    def trainIters(self,
                   n_iters,
                   dataloader,
                   print_every=1000,
                   plot_every=100,
                   learning_rate=0.01):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # print_every 마다 초기화
        plot_loss_total = 0  # plot_every 마다 초기화

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)

        criterion = nn.MSELoss()

        # 여기 batch 단위로 받도록 수정해야겠음.

        for iter in trange(n_iters) :
            iter += 1
            for src, tgt in dataloader:
                src, tgt = src.float(), tgt.float()

                loss = self.train(src,
                             tgt,
                             encoder_optimizer,
                             decoder_optimizer,
                             criterion
                             )

                print_loss_total += loss
                plot_loss_total += loss

                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                                 iter, iter / n_iters * 100, print_loss_avg))

                if iter % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

        eval_tools.showPlot(plot_losses)

    def train(self,
              src,
              tgt,
              encoder_optimizer,
              decoder_optimizer,
              criterion):
        batch_size = len(src)
        encoder_hidden = self.encoder.initHidden(batch_size)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss = 0

        encoder_output, encoder_hidden = self.encoder(src, encoder_hidden)

        # decoder_input = torch.tensor([[self.SOS_token]], device=self.device)
        decoder_hidden = encoder_hidden

        # use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        use_teacher_forcing = True
        if use_teacher_forcing:
            for di in range(tgt.size(1)-1) :
                decoder_src = tgt[:, di, :].unsqueeze(1)
                decoder_tgt = tgt[:, di+1, :].unsqueeze(1)
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_src, decoder_hidden, encoder_output)
                loss += criterion(decoder_output, decoder_tgt)

        else:
            decoder_src = tgt[:, 0, :].unsqueeze(1)
            for di in range(1, tgt.size(1)) :
                # Teacher forcing 미포함: 자신의 예측을 다음 입력으로 사용
                decoder_tgt = tgt[:, di, :].unsqueeze(1)
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_src, decoder_hidden, encoder_output)
                decoder_src = decoder_output

                loss += criterion(decoder_output, decoder_tgt)


        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / tgt.size(0)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))