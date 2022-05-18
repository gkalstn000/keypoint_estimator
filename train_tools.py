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
                 SOS_token,
                 EOS_token,
                 teacher_forcing_ratio):
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.device = device
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.teacher_forcing_ratio = teacher_forcing_ratio
    def trainIters(self,
                   n_iters,
                   pairs,
                   input_lang,
                   output_lang,
                   print_every=1000,
                   plot_every=100,
                   learning_rate=0.01):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # print_every 마다 초기화
        plot_loss_total = 0  # plot_every 마다 초기화

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        training_pairs = [utils.tensorsFromPair(random.choice(pairs), input_lang, output_lang, self.EOS_token, self.device)
                          for i in range(n_iters)]
        criterion = nn.NLLLoss()

        # 여기 batch 단위로 받도록 수정해야겠음.
        for iter in trange(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self.train(input_tensor,
                         target_tensor,
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
              input_tensor,
              target_tensor,
              encoder_optimizer,
              decoder_optimizer,
              criterion):
        encoder_hidden = self.encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[self.SOS_token]], device=self.device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing 포함: 목표를 다음 입력으로 전달
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Teacher forcing 미포함: 자신의 예측을 다음 입력으로 사용
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # 입력으로 사용할 부분을 히스토리에서 분리

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == self.EOS_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

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