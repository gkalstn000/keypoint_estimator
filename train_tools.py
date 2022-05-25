import torch
import random
import time
import math
import time

import torch.nn as nn
from torch import optim

import utils
import eval_tools
from tqdm import tqdm, trange
class Trainer :
    def __init__(self,
                 opt,
                 model,
                 grid_size_tensor,
                 device):
        self.opt = opt
        self.model = model
        self.grid_size_tensor = grid_size_tensor
        self.device = device

    def trainIters(self,
                   opt,
                   dataloader):
        n_epochs = opt.n_epochs
        print_every = opt.print_every
        plot_every = opt.plot_every
        learning_rate = opt.learning_rate

        start = time.time()
        plot_losses = []
        print_loss_total = 0  # print_every 마다 초기화
        plot_loss_total = 0  # plot_every 마다 초기화

        model_optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        criterion = nn.MSELoss()

        # 여기 batch 단위로 받도록 수정해야겠음.

        for epoch in trange(1, n_epochs+1) :
            for src, tgt, mid_point, length in tqdm(dataloader):
                self.model.train()
                src, tgt = src.float(), tgt.float()


                loss = self.train(src,
                                  tgt,
                                  model_optimizer,
                                  criterion)


                print(loss)
                print_loss_total += loss
                plot_loss_total += loss

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                             epoch, epoch / n_epochs * 100, print_loss_avg))

            if epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        eval_tools.showPlot(plot_losses)

    def train(self,
              src,
              tgt,
              model_optimizer,
              criterion):
        batch_size = len(src)
        model_optimizer.zero_grad()

        loss = 0
        # start = time.time()
        pred = self.model(src, self.grid_size_tensor)
        # end = time.time()
        # print(f"forward : {end - start:.5f} sec")
        loss += criterion(tgt, pred)
        # start = time.time()
        loss.backward()
        # end = time.time()
        # print(f"backward : {end - start:.5f} sec")
        # start = time.time()
        model_optimizer.step()
        # end = time.time()
        # print(f"step : {end - start:.5f} sec")
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