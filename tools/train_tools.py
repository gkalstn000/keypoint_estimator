import math
import time

import torch.nn as nn

import utils
import tools.eval_tools as eval_tools
from tqdm import trange
from custom_loss.pose_loss import cal_pose_loss
from custom_loss.limb_agreement import cal_limb_agreement

class Trainer :
    def __init__(self,
                 opt,
                 model):
        self.opt = opt
        self.model = model
        self.device = opt.device

    def trainIters(self,
                   opt,
                   model_optimizer,
                   scheduler,
                   dataloader):

        start = time.time()
        plot_losses = []
        loss_total = opt.loss  # print_every 마다 초기화

        mse = nn.MSELoss()
        limb_agreement = cal_limb_agreement
        for epoch in trange(opt.epoch, opt.n_epochs+1) :
            for src, tgt, mid_point, length in dataloader:
                self.model.train()
                src, tgt = src.float(), tgt.float()

                model_optimizer.zero_grad()
                pred = self.model(src)
                # loss = 0.001*mse(tgt, pred) + 1*limb_agreement(tgt, pred)
                loss = limb_agreement(tgt, pred)
                print(f'limb agreement : {limb_agreement(tgt, pred)}')

                # print(f'mse : {0.001*mse(tgt, pred)}\t limb agreement : {limb_agreement(tgt, pred)}')
                loss_total += (loss.item() / tgt.size(0))

                loss.backward()
                model_optimizer.step()
            scheduler.step(loss_total / epoch)

            if epoch % opt.print_every == 0:
                print_loss_avg = loss_total / epoch
                print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / opt.n_epochs),
                                             epoch, epoch / opt.n_epochs * 100, print_loss_avg))

            if epoch % opt.plot_every == 0:
                plot_loss_avg = loss_total / epoch
                plot_losses.append(plot_loss_avg)

            if epoch % opt.save_epoch == 0 :
                file_name = f'model_params_{epoch}_epoch'
                utils.save_model(opt, epoch, self.model, model_optimizer, scheduler, loss_total, file_name)

        eval_tools.showPlot(plot_losses)


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