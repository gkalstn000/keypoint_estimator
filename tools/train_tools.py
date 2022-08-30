import math
import time

import torch.nn as nn
import torch
import utils
import tools.eval_tools as eval_tools
from tqdm import trange
from custom_loss.pose_loss import cal_pose_loss
from custom_loss.limb_agreement import cal_limb_agreement
from scores.PCkH_score import pckh_score

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
                   train_dl,
                   valid_dl):

        start = time.time()
        plot_total_losses = []
        plot_keypoint_losses = []
        plot_occlusion_losses = []
        loss_total = opt.loss  # print_every 마다 초기화
        loss_keypoint = 0
        loss_occlusion = 0



        MSE = nn.MSELoss()
        BCE = nn.BCELoss()
        limb_agreement = cal_limb_agreement
        for epoch in trange(opt.epoch, opt.n_epochs+1) :
            train_pckh_list = []
            for src, tgt, mid_point, length in train_dl:
                self.model.train()
                src, tgt = src.float(), tgt.float()
                keypoint_tgt, occlusion_tgt = tgt[:, :, :2], tgt[:, :, 2].unsqueeze(2)
                visible_index = occlusion_tgt.squeeze() != 1
                model_optimizer.zero_grad()
                keypoint_logits, occlusion_logits = self.model(src)
                # loss = 0.001*mse(tgt, pred) + 1*limb_agreement(tgt, pred)

                occlusion_BCEloss = BCE(occlusion_logits, occlusion_tgt)
                keypoint_MSEloss = MSE(keypoint_tgt[visible_index], keypoint_logits[visible_index])

                loss =  opt.lambda_k * keypoint_MSEloss + opt.lambda_o * occlusion_BCEloss
                # print(f'mse : {keypoint_MSEloss}\t occlusion  : {occlusion_BCEloss}')
                loss_total += (loss.item() / tgt.size(0))
                loss_keypoint += (opt.lambda_k * keypoint_MSEloss.item() / tgt.size(0))
                loss_occlusion += (opt.lambda_o * occlusion_BCEloss.item() / tgt.size(0))

                # print(f'loss total : {loss_total / epoch}')

                loss.backward()
                model_optimizer.step()
                if epoch % opt.print_every == 0:
                    train_pckh_list.append(pckh_score('', keypoint_tgt, keypoint_logits))
            scheduler.step(loss_total / epoch)

            if epoch % opt.print_every == 0:
                print_loss_avg = loss_total / epoch
                print('%s (%d %d%%)' % (timeSince(start, epoch / opt.n_epochs),
                                             epoch, epoch / opt.n_epochs * 100))

                self.model.eval()
                valid_pckh_list = []
                with torch.no_grad():
                    for src, tgt, mid_point, length in valid_dl:
                        src, tgt = src.float(), tgt.float()
                        keypoint_tgt, occlusion_tgt = tgt[:, :, :2], tgt[:, :, 2].unsqueeze(2)
                        keypoint_logits, occlusion_logits = self.model(src)
                        valid_pckh_list.append(pckh_score('', keypoint_tgt, keypoint_logits))
                train_pckh = torch.mean(torch.Tensor(train_pckh_list))
                valid_pckh = torch.mean(torch.Tensor(valid_pckh_list))
                print('Total loss:', round(print_loss_avg, 3))
                print('Train pckh score:', round(train_pckh.item(), 3))
                print('Valid pckh score:', round(valid_pckh.item(), 3))


            if epoch % opt.plot_every == 0:
                plot_loss_avg = loss_total / epoch
                plot_keyloss_avg = loss_keypoint / epoch
                plot_occlusion_avg = loss_occlusion / epoch

                plot_total_losses.append(plot_loss_avg)
                plot_keypoint_losses.append(plot_keyloss_avg)
                plot_occlusion_losses.append(plot_occlusion_avg)

            if epoch % opt.save_epoch == 0 :
                file_name = f'model_params_{epoch}_epoch'
                utils.save_model(opt, epoch, self.model, model_optimizer, scheduler, loss_total, file_name)

        eval_tools.showPlot(plot_total_losses, plot_keypoint_losses, plot_occlusion_losses, opt)


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