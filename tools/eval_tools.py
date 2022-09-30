import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

import numpy as np
import random
import torch
from tqdm import tqdm
import utils
import key_point_name as kpn
import pandas as pd
import os
def showPlot(total_loss, key_loss, occ_loss, opt):
    id = opt.id
    root = 'checkpoints'
    filename = 'loss_graph.png'
    file_path = os.path.join(root, opt.model, id, filename)

    plt.figure()
    fig, ax = plt.subplots()
    x_range = range(len(total_loss))
    ax.plot(x_range, total_loss, label="Total loss", marker = 'o')
    ax.plot(x_range, key_loss, label="Keypoint loss", marker = 'x')
    ax.plot(x_range, occ_loss, label="Occlusion loss", marker = '^')

    ax.set(xlabel='epochs', ylabel='loss', title='Loss graph')
    ax.grid()
    plt.legend()
    plt.savefig(file_path)

    loss_log_path = os.path.join(root, opt.model, id, 'loss.txt')
    with open(loss_log_path, 'w') as f :
        for i, (total, key, occ) in enumerate(zip(total_loss, key_loss, occ_loss)) :
            f.write(f'epoch : {i}, total_loss : {total}, keypoint_loss : {key}, occlusion_loss : {occ}\n')

class Evaler :
    def __init__(self,
                 opt,
                 model,
                 dataloader):
        self.model = model
        self.device = opt.device
        self.dataloader = dataloader
        self.opt = opt

    def print_points(self, sources, targets, predictions, key_point_name):
        strFormat = '%-12s%-12s%-12s%-12s\n'
        strOut = strFormat % ('Name', 'Source', 'Target', 'Pred')
        for src, tgt, pred in zip(sources, targets, predictions) :
            for name, src_p, tgt_p, pred_ in zip(key_point_name, src.int(), tgt.int(), pred.int()) :

                strOut += strFormat % (name, src_p.tolist(), tgt_p.tolist(), pred_.tolist())
            strOut += '-'*50+'\n'
        print(strOut)
    def evaluate_score(self, score, verbose = False):
        srcs = []
        tgts = []
        preds = []
        occ_trues = []
        occ_preds = []

        l2_scores = []
        l2_mask_scores = []
        pckh_scores = []
        pckh_mask_scores = []

        accuracys = []
        recalls = []
        precisions = []
        f1s = []

        for src, tgt, mid_point, length in tqdm(self.dataloader):
            src, tgt, mid_point, length = src.float(), tgt.float(), mid_point.float(), length.float()

            src = src.to(self.opt.device)
            tgt = tgt.to(self.opt.device)
            mid_point = mid_point.to(self.opt.device)
            length = length.to(self.opt.device)

            # pred 계산
            keypoint_logits, occlusion_logits = self.evaluate(point=src)
            occlusion_logits = (occlusion_logits > 0.5) * 1
            keypoint_tgt, occlusion_tgt = tgt[:, :, :2], tgt[:, :, 2]
            src_denorm = self.denormalization(src[:, :, :-1].cpu(), mid_point.cpu(), length.cpu())

            src_denorm, keypoint_tgt, keypoint_logits, occlusion_logits, occlusion_tgt = src_denorm.cpu(), keypoint_tgt.cpu(), keypoint_logits.cpu(), occlusion_logits.cpu(), occlusion_tgt.cpu()


            # image 출력을 위한 append
            srcs.append(src_denorm)
            tgts.append(keypoint_tgt)
            preds.append(keypoint_logits)

            occ_preds.append(occlusion_logits.squeeze())
            occ_trues.append(occlusion_tgt)

            if verbose :
                self.print_points(src_denorm, keypoint_tgt, keypoint_logits, kpn.key_point_name)

            (l2, l2_mask), (pckh, pckh_mask), (acc, recall, precision, f1) = score(src_denorm, keypoint_tgt, keypoint_logits, occlusion_logits, occlusion_tgt)
            l2_scores.append(l2)
            l2_mask_scores.append(l2_mask)
            pckh_scores.append(pckh)
            pckh_mask_scores.append(pckh_mask)

            #occlusion scores
            accuracys.append(acc)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)

        l2 = np.nanmean(np.stack(l2_scores, axis = 0), axis = 0)
        l2_mask = np.nanmean(np.stack(l2_mask_scores, axis = 0), axis = 0)
        pckh = np.nanmean(np.stack(pckh_scores, axis = 0), axis = 0)
        pckh_mask = np.nanmean(np.stack(pckh_mask_scores, axis = 0), axis = 0)
        keypoint_df = pd.DataFrame([l2, l2_mask, pckh, pckh_mask],
                                columns = ['total']+kpn.key_point_name,
                                index = ['l2', 'l2_masked', 'pckh', 'pckh_masked'])

        accuracys = np.nanmean(np.stack(accuracys, axis = 0), axis = 0)
        recalls = np.nanmean(np.stack(recalls, axis=0), axis=0)
        precisions = np.nanmean(np.stack(precisions, axis=0), axis=0)
        f1s = np.nanmean(np.stack(f1s, axis=0), axis=0)

        occlusion_df = pd.DataFrame([accuracys, recalls, precisions, f1s],
                                    columns = ['total']+kpn.key_point_name,
                                    index = ['Acc', 'Recall', 'Precision', 'F1'])

        return torch.cat(srcs, dim = 0), torch.cat(tgts, dim = 0),  torch.cat(preds, dim = 0), torch.cat(occ_trues, dim = 0), torch.cat(occ_preds, dim = 0), keypoint_df.round(3), occlusion_df.round(3)

    def evaluate(self, point):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(point)
            return pred

    def denormalization(self, points, mid_point, length):
        max_point = torch.Tensor([256, 176])
        unknown = torch.Tensor([-1, -1])
        up_scale = (points * length) / 2 + mid_point

        return torch.where(up_scale > max_point, unknown, up_scale)

from data.mydata import MyDataSet
import torch.utils.data as Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.bidirectional_lstm_model import Bidirectional_LSTM
if __name__ == '__main__' :
    height = 256
    width = 256
    h_grid = 100
    w_grid = 100

    # model params
    input_dim = 3
    output_dim = 2
    embedding_dim = 3
    h_grid_size = 2 / h_grid
    w_grid_size = 2 / w_grid
    hidden_dim = 5
    n_layers = 3
    bidirectional = True
    dropout = 0.5

    # training params
    batch_size = 64
    learning_rate = 0.005
    n_epochs = 100

    data_path = '../dataset/train/pose_label.pkl'
    data_dict = utils.load_train_data(data_path)
    mydata = MyDataSet(data_dict, height, width)

    dataloader = Data.DataLoader(mydata, batch_size, True)
    grid_size_tensor = torch.Tensor([h_grid_size, w_grid_size])

    lstm = Bidirectional_LSTM(input_dim=input_dim,
                              output_dim=output_dim,
                              embedding_dim=embedding_dim,
                              h_grid=h_grid,
                              w_grid=w_grid,
                              hidden_dim=hidden_dim,
                              n_layers=n_layers,
                              bidirectional=bidirectional,
                              dropout=dropout,
                              device=device)

    eval = Evaler(model=lstm,
                  dataloader=dataloader,
                  device=device)

    eval.evaluateRandomly()