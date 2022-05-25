import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import random
import torch

import utils


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # 주기적인 간격에 이 locator가 tick을 설정
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
class Evaler :
    def __init__(self,
                 model,
                 grid_size_tensor,
                 dataloader,
                 device):
        self.model = model
        self.grid_size_tensor = grid_size_tensor
        self.device = device
        self.dataloader = dataloader

    def print_points(self, src, tgt, pred, key_point_name):
        pred = pred.numpy()
        strFormat = '%-12s%-12s%-12s%-12s\n'
        strOut = strFormat % ('Name', 'Source', 'Target', 'Pred')
        for name, src_p, tgt_p, pred_ in zip(key_point_name, src.astype(np.int8), tgt.astype(np.int8), pred.astype(np.int8)) :
            strOut += strFormat % (name, src_p, tgt_p, pred_)
        print(strOut)
    def evaluateRandomly(self,n=10):
        key_point_name = ['Nose', 'Neck', 'R_shoulder', 'R_elbow', 'R_wrist', 'L_shoulder', 'L_elbow', 'L_wrist',
                          'R_pelvis', 'R_knee', 'R_ankle', 'L_pelvis', 'L_knee', 'L_ankle','R_eye', 'L_eye', 'R_ear', 'L_ear']
        for i in range(n):
            src, tgt, mid_point, length = random.choice(self.dataloader.dataset)
            pred = self.evaluate(point=src)
            src_denorm = self.denormalization(src[:, :-1], 128, 256)
            self.print_points(src_denorm, tgt, pred, key_point_name)


    def evaluate(self, point):
        self.model.eval()
        with torch.no_grad():
            point = torch.from_numpy(point).float()
            pred = self.model(point, self.grid_size_tensor)
            return pred.squeeze()

    def denormalization(self, points, mid_point, length):
        max_point = np.array([256, 256])
        unknown = np.array([-1, -1])
        up_scale = (points * length) / 2 + mid_point

        return np.where(up_scale > max_point, unknown, up_scale)

from data.mydata import MyDataSet
import torch.utils.data as Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.bidirectional_lstm import Bidirectional_LSTM
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

    data_path = 'dataset/train/pose_label.pkl'
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
                  grid_size_tensor=grid_size_tensor,
                  dataloader=dataloader,
                  device=device)

    eval.evaluateRandomly()