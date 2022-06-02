import pickle
from tqdm import tqdm
import pandas as pd
import json
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


def load_train_data(data_path) :
    '''
    좌표가 반대로(x, y) 저장되어있음. 바꿔서(h, w) 사용하기
    :param data_path:
    :return:
    '''
    with open(data_path, 'rb') as f:
        train = pickle.load(f, encoding='iso-8859-1')
    train_X = {}
    for key, val in tqdm(train.items()):
        count_flag = 0
        for x, y in val:
            if not (x > 0 and y > 0):
                count_flag += 1
        if count_flag == 0:
            train_X[key] = [[y, x] for x, y in val] # (x, y) -> (h, w)
    return train_X

def load_test_data(data_path) :
    df = pd.read_csv(data_path, sep=':')
    test_X = {}
    for index, (file_name, h_points_str, w_points_str) in df.iterrows() :
        h_points = json.loads(h_points_str)
        w_points = json.loads(w_points_str)
        test_X[file_name] = [[h, w] for h, w in zip(h_points, w_points)]
    return test_X


def save_model(opt, epoch, model, optimizer, scheduler, loss, file_name) :
    id = opt.id
    root = 'checkpoints'
    latest_file_name = 'model_param_latest'

    file_path = os.path.join(root, opt.model, id, file_name)
    latest_file_path = os.path.join(root, opt.model, id, latest_file_name)

    print('saveing model')
    print(f"""
    epoch : {epoch}
    loss : {loss / epoch}
    model_state_dict
    optimizer_state_dict
    """)
    torch.save({'epoch' : epoch,
                'model_state_dict' : model.cpu().state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'loss' : loss}, file_path)

    torch.save({'epoch': epoch,
                'model_state_dict': model.cpu().state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': loss}, latest_file_path)

def load_model(opt, model, optimizer, scheduler) :
    if not opt.continue_train :
        return model, optimizer, scheduler, 1, 0
    id = opt.id
    root = 'checkpoints'
    file_path = os.path.join(root, opt.model, id, opt.model_name)
    assert os.path.isfile(file_path), f'there is no {file_path}'

    print(f'Load model, optimizer, epoch, loss from {file_path}')

    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, scheduler, epoch, loss

def plot_key_points(src, tgt, pred, path) :
    skeleton_tree = [[16, 14], [14, 0], [15, 0], [17, 15], [0, 1],
                     [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
                     [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13]]

    if type(pred) != np.ndarray or type(src) != np.ndarray:
        src.numpy()
        tgt.numpy()
        pred.numpy()


    # tgt drawing
    for p1, p2 in skeleton_tree:
        h1, w1 = tgt[p1]
        h2, w2 = tgt[p2]
        plt.plot([w1, w2], [-h1, -h2], color='crimson')
    # pred drawing
    for p1, p2 in skeleton_tree:
        h1, w1 = pred[p1]
        h2, w2 = pred[p2]
        plt.plot([w1, w2], [-h1, -h2], color='green')

    for (h_s, w_s), (h, w) in zip(src, tgt) :
        if h_s != -1 and w_s != -1 :
            plt.scatter(w, -h, c = 'g')
        else :
            plt.scatter(w, -h, c = 'r')

    plt.savefig(path)
    plt.cla()


if __name__ == "__main__":
    train_path = 'dataset/train/pose_label.pkl'
    test_path = 'dataset/test/fasion-annotation-test.csv'
    train_X = load_train_data(train_path)
    test_X = load_test_data(test_path)
