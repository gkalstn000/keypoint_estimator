import pickle
from tqdm import tqdm
import pandas as pd
import json
import torch
import os

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


def save_model(opt, epoch, model, optimizer, loss, file_name) :
    id = opt.id
    root = 'checkpoints'
    latest_file_name = 'model_param_latest'

    file_path = os.path.join(root, id, file_name)
    latest_file_path = os.path.join(root, id, latest_file_name)

    print('saveing model')
    print(f"""
    epoch : {epoch}
    loss : {loss}
    model_state_dict
    optimizer_state_dict
    """)
    torch.save({'epoch' : epoch,
                'model_state_dict' : model.cpu().state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'loss' : loss}, file_path)

    torch.save({'epoch': epoch,
                'model_state_dict': model.cpu().state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, latest_file_path)

def load_model(opt, model, optimizer) :
    if not opt.continue_train :
        return model, optimizer, 1, 0
    id = opt.id
    root = 'checkpoints'
    file_path = os.path.join(root, id, opt.model_name)
    assert os.path.isfile(file_path), f'there is no {file_path}'

    print(f'Load model, optimizer, epoch, loss from {file_path}')

    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss




if __name__ == "__main__":
    train_path = 'dataset/train/pose_label.pkl'
    test_path = 'dataset/test/fasion-annotation-test.csv'
    train_X = load_train_data(train_path)
    test_X = load_test_data(test_path)
