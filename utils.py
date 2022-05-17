import pickle
from tqdm import tqdm

def load__train_data(data_path) :
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
            train_X[key] = val
    return train



if __name__ == "__main__":

    pass