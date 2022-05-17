import pickle
from tqdm import tqdm
import pandas as pd
import json
import torch

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


#=================== 밑에는 doc code ========================


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence, EOS_token, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair, input_lang, output_lang, EOS_token, device):
    input_tensor = tensorFromSentence(input_lang, pair[0], EOS_token, device)
    target_tensor = tensorFromSentence(output_lang, pair[1], EOS_token, device)
    return (input_tensor, target_tensor)






if __name__ == "__main__":
    train_path = 'dataset/train/pose_label.pkl'
    test_path = 'dataset/test/fasion-annotation-test.csv'
    train_X = load_train_data(train_path)
    test_X = load_test_data(test_path)
