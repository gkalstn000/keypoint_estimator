import json
import os
import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import key_point_name as kpn
import pandas as pd

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_df(path) :
    return pd.read_csv(path, sep=':')


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls

def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.id, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()













def load_train_data(data_path) :
    '''
    Load keypoint DataFrame
    :param data_path
    :return:
    '''

    return pd.read_csv(data_path, sep=':')

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
    model.to(opt.device)

def load_model(opt, model, optimizer, scheduler) :
    if not opt.continue_train and opt.mode == 'train' :
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

def plot_key_points(src, tgt, pred, occ_true, occ_pred, path) :
    if type(pred) != np.ndarray or type(src) != np.ndarray:
        src.numpy()
        tgt.numpy()
        pred.numpy()
        occ_true.numpy()
        occ_pred.numpy()


    # True drawing
    for p1, p2 in kpn.skeleton_tree: # line 그리기
        h1, w1 = tgt[p1]
        h2, w2 = tgt[p2]
        if occ_true[p1] == 1 or occ_true[p2] == 1 : continue # occlusion 제외
        plt.plot([w1, w2], [-h1, -h2], color='crimson')

    for (h_s, w_s), (h, w), occ, key in zip(src, tgt, occ_true, kpn.key_point_name) :
        if occ == 1 : continue
        if h_s != -1 and w_s != -1 :
            plt.scatter(w, -h, marker='o', c = 'g')
        else :
            plt.scatter(w, -h, marker='x', c = 'r')
        plt.text(w, -h, key)

    # Pred drawing
    for p1, p2 in kpn.skeleton_tree: # line 그리기
        h1, w1 = tgt[p1] if src[p1][0] != -1 else pred[p1]
        h2, w2 = tgt[p2] if src[p2][0] != -1 else pred[p2]
        if occ_pred[p1] == 1 or occ_pred[p2] == 1 : continue # occlusion 제외
        plt.plot([w1+150, w2+150], [-h1, -h2], color='green')

    for src_p, true_p, pred_p, occ_p, occ_t, key in zip(src, tgt, pred, occ_pred, occ_true, kpn.key_point_name) :
        if occ_p == 1: continue
        h, w = true_p if src_p[0] != -1 else pred_p
        if occ_p == occ_t :
            plt.scatter(w+150, -h, marker='^', color='blue')
        else :
            plt.scatter(w + 150, -h, marker='^', color='red')
        plt.text(w+150, -h, key)

    plt.savefig(path)
    plt.cla()

def single_plot_key_points(points, loss)  :
    if type(points) != np.ndarray:
        points.numpy()

    for p1, p2 in kpn.skeleton_tree:
        h1, w1 = points[p1]
        h2, w2 = points[p2]
        plt.plot([w1, w2], [-h1, -h2], color='crimson')
    plt.title(f'Pose loss : {loss.tolist()}' )
    plt.show()
if __name__ == "__main__":
    train_path = 'dataset/train/pose_label.pkl'
    test_path = 'dataset/test/fasion-annotation-test.csv'
    train_X = load_train_data(train_path)
    test_X = load_test_data(test_path)
