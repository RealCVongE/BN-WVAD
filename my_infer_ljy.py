import torch
import numpy as np
# from options import parse_args
import utils
import os
from .models.my_model import WSAD
from tqdm import tqdm
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import prettytable
from torch.utils.data import DataLoader

def get_predict(_data, net):
    frame_predict = []
    res = net(_data)   
    
    a_predict = res.cpu().numpy().mean(0)   

    fpre_ = np.repeat(a_predict, 16)
    frame_predict.append(fpre_)

    frame_predict = np.concatenate(frame_predict, axis=0)
    return frame_predict


def test(net, _data):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        frame_predict = get_predict(_data, net)


        return frame_predict
def wvad_model_load(args):
    net = WSAD(args.len_feature, flag = "Test", args = args)
    net = net.cuda()
    net.load_state_dict(torch.load(args.model_path))
    return net

def infer(args,net):
    video_feature = np.load(args.root_dir).astype(np.float32)
    video_feature= np.expand_dims(video_feature,axis=0)
    video_feature =torch.from_numpy(video_feature).float().to("cuda:0")
    res = test(net, video_feature )
    return res
def infer_numpy(net,video_feature):
    video_feature= np.expand_dims(video_feature,axis=0)
    video_feature =torch.from_numpy(video_feature).float().to("cuda:0")
    res = test(net, video_feature )
    return res
from types import SimpleNamespace
import os

def parse_args():
    descript = 'Pytorch Implementation of UR-DMU'
    args = SimpleNamespace(
        len_feature=1024,
        root_dir='xd/',
        log_path='/home/bigdeal/mnt2/workspace/mmaction2/bn_wvad/logs/',
        model_path='/home/bigdeal/mnt2/BN-WVAD/ckpts/2024-04-03/my/train/my_best_2022_over.pkl',
        lr='[0.0001]*1000',
        batch_size=64,
        num_workers=4,
        num_segments=200,
        seed=2022,
        debug=False,
        processed=False,
        plot_freq=5,
        weight_decay=0.00005,
        version='train',
        ratio_sample=0.2,
        ratio_batch=0.4,
        ratios=[16, 32],
        kernel_sizes=[1, 1, 1]
    )
    return init_args(args)

def init_args(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    args.lr = eval(args.lr)
    args.num_iters = len(args.lr)
    return args

if __name__ == "__main__":
    args = parse_args()
    net =wvad_model_load(args)
    res = infer(args,net)
    print(res)
