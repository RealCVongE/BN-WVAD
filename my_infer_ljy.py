import torch
import numpy as np
from dataset_loader_ljy import myVideo
from options import parse_args
import pdb
import utils
import os
from models.my_model import WSAD
from tqdm import tqdm
from dataset_loader_my import data
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
    if args.seed >= 0:
        utils.set_seed(args.seed)
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

if __name__ == "__main__":
    args = parse_args()
    net =wvad_model_load(args)
    res = infer(args,net)
    print(res)
