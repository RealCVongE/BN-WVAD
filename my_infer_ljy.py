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

def get_predict(test_loader, net):
    # load_iter = iter(test_loader.dataset)
    frame_predict = []
        
    # for i in range(len(test_loader)):
    #     _data, _label = next(load_iter)
    for _data, _label in test_loader:

        _data = _data.cuda()
        _label = _label.cuda()
        res = net(_data)   
        
        a_predict = res.cpu().numpy().mean(0)   

        fpre_ = np.repeat(a_predict, 16)
        frame_predict.append(fpre_)

    frame_predict = np.concatenate(frame_predict, axis=0)
    return frame_predict


def test(net, test_loader, step, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))
        
        frame_predict = get_predict(test_loader, net)


        return frame_predict


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()
    worker_init_fn = None
    if args.seed >= 0:
        utils.set_seed(args.seed)
        worker_init_fn = np.random.seed(args.seed)
    net = WSAD(args.len_feature, flag = "Test", args = args)
    net = net.cuda()
    
    # test_loader = data.DataLoader(
    #     myVideo(file_path = args.root_dir, mode = 'Test', num_segments = args.num_segments, len_feature = args.len_feature),
    #         # batch_size = 5,
    #         batch_size = 1,
    #         shuffle = False, num_workers = args.num_workers,
    #         worker_init_fn = worker_init_fn)

    # myVideo 인스턴스 생성
    dataset = myVideo(file_path = args.root_dir, mode='Test', num_segments=10, len_feature=1024)

    # DataLoader 생성
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)


    res = test(net, test_loader, 1, model_file = args.model_path)

    print(res)
