import torch
import numpy as np
from dataset_loader import XDVideo
from dataset_loader_ucf import UCFVideo
from options import parse_args
import pdb
import utils
import os
from models import WSAD
from tqdm import tqdm
from dataset_loader import data
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import prettytable

def get_predict(test_loader, net):
    """
    테스트 데이터셋에서 예측값을 추출하고 배열로 반환합니다.

    Args:
        test_loader: 테스트 데이터 로더
        net: 신경망 모델

    Returns:
        frame_predict: 비디오 프레임별 예측값 배열
    """

    # 테스트 데이터 로더를 반복적으로 처리
    load_iter = iter(test_loader)
    frame_predict = []
    
    
    for i in range(len(test_loader.dataset)//5):
        # 데이터 및 레이블 배치 가져오기
        _data, _label = next(load_iter)

        # 데이터 및 레이블을 GPU로 전송
        _data = _data.cuda()
        _label = _label.cuda()

        #(5,67,1024)
        #(5,120,1024)
        # 데이터를 신경망 모델에 입력
        
        #(5,67) 점수가 나옴
        res = net(_data)   

        # 출력을 NumPy 배열로 변환하고 세그먼트 평균 계산
        a_predict = res.cpu().numpy().mean(0)   

        # 각 비디오 프레임에 대한 예측값 생성
        fpre_ = np.repeat(a_predict, 16)
        frame_predict.append(fpre_)

    # 예측값 배열을 연결 xd 데이터셋에서 800개 컨켓 => 2330384
    frame_predict = np.concatenate(frame_predict, axis=0)
    return frame_predict

def get_sub_metrics(frame_predict, frame_gt):
    """
    이상 윈도우에 속하는 프레임에 대한 성능 지표를 계산합니다.

    Args:
        frame_predict: 비디오 프레임별 예측값 배열
        frame_gt: 비디오 프레임별 실제 레이블 배열

    Returns:
        auc_sub: 이상 윈도우 AUC
        ap_sub: 이상 윈도우 AP
    """

    # 이상 윈도우 마스크 로드
    anomaly_mask = np.load('frame_label/anomaly_mask_ucf.npy')
    sub_predict = frame_predict[anomaly_mask]
    sub_gt = frame_gt[anomaly_mask]

    # ROC-AUC 계산
    fpr, tpr, _ = roc_curve(sub_gt, sub_predict)
    auc_sub = auc(fpr, tpr)

    # Average Precision 계산
    precision, recall, th = precision_recall_curve(sub_gt, sub_predict)
    ap_sub = auc(recall, precision)
    return auc_sub, ap_sub

def get_metrics(frame_predict, frame_gt):
    """
    전체 프레임 및 이상 윈도우에 대한 성능 지표를 계산합니다.

    Args:
        frame_predict: 비디오 프레임별 예측값 배열
        frame_gt: 비디오 프레임별 실제 레이블 배열

    Returns:
        metrics: 성능 지표 딕셔너리
    """

    # 성능 지표 딕셔너리 생성
    metrics = {}

    # 전체 프레임 ROC-AUC 계산
    fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
    metrics['AUC'] = auc(fpr, tpr)

    # 전체 프레임 Average Precision 계산
    precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
    metrics['AP'] = auc(recall, precision)

    # 이상 윈도우 성능 지표 계산
    auc_sub, ap_sub = get_sub_metrics(frame_predict, frame_gt)
    metrics['AUC_sub'] = auc_sub
    metrics['AP_sub'] = ap_sub

    return metrics
def test(net, test_loader, test_info, step, model_file = None):
    """
    모델을 평가하고 성능 지표를 출력합니다.

    Args:
        net: 신경망 모델
        test_loader: 테스트 데이터 로더
        test_info: 평가 결과 저장 딕셔너리
        step: 현재 학습 스텝
        model_file: 모델 파일 경로 (선택 사항)

    Returns:
        metrics: 평가 결과 딕셔너리
    """

    with torch.no_grad():
        # 평가 모드 설정
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            # 모델 파일 로드
            net.load_state_dict(torch.load(model_file))

        # 실제 레이블 로드
        frame_gt = np.load("frame_label/gt-ucf.npy")

        # 예측값 추출
        frame_predict = get_predict(test_loader, net)

        # 성능 지표 계산
        metrics = get_metrics(frame_predict, frame_gt)

        # 평가 결과 저장
        test_info['step'].append(step)
        for score_name, score in metrics.items():
            metrics[score_name] = score * 100
            test_info[score_name].append(metrics[score_name])

        return metrics

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
    test_loader = data.DataLoader(
        UCFVideo(root_dir = args.root_dir, mode = 'Test', num_segments = args.num_segments, len_feature = args.len_feature),
            batch_size = 5,
            shuffle = False, num_workers = args.num_workers,
            worker_init_fn = worker_init_fn)
    
    test_info = {'step': [], 'AUC': [], 'AUC_sub': [], 'AP': [], 'AP_sub': []}

    res = test(net, test_loader, test_info, 1, model_file = args.model_path)

    pt = prettytable.PrettyTable()
    pt.field_names = ['AUC', 'AUC_sub', 'AP', 'AP_sub']
    for k, v in res.items():
        res[k] = round(v, 2)
    pt.add_row([res['AUC'], res['AUC_sub'], res['AP'], res["AP_sub"]])
    print(pt)