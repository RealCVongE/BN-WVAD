import torch
import torch.utils.data as data
import os
import numpy as np
import utils

class XDVideo(data.DataLoader):
    def __init__(self, root_dir, mode, num_segments, len_feature, seed=-1, is_normal=None):
        # 랜덤 시드 설정
        if seed >= 0:
            utils.set_seed(seed)

        # 데이터 경로 및 모드 저장
        self.data_path=root_dir
        self.mode=mode
        self.num_segments = num_segments
        self.len_feature = len_feature

        # 특징 경로 저장 및 분할 파일로부터 비디오 목록 로드
        self.feature_path = self.data_path
        split_path = os.path.join("list",'XD_{}.list'.format(self.mode))
        split_file = open(split_path, 'r',encoding="utf-8")
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()

        # 훈련 모드에서 이상 탐지 훈련을 위한 데이터셋 조정
        if self.mode == "Train":
            if is_normal is True:
                # 후반부 데이터 (정상 데이터) 사용
                self.vid_list = self.vid_list[9525:]
            elif is_normal is False:
                # 초반부 데이터 (이상 데이터) 사용
                self.vid_list = self.vid_list[:9525]
            else:
                # 'is_normal' 값이 지정되지 않으면 에러 발생
                print("Please sure is_normal = [True/False]")
                self.vid_list = []

    def __len__(self):
        # 데이터셋에 포함된 비디오 수 반환
        return len(self.vid_list)

    def __getitem__(self, index):
        # 특정 인덱스에 해당하는 비디오 특징 및 레이블 로드
        data, label = self.get_data(index)
        return data, label

    def get_data(self, index):
        # 비디오 이름 가져오기
        vid_name = self.vid_list[index][0]

        # 비디오 이름을 기반으로 레이블 결정 (0 = 정상, 1 = 이상)
        label = 0
        if "_label_A" not in vid_name:
            label = 1
        # 파일로부터 비디오 특징 로드
        video_feature = np.load(os.path.join(self.feature_path, vid_name)).astype(np.float32)

        # 훈련 모드 전처리: 특징을 세그먼트 기반으로 샘플링
        if self.mode == "Train":  # 훈련 모드에서만 실행
            new_feature = np.zeros((self.num_segments, self.len_feature)).astype(np.float32)
            # 세그먼트 수와 특징 길이에 맞춰 새로운 배열 생성

            sample_index = np.linspace(0, video_feature.shape[0], self.num_segments+1, dtype=np.uint16)
            # 비디오 특징 길이를 세그먼트 수 + 1개의 지점으로 균등하게 분할하는 인덱스 생성

            for i in range(len(sample_index)-1):  # 각 세그먼트에 대해
                if sample_index[i] == sample_index[i+1]:
                    # 샘플링 결과 동일한 인덱스가 발생하는 경우
                    new_feature[i,:] = video_feature[sample_index[i],:]
                    # 동일한 인덱스의 특징 벡터를 사용

                else:
                    # 일반적인 경우: 두 인덱스 사이의 특징 평균 사용
                    new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0)

            video_feature = new_feature  # 원본 특징을 세그먼트 기반 특징으로 대체

        return video_feature, label    