import torch
import torch.utils.data as data
import os
import numpy as np
import utils 

class myVideo(data.DataLoader):
    def __init__(self, file_path, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.file_path=file_path
        self.mode=mode
        self.num_segments = num_segments
        self.len_feature = len_feature
        
        self.feature_path = self.file_path
        # split_path = os.path.join("list",'my_{}.list'.format(self.mode))
        # split_path = os.path.join("list",'my_{}.list'.format(self.mode))
        # split_file = open(split_path, 'r',encoding="utf-8")

        # split_file = root_dir
        # self.vid_list = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')]
        # split_file.close()

        self.vid_list = [file_path]

        if self.mode == "Train":
            if is_normal is True:
                # self.vid_list = self.vid_list[9525:]
                self.vid_list = self.vid_list[641:]
            elif is_normal is False:
                # self.vid_list = self.vid_list[:9525]
                self.vid_list = self.vid_list[:641]
            else:
                assert (is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.vid_list=[]
        
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data,label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid_path = self.vid_list[index]
        label=0
        if "normal" not in vid_path:
            label=1 

        #video_feature = np.load(os.path.join(self.feature_path, vid_name )).astype(np.float32)
        # if isinstance(vid_name, list):
        #     vid_name = vid_name[0]  # 리스트의 첫 번째 요소를 선택

        video_feature = np.load(vid_path).astype(np.float32)

        assert len(video_feature.shape)==2, f"video_feature.shape: {video_feature.shape}, vid_name: {vid_name}"
        assert video_feature.shape[1]==1024
        # assert video_feature.shape[0]==11, f"video_feature.shape: {video_feature.shape}, vid_name: {vid_name}"
        # video_feature = np.load(vid_name).astype(np.float32)
        if self.mode == "Train":
            new_feature = np.zeros((self.num_segments, self.len_feature)).astype(np.float32)

            sample_index = np.linspace(0, video_feature.shape[0], self.num_segments+1, dtype=np.uint16)

            for i in range(len(sample_index)-1):
                if sample_index[i] == sample_index[i+1]:
                    new_feature[i,:] = video_feature[sample_index[i],:]
                else:
                    new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0)
                    
            video_feature = new_feature
        return video_feature, label    