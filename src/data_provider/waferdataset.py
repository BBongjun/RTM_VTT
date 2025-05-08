import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tqdm import tqdm
import os
import h5py

"""
GDN에 적합하도록 WaferDataset을 변환한 데이터셋 클래스
dataset_preprocessing.ipynb에서 생성하여, h5 파일 이용
"""
class WaferDataset(Dataset):
    def __init__(self, file_list, model_type='reconstruction'):
        self.file_list = file_list
        self.model_type = model_type

        all_data, all_next_steps, all_labels = [], [], []
        self.lotids = []
        self.wafer_numbers = []
        self.step_nums = []

        for file_path in tqdm(file_list, desc="Loading and merging data"):
            with h5py.File(file_path, 'r') as f:
                all_data.append(f['data'][:].astype(float))
                all_next_steps.append(f['next_step'][:].astype(float))
                all_labels.append(f['labels'][:])

                self.lotids.extend(f['lotids'][:].astype(str))
                self.wafer_numbers.extend(f['wafer_numbers'][:].astype(str))
                self.step_nums.extend(f['step_num'][:])

        self.all_data = np.concatenate(all_data, axis=0)
        self.all_next_steps = np.concatenate(all_next_steps, axis=0)
        self.all_labels = np.concatenate(all_labels, axis=0)
        self.n_sensor = self.all_data.shape[2]

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        window_data = self.all_data[idx] # (window_size, n_features)
        y = self.all_next_steps[idx]
        label = self.all_labels[idx]
        lotid = self.lotids[idx]
        wafer_number = self.wafer_numbers[idx]
        step_num = int(self.step_nums[idx])
        
        item = {
            'given': window_data
        }
        
        if self.model_type == 'reconstruction':
            # item["ts"] = self.ts[i:last]
            item["answer"] = window_data
            # if self.with_attack:
            #     item['attack'] = self.attacks[i:last]
        elif self.model_type == 'prediction':
            # item["ts"] = self.ts[last]
            item["answer"] = y
            # if self.with_attack:
            #     item['attack'] = self.attacks[last]
        return item

def get_dataloader(data_info, loader_params: dict):
    """
    GDN 모델에 적합한 DataLoader 생성 함수 (train/val split 포함)

    Args:
        data_info (dict): 'train_dir' 키 포함
        loader_params (dict): batch_size, use_val 등 포함

    Returns:
        tuple: (train_loader, val_loader or None, test_loader)
    """
    # 전체 train 파일 리스트
    all_train_files = sorted([
        os.path.join(data_info['train_dir'], f)
        for f in os.listdir(data_info['train_dir'])
        if f.endswith(".h5")
    ])

    # validation 포함 여부
    if loader_params.get('use_val', False):
        split_idx = int(len(all_train_files) * 0.9)
        train_files = all_train_files[:split_idx]
        val_files = all_train_files[split_idx:]

        val_dataset = WaferDataset(val_files)
        val_loader = DataLoader(val_dataset,
                                batch_size=loader_params['batch_size'],
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=False)
    else:
        train_files = all_train_files
        val_loader = None

    # test 파일 리스트 (필요시 별도 dir로 교체 가능)
    test_files = sorted([
        os.path.join(data_info['test_dir'], f)
        for f in os.listdir(data_info['test_dir'])
        if f.endswith(".h5")
    ])

    # Dataset & DataLoader
    train_dataset = WaferDataset(train_files)
    test_dataset = WaferDataset(test_files)

    train_loader = DataLoader(train_dataset,
                              batch_size=loader_params['batch_size'],
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=loader_params['batch_size'],
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False)

    return train_loader, val_loader, test_loader
