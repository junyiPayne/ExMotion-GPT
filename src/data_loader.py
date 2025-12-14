import numpy as np
import pandas as pd
import torch
from scipy import signal
from torch.utils.data import Dataset

class DataProcessor:
    def __init__(self, fs=200, window_ms=500, overlap_ms=450):
        """
        数据处理管线 [cite: 300-302]
        """
        self.fs = fs
        self.window = int(fs * window_ms / 1000)
        self.step = int(fs * (window_ms - overlap_ms) / 1000)

    def apply_filters(self, data):
        # 滤波: 50Hz 陷波 + 20-95Hz 带通 [cite: 307-313]
        b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=self.fs)
        data = signal.filtfilt(b_notch, a_notch, data, axis=0)
        
        nyquist = 0.5 * self.fs
        b_band, a_band = signal.butter(4, [20.0/nyquist, 95.0/nyquist], btype='bandpass')
        return signal.filtfilt(b_band, a_band, data, axis=0)

    def normalize(self, data, mvic=None):
        # Z-Score 标准化 [cite: 314-322]
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True) + 1e-8
        z = (data - mean) / std
        if mvic is not None:
            z = z / (mvic + 1e-6)
        return z

    def segment(self, data, labels):
        # 滑动窗口切片 [cite: 323-333]
        segments, seg_labels = [], []
        for start in range(0, data.shape[0] - self.window + 1, self.step):
            end = start + self.window
            window = data[start:end]
            majority = np.bincount(labels[start:end]).argmax()
            segments.append(window)
            seg_labels.append(majority)
        
        X = np.stack(segments).transpose(0, 2, 1) # (N, C, L)
        y = np.array(seg_labels)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def compute_baseline_features(window):
    """
    计算手工特征 (Hybrid Features) [cite: 342-350]
    """
    # 时域特征
    rms = torch.sqrt(torch.mean(window ** 2, dim=-1))
    ptp = window.max(dim=-1).values - window.min(dim=-1).values
    rss = torch.sqrt(torch.sum(window ** 2, dim=-1))
    
    # 频域特征 (简化版，实际应用需做FFT)
    # 这里为了代码可运行，使用简单的统计特征代替
    mean_val = window.mean(dim=-1)
    std_val = window.std(dim=-1)
    
    # 堆叠特征
    return torch.stack([rms, ptp, rss, mean_val, std_val], dim=-1)

class SlidingWindowDataset(Dataset):
    def __init__(self, X_raw, y, use_features=True):
        """
        支持混合特征的数据集 [cite: 351-366]
        """
        self.X_raw = X_raw
        self.y = y
        self.use_features = use_features
        if use_features:
            # 预计算手工特征
            self.X_feat = compute_baseline_features(X_raw)
            
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        sample = {"raw": self.X_raw[idx], "label": self.y[idx]}
        if self.use_features:
            sample["feat"] = self.X_feat[idx]
        return sample