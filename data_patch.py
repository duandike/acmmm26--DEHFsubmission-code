import os
import numpy as np
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """
    Dataset: 以 Lead II 的 min-max 版本作为条件输入，
    以 12 导联整体 max-abs 域信号作为监督目标。

    目标 12 导联顺序固定为：
        [Lead II, Lead I, Lead III, Lead aVR, Lead aVL, Lead aVF,
         Lead V1, Lead V2, Lead V3, Lead V4, Lead V5, Lead V6]

    Args:
        leadII_input_minmax: np.ndarray, shape (N, T)
        target_12lead_maxabs: np.ndarray, shape (N, T, 12)
        sampling_rate: 采样率(Hz)
    """
    def __init__(self, leadII_input_minmax, target_12lead_maxabs, sampling_rate: int = 128):
        self.leadII_input_minmax = np.asarray(leadII_input_minmax, dtype=np.float32)
        self.target_12lead_maxabs = np.asarray(target_12lead_maxabs, dtype=np.float32)
        self.sampling_rate = int(sampling_rate)

        if self.leadII_input_minmax.ndim != 2:
            raise ValueError(f"leadII_input_minmax 应为 (N,T)，当前 shape={self.leadII_input_minmax.shape}")
        if self.target_12lead_maxabs.ndim != 3 or self.target_12lead_maxabs.shape[-1] != 12:
            raise ValueError(f"target_12lead_maxabs 应为 (N,T,12)，当前 shape={self.target_12lead_maxabs.shape}")
        if self.leadII_input_minmax.shape[0] != self.target_12lead_maxabs.shape[0]:
            raise ValueError("输入与目标的样本数不一致")
        if self.leadII_input_minmax.shape[1] != self.target_12lead_maxabs.shape[1]:
            raise ValueError("输入与目标的时间长度不一致")

    def __getitem__(self, index):
        leadII = self.leadII_input_minmax[index]      # (T,)
        full12 = self.target_12lead_maxabs[index]     # (T, 12)
        window_size = leadII.shape[-1]
        ecg_roi_array = np.zeros((1, window_size), dtype=np.float32)

        return (
            leadII.reshape(1, window_size).astype(np.float32).copy(),
            full12.T.astype(np.float32).copy(),
            ecg_roi_array.copy(),
        )

    def __len__(self):
        return len(self.leadII_input_minmax)



def _ensure_eleven_shape_nt11(eleven: np.ndarray) -> np.ndarray:
    """统一 eleven 为 (N, T, 11)。"""
    eleven = np.nan_to_num(eleven.astype(np.float32))

    if eleven.ndim != 3:
        raise ValueError(f"eleven 应该是 3 维，当前 shape={eleven.shape}")

    if eleven.shape[-1] == 11:
        return np.ascontiguousarray(eleven, dtype=np.float32)
    if eleven.shape[1] == 11:
        return np.ascontiguousarray(np.transpose(eleven, (0, 2, 1)), dtype=np.float32)

    raise ValueError(f"Unexpected shape for eleven: {eleven.shape}")



def _merge_to_12leads(leadII: np.ndarray, eleven: np.ndarray) -> np.ndarray:
    """
    输入:
        leadII: (N, T)
        eleven: (N, T, 11)
    输出:
        full12: (N, T, 12)

    导联顺序固定为：
        [Lead II, Lead I, Lead III, Lead aVR, Lead aVL, Lead aVF,
         Lead V1, Lead V2, Lead V3, Lead V4, Lead V5, Lead V6]
    """
    leadII = np.nan_to_num(leadII.astype(np.float32))
    eleven = np.nan_to_num(eleven.astype(np.float32))

    if leadII.ndim != 2:
        raise ValueError(f"leadII 应该是 (N,T)，当前 shape={leadII.shape}")
    if eleven.ndim != 3:
        raise ValueError(f"eleven 应该是 (N,T,11)，当前 shape={eleven.shape}")

    N, T = leadII.shape
    N2, T2, C = eleven.shape
    if not (N == N2 and T == T2 and C == 11):
        raise ValueError(
            f"leadII 与 eleven 维度不匹配: leadII={leadII.shape}, eleven={eleven.shape}"
        )

    full12 = np.concatenate([leadII[..., None], eleven], axis=-1)
    return np.ascontiguousarray(full12, dtype=np.float32)



def _normalize_12leads_per_sample_maxabs(x12: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """对每个样本的 12 导联整体做 max-abs 归一化。"""
    if x12.ndim != 3:
        raise ValueError(f"x12 应该是 (N,T,12)，当前 shape={x12.shape}")
    if x12.shape[-1] != 12:
        raise ValueError(f"x12 最后一维应为 12，当前 shape={x12.shape}")

    x12 = np.nan_to_num(x12.astype(np.float32, copy=False))
    N = x12.shape[0]
    flat = x12.reshape(N, -1)
    scale = np.max(np.abs(flat), axis=1, keepdims=True)
    scale = np.maximum(scale, eps)
    flat_scaled = flat / scale
    x12_norm = flat_scaled.reshape(x12.shape)
    return np.ascontiguousarray(x12_norm, dtype=np.float32)



def _minmax_scale_per_sample(x: np.ndarray, feature_range=(-1.0, 1.0), eps: float = 1e-8) -> np.ndarray:
    """对每个样本的一维时序做 min-max 到指定区间。"""
    if x.ndim != 2:
        raise ValueError(f"x 应该是 (N,T)，当前 shape={x.shape}")

    x = np.nan_to_num(x.astype(np.float32, copy=False))
    low, high = feature_range

    x_min = np.min(x, axis=1, keepdims=True).astype(np.float32)
    x_max = np.max(x, axis=1, keepdims=True).astype(np.float32)
    denom = x_max - x_min

    x_scaled = np.zeros_like(x, dtype=np.float32)
    safe = denom > eps
    if np.any(safe):
        idx = safe[:, 0]
        x01 = (x[idx] - x_min[idx]) / denom[idx]
        x_scaled[idx] = x01 * (high - low) + low

    return np.ascontiguousarray(x_scaled, dtype=np.float32)



def _extract_leadII_from_12(x12: np.ndarray) -> np.ndarray:
    if x12.ndim != 3 or x12.shape[-1] != 12:
        raise ValueError(f"x12 应该是 (N,T,12)，当前 shape={x12.shape}")
    return np.ascontiguousarray(x12[:, :, 0], dtype=np.float32)



def _load_one_dataset_processed(dataset_dir: str):
    """
    加载单个数据集，并完成：
        1) 读取原始 npy
        2) eleven 统一为 (N,T,11)
        3) merge -> (N,T,12)
        4) 每样本 12 导联整体 max-abs 归一化
        5) 取归一化后的完整 12 导联作为监督目标
        6) 仅对 Lead II 再做 min-max 到 [-1,1]，作为条件输入
    """
    leadII_train = np.load(os.path.join(dataset_dir, "II_train.npy"), allow_pickle=True)
    leadII_test = np.load(os.path.join(dataset_dir, "II_test.npy"), allow_pickle=True)

    eleven_train = np.load(os.path.join(dataset_dir, "eleven_train.npy"), allow_pickle=True)
    eleven_test = np.load(os.path.join(dataset_dir, "eleven_test.npy"), allow_pickle=True)

    leadII_train = np.nan_to_num(leadII_train.astype(np.float32))
    leadII_test = np.nan_to_num(leadII_test.astype(np.float32))
    eleven_train = _ensure_eleven_shape_nt11(eleven_train)
    eleven_test = _ensure_eleven_shape_nt11(eleven_test)

    full12_train = _merge_to_12leads(leadII_train, eleven_train)
    full12_test = _merge_to_12leads(leadII_test, eleven_test)

    full12_train = _normalize_12leads_per_sample_maxabs(full12_train)
    full12_test = _normalize_12leads_per_sample_maxabs(full12_test)

    leadII_train_cond = _minmax_scale_per_sample(_extract_leadII_from_12(full12_train), feature_range=(-1.0, 1.0))
    leadII_test_cond = _minmax_scale_per_sample(_extract_leadII_from_12(full12_test), feature_range=(-1.0, 1.0))

    return leadII_train_cond, full12_train, leadII_test_cond, full12_test



def get_datasets(
    DATA_PATH="/root/autodl-tmp",
    datasets=("splits",),
    window_size: int = 4,
    sampling_rate: int = 128,
):
    """
    固定接口：
        输入  : Lead II 的 min-max 版本，shape -> (1, T)
        输出  : 12 导联 max-abs 目标，shape -> (12, T)

    数据流程：
        1) II + 11 导联合并为 12 导联，顺序固定为
           [II, I, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
        2) 对每个样本整体做 12 导联 max-abs 归一化
        3) 归一化后的完整 12 导联作为生成目标
        4) 仅将归一化后的 Lead II 再做 min-max 到 [-1,1]，作为条件输入
    """
    _ = window_size

    train_leadII_list = []
    train_full12_list = []
    test_leadII_list = []
    test_full12_list = []

    for dataset_name in datasets:
        dataset_dir = os.path.join(DATA_PATH, dataset_name)
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"数据目录不存在: {dataset_dir}")

        leadII_train_cond, full12_train, leadII_test_cond, full12_test = _load_one_dataset_processed(dataset_dir)
        train_leadII_list.append(leadII_train_cond)
        train_full12_list.append(full12_train)
        test_leadII_list.append(leadII_test_cond)
        test_full12_list.append(full12_test)

    leadII_train_all = np.concatenate(train_leadII_list, axis=0)
    full12_train_all = np.concatenate(train_full12_list, axis=0)
    leadII_test_all = np.concatenate(test_leadII_list, axis=0)
    full12_test_all = np.concatenate(test_full12_list, axis=0)

    dataset_train = ECGDataset(leadII_train_all, full12_train_all, sampling_rate=sampling_rate)
    dataset_test = ECGDataset(leadII_test_all, full12_test_all, sampling_rate=sampling_rate)
    return dataset_train, dataset_test
