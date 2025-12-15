import copy

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import hashlib
from modelscope import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import re
from collections import Counter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import os
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from Bio import SeqIO
from tqdm import tqdm
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import json


# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)

# 创建必要文件夹
for dir_name in ['results', 'checkpoints', 'ha_embeddings']:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# ==================== 生物学语言模型工具类 ====================
class BiologicalLanguageModel:
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", device=None, cache_dir=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir, trust_remote_code=True
            )
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name, cache_dir=cache_dir, trust_remote_code=True, output_hidden_states=True
            ).to(self.device)
            self.embedding_dim = self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else 320
        except:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "facebook/esm2_t12_35M_UR50D", cache_dir=cache_dir, trust_remote_code=True
                )
                self.model = AutoModelForMaskedLM.from_pretrained(
                    "facebook/esm2_t12_35M_UR50D", cache_dir=cache_dir, trust_remote_code=True,
                    output_hidden_states=True
                ).to(self.device)
                self.embedding_dim = self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else 480
            except:
                raise RuntimeError("无法加载预训练模型")
        self.model.eval()

    def get_embedding(self, sequence, pooling="mean"):
        seq = self._preprocess(sequence)
        if not seq:
            return np.zeros(self.embedding_dim)
        try:
            inputs = self.tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hs = outputs.hidden_states[-1]
            elif hasattr(outputs, 'last_hidden_state'):
                hs = outputs.last_hidden_state
            else:
                hs = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
            if pooling == "mean":
                embed = hs.mean(dim=1).squeeze().cpu().numpy()
            elif pooling == "max":
                embed = hs.max(dim=1).values.squeeze().cpu().numpy()
            elif pooling == "cls":
                embed = hs[:, 0, :].squeeze().cpu().numpy()
            else:
                return np.zeros(self.embedding_dim)
            return embed[:self.embedding_dim] if embed.shape[0] > self.embedding_dim else np.pad(embed, (
                0, self.embedding_dim - len(embed)))
        except:
            return np.zeros(self.embedding_dim)

    def _preprocess(self, seq):
        if not isinstance(seq, str) or len(seq) < 10:
            return ""
        valid_aas = "ACDEFGHIKLMNPQRSTVWY"
        return "".join([aa for aa in seq.upper() if aa in valid_aas])

    def batch_get_embeddings(self, sequences, batch_size=8, pooling="mean"):
        embeds = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="提取序列嵌入"):
            batch = sequences[i:i + batch_size]
            batch_embeds = [self.get_embedding(s, pooling) for s in batch]
            embeds.extend(batch_embeds)
        return np.array(embeds)


def process_ha_sequences_with_plm(ha_df, model_name="facebook/esm2_t6_8M_UR50D", cache_dir="ha_embeddings"):
    """处理HA序列，提取PLM嵌入特征"""
    os.makedirs(cache_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ha_df.empty:
        print("警告: 输入的HA DataFrame为空")
        return pd.DataFrame(columns=['sequence', 'sample_date', 'embedding'])

    if 'sequence' not in ha_df.columns or 'sample_date' not in ha_df.columns:
        print("警告: HA DataFrame缺少必要列")
        return pd.DataFrame(columns=['sequence', 'sample_date', 'embedding'])

    try:
        blm = BiologicalLanguageModel(model_name=model_name, device=device)
        print(f"成功加载语言模型，嵌入维度: {blm.embedding_dim}")
    except Exception as e:
        print(f"加载语言模型失败: {e}")
        return pd.DataFrame(columns=['sequence', 'sample_date', 'embedding'])

    embeddings, valid_seqs, valid_dates = [], [], []

    if not pd.api.types.is_datetime64_any_dtype(ha_df['sample_date']):
        ha_df['sample_date'] = pd.to_datetime(ha_df['sample_date'], errors='coerce')

    ha_df = ha_df.dropna(subset=['sample_date', 'sequence'])

    if ha_df.empty:
        print("警告: 过滤缺失值后HA DataFrame为空")
        return pd.DataFrame(columns=['sequence', 'sample_date', 'embedding'])

    print(f"开始处理 {len(ha_df)} 条HA序列...")

    for _, row in tqdm(ha_df.iterrows(), total=len(ha_df), desc="处理HA序列"):
        seq, date = row['sequence'], row['sample_date']

        if not is_valid_sequence(seq, "amino_acid"):
            continue

        seq_md5 = hashlib.md5(seq.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{seq_md5}.npy")

        if os.path.exists(cache_path):
            try:
                embed = np.load(cache_path)
                if embed.shape[0] == blm.embedding_dim:
                    embeddings.append(embed)
                    valid_seqs.append(seq)
                    valid_dates.append(date)
                    continue
                else:
                    print(f"缓存嵌入维度不匹配: {embed.shape[0]} != {blm.embedding_dim}")
            except Exception as e:
                print(f"加载缓存失败: {e}")

        aa_seq = seq.upper()
        if len(aa_seq) < 50:
            continue

        try:
            embed = blm.get_embedding(aa_seq)
            if embed is not None and embed.shape[0] == blm.embedding_dim:
                np.save(cache_path, embed)
                embeddings.append(embed)
                valid_seqs.append(seq)
                valid_dates.append(date)
            else:
                print(f"嵌入提取失败或维度不匹配: {embed.shape if embed is not None else 'None'}")
        except Exception as e:
            print(f"处理序列失败: {e}")

    print(f"成功处理 {len(embeddings)}/{len(ha_df)} 条HA序列的嵌入特征")

    result_df = pd.DataFrame({
        'sequence': valid_seqs,
        'sample_date': valid_dates,
        'embedding': embeddings
    })

    return result_df


def check_ha_data_quality(ha_processed):
    """检查HA数据质量"""
    if ha_processed.empty:
        print("警告: 没有HA序列数据")
        return False

    print("\n=== HA数据质量检查 ===")
    print(f"总HA序列数: {len(ha_processed)}")
    print(f"时间范围: {ha_processed['sample_date'].min()} 到 {ha_processed['sample_date'].max()}")

    embeddings = np.array(ha_processed['embedding'].tolist())
    print(f"嵌入形状: {embeddings.shape}")
    print(f"嵌入均值: {embeddings.mean():.6f} ± {embeddings.std():.6f}")
    print(f"零值比例: {np.mean(embeddings == 0):.3f}")

    seq_lengths = ha_processed['sequence'].str.len()
    print(f"序列长度: {seq_lengths.min()}-{seq_lengths.max()} 氨基酸")

    return len(ha_processed) > 0


def augment_ha_sequences(ha_processed, augmentation_factor=2):
    """HA序列数据增强（通过添加噪声）"""
    if ha_processed.empty:
        return ha_processed

    augmented_data = []
    for _, row in ha_processed.iterrows():
        original_embed = row['embedding']
        augmented_data.append({
            'sequence': row['sequence'],
            'sample_date': row['sample_date'],
            'embedding': original_embed
        })

        for i in range(augmentation_factor - 1):
            noise = np.random.normal(0, 0.01, original_embed.shape)
            augmented_embed = original_embed + noise
            augmented_data.append({
                'sequence': row['sequence'] + f"_aug_{i}",
                'sample_date': row['sample_date'],
                'embedding': augmented_embed
            })

    return pd.DataFrame(augmented_data)


# ==================== 辅助函数 ====================
def is_valid_sequence(seq, seq_type="amino_acid", min_len=100, max_len=2000):
    """增强版序列验证函数，允许最多5%的非有效氨基酸"""
    if not isinstance(seq, str) or len(seq) < min_len or len(seq) > max_len:
        return False
    if seq_type == "amino_acid":
        valid_aas = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                     'X'}
        invalid_count = sum(1 for c in seq.upper() if c not in valid_aas)
        invalid_ratio = invalid_count / len(seq)
        if invalid_ratio > 0.05:
            return False
    return True


def parse_fasta_date(date_str):
    """解析FASTA中的日期字符串，处理多种格式和仅年份的情况"""
    date_formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
        "%Y-%m", "%Y/%m", "%Y"
    ]

    for fmt in date_formats:
        try:
            date = pd.to_datetime(date_str, format=fmt, errors='coerce')
            if not pd.isna(date):
                if fmt == "%Y":
                    return date.replace(month=1, day=1)
                return date
        except:
            continue

    numbers = re.findall(r'\d+', date_str)
    if len(numbers) == 3:
        try:
            return pd.to_datetime(f"{numbers[0]}-{numbers[1]}-{numbers[2]}")
        except:
            pass
    elif len(numbers) == 2:
        try:
            return pd.to_datetime(f"{numbers[0]}-{numbers[1]}-01")
        except:
            pass
    elif len(numbers) == 1 and len(numbers[0]) == 4:
        try:
            return pd.to_datetime(f"{numbers[0]}-01-01")
        except:
            pass

    return pd.NaT


# ==================== 特征工程函数 ====================
def create_safe_features(df):
    """创建安全的特征 - 不依赖目标变量"""
    df_copy = df.copy()

    # 1. 时间特征（绝对安全）
    if 'time' in df_copy.columns:
        df_copy['year'] = df_copy['time'].dt.year
        df_copy['quarter'] = df_copy['time'].dt.quarter
        df_copy['month'] = df_copy['time'].dt.month
        df_copy['week'] = df_copy['time'].dt.isocalendar().week

        # 季节特征
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
        df_copy['week_sin'] = np.sin(2 * np.pi * df_copy['week'] / 52)
        df_copy['week_cos'] = np.cos(2 * np.pi * df_copy['week'] / 52)
        df_copy['is_peak_season'] = df_copy['month'].isin([1, 2, 12]).astype(int)
        # 季节性指标
        df_copy['is_winter'] = df_copy['month'].isin([12, 1, 2]).astype(int)
        df_copy['is_spring'] = df_copy['month'].isin([3, 4, 5]).astype(int)

    # 2. 环境特征衍生（安全）
    if 'temp_max' in df_copy.columns and 'temp_min' in df_copy.columns:
        df_copy['temp_range'] = df_copy['temp_max'] - df_copy['temp_min']

    return df_copy


def create_time_series_features(df, train_df=None):
    """创建时间序列特征 - 增强版"""
    df_copy = df.copy()

    # 1. 基础Lag特征 (短期)
    for i in range(1, 5):  # 增加到 Lag 4
        df_copy[f'ili_lag_{i}'] = df_copy['%UNWEIGHTED ILIS'].shift(i)

        # 2. 增强型年同比特征 (模糊匹配)
    # 不仅看52周前，还看51和53周前，防止季节性漂移
    # df_copy['ili_lag_51'] = df_copy['%UNWEIGHTED ILIS'].shift(51)
    df_copy['ili_lag_52'] = df_copy['%UNWEIGHTED ILIS'].shift(52)
    # df_copy['ili_lag_53'] = df_copy['%UNWEIGHTED ILIS'].shift(53)

    # 3. 关键改进：增加"速度"和"加速度"特征
    # 这能告诉模型：现在是在快速上升，还是在由升转降
    df_copy['velocity'] = df_copy['ili_lag_1'] - df_copy['ili_lag_2']
    df_copy['acceleration'] = (df_copy['ili_lag_1'] - df_copy['ili_lag_2']) - (
                df_copy['ili_lag_2'] - df_copy['ili_lag_3'])


    # 3. 两年周期特征 (Lag 104)
    # 很多流感毒株有大小年循环，看两年前的数据很有参考价值
    df_copy['ili_lag_104'] = df_copy['%UNWEIGHTED ILIS'].shift(104)

    # 4. 趋势特征
    # 计算最近4周的平均变化率，告诉模型现在是在上升期还是下降期
    # (Lag1 - Lag4) / 3
    df_copy['trend_short'] = (df_copy['ili_lag_1'] - df_copy['ili_lag_4']) / 3

    # 5. 移动平均
    df_copy['ili_ma4'] = df_copy['%UNWEIGHTED ILIS'].rolling(window=4, min_periods=1).mean()
    df_copy['ili_ma8'] = df_copy['%UNWEIGHTED ILIS'].rolling(window=8, min_periods=1).mean()  # 增加一个长周期的均值
    # EMA 对近期数据权重更高，反应更快
    df_copy['ili_ema4'] = df_copy['%UNWEIGHTED ILIS'].ewm(span=4, adjust=False).mean()

    # 填充 NaN
    # df_copy = df_copy.fillna(0)

    return df_copy


# ==================== 异常值处理函数 ====================
def robust_outlier_handling_train(df, target_col='%UNWEIGHTED ILIS'):
    """训练集异常值处理 - 只使用训练集数据"""
    df_copy = df.copy()

    window_size = 4
    df_copy['rolling_median'] = df_copy[target_col].rolling(
        window=window_size, min_periods=1, center=False
    ).median()

    df_copy['abs_deviation'] = np.abs(df_copy[target_col] - df_copy['rolling_median'])
    mad = df_copy['abs_deviation'].median()

    threshold = 3.0 if mad > 0 else 1.0
    df_copy['is_outlier'] = df_copy['abs_deviation'] > threshold * mad

    high_outliers = df_copy['is_outlier'] & (df_copy[target_col] > df_copy['rolling_median'])
    low_outliers = df_copy['is_outlier'] & (df_copy[target_col] <= df_copy['rolling_median'])

    df_copy.loc[high_outliers, target_col] = df_copy.loc[high_outliers, 'rolling_median'] * 1.5
    df_copy.loc[low_outliers, target_col] = df_copy.loc[low_outliers, 'rolling_median'] * 0.5

    df_copy[target_col] = df_copy[target_col].clip(lower=0)

    # 保存异常值处理的参数
    outlier_params = {
        'rolling_median': df_copy['rolling_median'].iloc[-window_size:].values.tolist(),
        'mad': mad,
        'threshold': threshold
    }

    return df_copy.drop(['rolling_median', 'abs_deviation'], axis=1), outlier_params


def robust_outlier_handling_test(df, outlier_params, target_col='%UNWEIGHTED ILIS'):
    """测试集异常值处理 - 使用训练集的参数"""
    df_copy = df.copy()

    # 使用训练集的滚动中位数
    rolling_median_values = outlier_params['rolling_median']
    mad = outlier_params['mad']
    threshold = outlier_params['threshold']

    # 为测试集创建滚动中位数（使用训练集的最后几个值）
    if len(rolling_median_values) >= 4:
        df_copy['rolling_median'] = np.mean(rolling_median_values)
    else:
        df_copy['rolling_median'] = np.mean(rolling_median_values) if rolling_median_values else df_copy[
            target_col].median()

    df_copy['abs_deviation'] = np.abs(df_copy[target_col] - df_copy['rolling_median'])
    df_copy['is_outlier'] = df_copy['abs_deviation'] > threshold * mad

    high_outliers = df_copy['is_outlier'] & (df_copy[target_col] > df_copy['rolling_median'])
    low_outliers = df_copy['is_outlier'] & (df_copy[target_col] <= df_copy['rolling_median'])

    df_copy.loc[high_outliers, target_col] = df_copy.loc[high_outliers, 'rolling_median'] * 1.5
    df_copy.loc[low_outliers, target_col] = df_copy.loc[low_outliers, 'rolling_median'] * 0.5

    df_copy[target_col] = df_copy[target_col].clip(lower=0)

    return df_copy.drop(['rolling_median', 'abs_deviation'], axis=1)


# ==================== HA序列处理函数 ====================
def collect_ha_sequences_enhanced(target_date, ha_df, weeks_back=16, max_seqs=24):
    """增强版HA序列收集"""
    if ha_df.empty or 'sample_date' not in ha_df.columns:
        return [], [], 0

    if not pd.api.types.is_datetime64_any_dtype(ha_df['sample_date']):
        try:
            ha_df['sample_date'] = pd.to_datetime(ha_df['sample_date'])
        except:
            return [], [], 0

    start_date = target_date - pd.DateOffset(weeks=weeks_back)
    sub_df = ha_df[(ha_df['sample_date'] >= start_date) & (ha_df['sample_date'] <= target_date)]

    if sub_df.empty:
        return [], [], 0

    embeds = sub_df['embedding'].tolist()
    time_diffs = [(target_date - d).days / 7.0 for d in sub_df['sample_date']]

    if len(embeds) > max_seqs:
        sorted_pairs = sorted(zip(time_diffs, embeds), key=lambda x: x[0])
        time_diffs, embeds = zip(*sorted_pairs[:max_seqs])
        time_diffs = list(time_diffs)
        embeds = list(embeds)

    return embeds, time_diffs, len(embeds)


def diagnose_ha_processing(train_df, test_df):
    """详细诊断HA序列处理情况"""
    print(f"\n=== HA序列诊断 ===")

    train_has_ha = (train_df['ha_count'] > 0).sum()
    print(f"训练集: 总样本 {len(train_df)}, 有HA序列的样本 {train_has_ha} ({train_has_ha / len(train_df) * 100:.1f}%)")

    test_has_ha = (test_df['ha_count'] > 0).sum()
    print(f"测试集: 总样本 {len(test_df)}, 有HA序列的样本 {test_has_ha} ({test_has_ha / len(test_df) * 100:.1f}%)")


# ==================== 新增：HA序列注意力模块 ====================
class HASequenceAttention(nn.Module):
    """
    对每个时间步内的HA序列集合使用注意力机制进行聚合。
    它使用环境特征作为查询(Query)，来决定关注哪些HA序列。
    """

    def __init__(self, d_model):
        super().__init__()
        # 查询（环境特征）的线性变换
        self.query_proj = nn.Linear(d_model, d_model)
        # 键（HA序列嵌入）的线性变换
        self.key_proj = nn.Linear(d_model, d_model)
        # 用于计算最终注意力分数的向量
        self.attention_vector = nn.Parameter(torch.randn(d_model))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, keys, mask=None):
        """
        Args:
            query (torch.Tensor): 环境特征编码, shape: (batch_size, input_steps, d_model)
            keys (torch.Tensor): HA序列嵌入, shape: (batch_size, input_steps, max_ha_seqs, d_model)
            mask (torch.Tensor): 掩码，用于忽略填充的HA序列, shape: (batch_size, input_steps, max_ha_seqs)

        Returns:
            torch.Tensor: 经过注意力加权聚合后的HA特征, shape: (batch_size, input_steps, d_model)
            torch.Tensor: 注意力权重, shape: (batch_size, input_steps, max_ha_seqs)
        """
        # 1. 线性变换
        # query: [B, T, D] -> [B, T, 1, D]
        # keys:  [B, T, S, D] -> [B, T, S, D]
        # (B=batch_size, T=input_steps, S=max_ha_seqs, D=d_model)
        q_proj = self.query_proj(query).unsqueeze(2)
        k_proj = self.key_proj(keys)

        # 2. 计算注意力分数
        # 使用加性注意力 (Additive Attention)
        # combined: [B, T, S, D]
        combined = torch.tanh(q_proj + k_proj)

        # scores: [B, T, S]
        scores = torch.einsum('btsd,d->bts', combined, self.attention_vector)

        # 3. 应用掩码
        if mask is not None:
            # 将掩码中为0的位置的分数设置为一个很大的负数，这样softmax后权重接近0
            scores.masked_fill_(mask == 0, -1e9)

            # 4. 计算注意力权重
        # weights: [B, T, S]
        weights = self.softmax(scores)

        # 5. 计算加权和 (Context Vector)
        # weights.unsqueeze(-1): [B, T, S, 1]
        # keys:                  [B, T, S, D]
        # context:               [B, T, D]
        context = torch.sum(weights.unsqueeze(-1) * keys, dim=2)

        return context, weights




    # ==================== 位置编码 ====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


# ==================== 简化模型 ====================
# ==================== 修改后的模型 ====================
class StableMultiStepTransformer(nn.Module):
    def __init__(self, n_env, plm_dim, max_ha, input_steps, output_steps,
                 d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.d_model = d_model

        # 特征编码
        self.env_encoder = nn.Linear(n_env, d_model)
        self.ha_encoder = nn.Linear(plm_dim, d_model)

        # ==================== 新增/修改的部分 ====================
        # 引入HA序列注意力模块
        self.ha_attention = HASequenceAttention(d_model)
        # ========================================================

        self.pos_encoding = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_steps)
        )

        # 存储注意力权重以便分析
        self.ha_attention_weights = None

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)

    def forward(self, env, plm, time_diff, seq_lens):
        batch_size, input_steps, max_seqs, plm_dim = plm.shape

        # 1. 环境特征编码
        env_encoded = self.env_encoder(env)  # Shape: [B, T, D]

        # 2. HA序列特征编码
        plm_flat = plm.view(batch_size * input_steps * max_seqs, plm_dim)
        ha_encoded_flat = self.ha_encoder(plm_flat)
        ha_encoded = ha_encoded_flat.view(batch_size, input_steps, max_seqs, self.d_model)  # Shape: [B, T, S, D]

        # ==================== 新增/修改的部分 ====================
        # 3. 创建HA序列的掩码 (Mask)
        # seq_lens shape: [B, T]
        # mask shape: [B, T, S]
        device = env.device
        arange = torch.arange(max_seqs, device=device).expand(batch_size, input_steps, max_seqs)
        mask = arange < seq_lens.unsqueeze(-1)

        # 4. 使用注意力机制聚合HA特征
        # ha_aggregated shape: [B, T, D]
        # self.ha_attention_weights shape: [B, T, S]
        ha_aggregated, self.ha_attention_weights = self.ha_attention(
            query=env_encoded,
            keys=ha_encoded,
            mask=mask
        )
        # ========================================================

        # 5. 特征融合（简单相加）
        combined = env_encoded + ha_aggregated

        # 6. 位置编码
        combined = self.pos_encoding(combined)

        # 7. 主Transformer编码器处理
        encoded = self.encoder(combined)

        # 8. 使用最后一个时间步进行预测
        last_step = encoded[:, -1, :]

        # 9. 输出预测
        output = self.output_layer(last_step)

        return output


def create_stable_model(n_env, plm_dim, max_ha, input_steps, output_steps):
    """更轻量级的模型，防止过拟合"""
    return StableMultiStepTransformer(
        n_env=n_env,
        plm_dim=plm_dim,
        max_ha=max_ha,
        input_steps=input_steps,
        output_steps=output_steps,
        d_model=64,  # 减小维度 128 -> 64
        nhead=4,     # 减少头数 8 -> 4
        num_layers=1, # 减少层数 2 -> 1 (数据太少了，一层足够)
        dropout=0.4   # 增加 Dropout 0.1 -> 0.3
    )



# ==================== 数据加载与处理 ====================
class MultiStepInfluenzaDataset(Dataset):
    def __init__(self, env, plm, time_diff, seq_lens, targets, dates):
        self.env = torch.FloatTensor(env)
        self.plm = torch.FloatTensor(plm)
        self.time_diff = torch.FloatTensor(time_diff)
        self.seq_lens = torch.LongTensor(seq_lens)
        self.targets = torch.FloatTensor(targets)

        date_timestamps = [int(date_sequence[0].timestamp()) for date_sequence in dates if date_sequence]

        self.dates = torch.LongTensor(date_timestamps)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.env[idx], self.plm[idx], self.time_diff[idx],
            self.seq_lens[idx], self.targets[idx], self.dates[idx]
        )



def create_multi_step_seq(data, target_col, env_cols, max_ha, plm_dim, input_steps, output_steps):
    """创建多步预测的序列数据"""
    X_env, X_plm, X_time, seq_lens, y, dates = [], [], [], [], [], []

    n = len(data) - input_steps - output_steps + 1

    for i in range(n):
        # 输入序列
        env = data[env_cols].iloc[i:i + input_steps]
        if env.isna().sum().sum() > len(env_cols):
            continue

        plm_per_step, time_per_step, lens_per_step = [], [], []

        # 处理输入时间步的HA序列
        for j in range(i, i + input_steps):
            embeds = data['ha_embeds'].iloc[j]
            diffs = data['ha_time_diffs'].iloc[j]
            min_len = min(len(embeds), len(diffs))
            embeds = embeds[:min_len] if min_len > 0 else []
            diffs = diffs[:min_len] if min_len > 0 else []

            padded_plm = np.zeros((max_ha, plm_dim), dtype=np.float32)
            padded_time = np.zeros((max_ha, 1), dtype=np.float32)

            if embeds:
                min_len = min(max_ha, len(embeds))
                for k in range(min_len):
                    if len(embeds[k]) == plm_dim:
                        padded_plm[k] = embeds[k]
                        padded_time[k, 0] = diffs[k] if k < len(diffs) else 0
                lens_per_step.append(min_len)
            else:
                lens_per_step.append(0)

            plm_per_step.append(padded_plm)
            time_per_step.append(padded_time)

        # 多步目标值
        target_start = i + input_steps
        target_end = target_start + output_steps

        if target_end > len(data):
            continue

        targets = data[target_col].iloc[target_start:target_end].values

        if len(targets) != output_steps or np.any(pd.isna(targets)) or np.any(targets < 0):
            continue

        X_env.append(env.values)
        X_plm.append(np.array(plm_per_step, dtype=np.float32))
        X_time.append(np.array(time_per_step, dtype=np.float32))
        seq_lens.append(lens_per_step)
        y.append(targets)
        # 将Series转换为list
        dates.append(data['time'].iloc[i + input_steps:i + input_steps + output_steps].tolist())

    if len(X_env) == 0:
        print(f"警告: 无法创建有效的多步序列数据")
        return (np.array(X_env, dtype=np.float32), np.array(X_plm, dtype=np.float32),
                np.array(X_time, dtype=np.float32), np.array(seq_lens),
                np.array(y, dtype=np.float32), dates)  # 注意这里返回的是 dates

    X_env_array = np.array(X_env, dtype=np.float32)
    X_plm_array = np.array(X_plm, dtype=np.float32)
    X_time_array = np.array(X_time, dtype=np.float32)
    seq_lens_array = np.array(seq_lens)
    y_array = np.array(y, dtype=np.float32)

    return X_env_array, X_plm_array, X_time_array, seq_lens_array, y_array, dates


def create_multi_step_seq_for_prediction(data, env_cols, max_ha, plm_dim, input_steps, output_steps):
    """为多步预测创建序列数据（不使用目标变量真实值）"""
    X_env, X_plm, X_time, seq_lens = [], [], [], []

    if len(data) < input_steps:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # 使用最后input_steps个时间步进行预测
    i = len(data) - input_steps
    env = data[env_cols].iloc[i:i + input_steps]

    if env.isna().sum().sum() > 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    plm_per_step, time_per_step, lens_per_step = [], [], []

    for j in range(i, i + input_steps):
        embeds = data['ha_embeds'].iloc[j] if 'ha_embeds' in data.columns else []
        diffs = data['ha_time_diffs'].iloc[j] if 'ha_time_diffs' in data.columns else []

        min_len = min(len(embeds), len(diffs))
        embeds = embeds[:min_len] if min_len > 0 else []
        diffs = diffs[:min_len] if min_len > 0 else []

        padded_plm = np.zeros((max_ha, plm_dim), dtype=np.float32)
        padded_time = np.zeros((max_ha, 1), dtype=np.float32)

        if embeds:
            min_len = min(max_ha, len(embeds))
            for k in range(min_len):
                if len(embeds[k]) == plm_dim:
                    padded_plm[k] = embeds[k]
                    padded_time[k, 0] = diffs[k] if k < len(diffs) else 0
            lens_per_step.append(min_len)
        else:
            lens_per_step.append(0)

        plm_per_step.append(padded_plm)
        time_per_step.append(padded_time)

    X_env.append(env.values)
    X_plm.append(np.array(plm_per_step, dtype=np.float32))
    X_time.append(np.array(time_per_step, dtype=np.float32))
    seq_lens.append(lens_per_step)

    return (np.array(X_env, dtype=np.float32),
            np.array(X_plm, dtype=np.float32),
            np.array(X_time, dtype=np.float32),
            np.array(seq_lens))


# ==================== 训练策略 ====================
def conservative_training_strategy(model, train_loader, val_loader=None, device=None, epochs=100, patience=15):
    """
    优化后的训练策略：包含验证集评估、早停机制（Early Stopping）和最佳模型保存。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # 损失函数
    criterion = nn.SmoothL1Loss(beta=1.0)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {'epochs': [], 'train_loss': [], 'val_loss': []}

    # 早停与最佳模型记录
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = epochs  # 默认值为最大epochs
    patience_counter = 0

    print(f"开始训练 - 最大Epochs: {epochs}, 早停Patience: {patience}")

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0
        num_train_batches = 0

        for batch in train_loader:
            env, plm, time_diff, seq_len, y, _ = batch
            env, plm, time_diff, seq_len, y = [x.to(device) for x in [env, plm, time_diff, seq_len, y]]

            if torch.isnan(env).any() or torch.isinf(env).any(): continue

            optimizer.zero_grad()
            pred = model(env, plm, time_diff, seq_len)
            loss = criterion(pred, y)

            if torch.isnan(loss): continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0

        # --- 验证阶段 ---
        avg_val_loss = float('inf')
        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            num_val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    env, plm, time_diff, seq_len, y, _ = batch
                    env, plm, time_diff, seq_len, y = [x.to(device) for x in [env, plm, time_diff, seq_len, y]]

                    pred = model(env, plm, time_diff, seq_len)
                    loss = criterion(pred, y)

                    total_val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0

            # 更新学习率
        scheduler.step()

        # --- 记录历史 ---
        history['epochs'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # --- 打印进度 ---
        if epoch % 5 == 0 or epoch == epochs - 1:
            val_str = f" | Val Loss: {avg_val_loss:.6f}" if val_loader else ""
            print(
                f'Epoch {epoch}: Train Loss: {avg_train_loss:.6f}{val_str} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

            # ---早停与保存最佳模型 (Checkpointing) ---
        if val_loader is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1  # 记录最佳 epoch (从1开始计数)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发！在第 {epoch} 轮停止。最佳 Epoch: {best_epoch}, Val Loss: {best_val_loss:.6f}")
                    break

                    # 训练结束后，加载验证集表现最好的参数
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("已恢复至验证集Loss最低的模型参数。")

    return model, history, best_epoch


# ==================== 预测函数 ====================
def evaluate_ensemble(models, full_df, train_end_index, top_feats, max_ha, plm_dim,
                      input_steps, output_steps, device, inverse_transform_func):
    """
    集成评估：使用多个模型进行投票预测 (Bagging)
    """
    print(f"进行集成滚动评估 (模型数量: {len(models)})...")
    for m in models: m.eval()

    all_predictions = []
    all_dates = []
    all_true_values = []

    total_len = len(full_df)
    current_idx = train_end_index

    with torch.no_grad():
        while current_idx < total_len:
            if current_idx - input_steps < 0:
                current_idx += output_steps;
                continue

            input_window_df = full_df.iloc[current_idx - input_steps: current_idx].copy()
            if len(input_window_df) < input_steps: break

            X_env, X_plm, X_time, seq_len = create_multi_step_seq_for_prediction(
                input_window_df, top_feats, max_ha, plm_dim, input_steps, output_steps
            )
            if len(X_env) == 0: break

            env_tensor = torch.FloatTensor(X_env).to(device)
            plm_tensor = torch.FloatTensor(X_plm).to(device)
            time_tensor = torch.FloatTensor(X_time).to(device)
            seq_len_tensor = torch.LongTensor(seq_len).to(device)

            # === 关键修改：所有模型同时预测，取平均 ===
            ensemble_preds = []
            for model in models:
                pred = model(env_tensor, plm_tensor, time_tensor, seq_len_tensor)
                pred = pred.cpu().numpy()[0]
                ensemble_preds.append(pred)

                # 取平均 (Model Averaging)
            avg_pred_trans = np.mean(ensemble_preds, axis=0)

            # 反变换
            predictions = inverse_transform_func(avg_pred_trans)
            predictions = np.maximum(predictions, 0)

            # 获取真实值
            end_idx = min(current_idx + output_steps, total_len)
            steps_to_take = end_idx - current_idx
            target_dates = full_df['time'].iloc[current_idx: end_idx].tolist()
            target_values = full_df['%UNWEIGHTED ILIS'].iloc[current_idx: end_idx].values.tolist()

            predictions = predictions[:steps_to_take]

            if len(predictions) > 0:
                all_predictions.append(predictions)
                all_dates.append(target_dates)
                all_true_values.append(target_values)

            current_idx += output_steps

    return all_predictions, all_dates, all_true_values


def calculate_multi_step_metrics(all_true, all_pred):
    """计算多步预测的评估指标"""
    metrics_by_step = {}

    # 找到最大步数
    max_steps = max(len(pred) for pred in all_pred) if all_pred else 0

    for step in range(max_steps):
        step_true, step_pred = [], []

        for true_vals, pred_vals in zip(all_true, all_pred):
            if step < len(pred_vals) and step < len(true_vals):
                step_true.append(true_vals[step])
                step_pred.append(pred_vals[step])

        if step_true and step_pred:
            step_true = np.array(step_true)
            step_pred = np.array(step_pred)

            metrics_by_step[step] = {
                'mse': mean_squared_error(step_true, step_pred),
                'mae': mean_absolute_error(step_true, step_pred),
                'rmse': np.sqrt(mean_squared_error(step_true, step_pred)),
                'r2': r2_score(step_true, step_pred),
                'mape': np.mean(np.abs((step_true - step_pred) / np.maximum(step_true, 1e-6))) * 100
            }

    # 总体指标（所有步的平均）
    all_true_flat = []
    all_pred_flat = []
    for true_vals, pred_vals in zip(all_true, all_pred):
        min_len = min(len(true_vals), len(pred_vals))
        all_true_flat.extend(true_vals[:min_len])
        all_pred_flat.extend(pred_vals[:min_len])

    if all_true_flat and all_pred_flat:
        overall_metrics = {
            'mse': mean_squared_error(all_true_flat, all_pred_flat),
            'mae': mean_absolute_error(all_true_flat, all_pred_flat),
            'rmse': np.sqrt(mean_squared_error(all_true_flat, all_pred_flat)),
            'r2': r2_score(all_true_flat, all_pred_flat),
            'mape': np.mean(np.abs(
                (np.array(all_true_flat) - np.array(all_pred_flat)) / np.maximum(np.array(all_true_flat), 1e-6))) * 100
        }
    else:
        overall_metrics = {'mse': np.nan, 'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'mape': np.nan}

    return metrics_by_step, overall_metrics



def plot_multi_step_test_results(test_df, predictions, dates):
    """绘制多步预测测试集结果"""
    plt.figure(figsize=(15, 10))

    # 提取所有真实值和预测值
    all_true_dates = []
    all_true_values = []
    all_pred_dates = []
    all_pred_values = []

    for pred_sequence, date_sequence in zip(predictions, dates):
        for i, (pred, date) in enumerate(zip(pred_sequence, date_sequence)):
            all_pred_dates.append(date)
            all_pred_values.append(pred)

            # 找到对应的真实值
            true_val = test_df[test_df['time'] == date]['%UNWEIGHTED ILIS']
            if not true_val.empty:
                all_true_dates.append(date)
                all_true_values.append(true_val.values[0])

    # 绘制真实值
    plt.plot(test_df['time'], test_df['%UNWEIGHTED ILIS'], label='真实值', linewidth=2, color='blue', alpha=0.7)

    # 绘制预测值
    plt.scatter(all_pred_dates, all_pred_values, label='多步预测值', color='red', s=30, alpha=0.7)

    # 连接预测点以显示趋势
    sorted_indices = np.argsort(all_pred_dates)
    sorted_dates = [all_pred_dates[i] for i in sorted_indices]
    sorted_preds = [all_pred_values[i] for i in sorted_indices]
    plt.plot(sorted_dates, sorted_preds, color='red', linestyle='--', alpha=0.5)

    plt.xlabel('日期')
    plt.ylabel('ILI值')
    plt.title('独立测试集多步预测结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/multi_step_final_test_set_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_multi_step_results(train_df, test_df, final_metrics, step_metrics):
    """保存多步预测结果"""
    results = {
        'data_info': {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'train_period': f"{train_df['time'].min()} 到 {train_df['time'].max()}",
            'test_period': f"{test_df['time'].min()} 到 {test_df['time'].max()}",
            'prediction_horizon': '多步预测'
        },
        'final_test_metrics': final_metrics,
        'step_by_step_metrics': step_metrics
    }

    with open('results/multi_step_final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("多步预测最终结果已保存")


# ==================== 主程序 (重构后) ====================
def main():
    print("流感多步预测模型启动...")
    print("应用严格无数据泄露的数据处理流程...")

    # --- 参数设置 ---
    input_steps = 12
    output_steps = 4
    num_ensemble_models = 3
    test_size = 52  # 最后52周作为测试集

    MODEL_CONFIG = {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 1,
        'dropout': 0.4
    }
    print(f"多步预测配置: 输入{input_steps}周，预测未来{output_steps}周 | 集成模型数: {num_ensemble_models}")

    # --- 数据加载与初步聚合 ---
    print("读取并聚合环境与流感数据...")
    try:
        df_weekly = pd.read_excel("../data/yinzi19.xlsx")
        df_ili = pd.read_excel("../data/NY19.xlsx")
    except FileNotFoundError as e:
        print(f"错误：数据文件未找到 - {e}")
        return

    df_weekly['time'] = pd.to_datetime(df_weekly['time'])
    # ... (与您原代码相同的聚合逻辑) ...
    agg = {
        'temperature_2m_mean (°C)': 'mean', 'sunshine_duration (s)': 'sum', 'precipitation_sum (mm)': 'sum',
        'relative_humidity_2m_mean (%)': 'mean', 'surface_pressure_mean (hPa)': 'mean',
        'wind_speed_10m_mean (km/h)': ['mean', 'max'],
        'pressure_msl_mean (hPa)': 'mean', 'wet_bulb_temperature_2m_mean (°C)': 'mean',
        'apparent_temperature_mean (°C)': 'mean',
        'temperature_2m_max (°C)': 'max', 'temperature_2m_min (°C)': 'min', 'leaf_wetness_probability_mean (%)': 'mean',
        'cloud_cover_mean (%)': 'mean', 'pm10 (μg/m³)': 'mean', 'pm2_5 (μg/m³)': 'mean',
        'nitrogen_dioxide (μg/m³)': 'mean',
        'ozone (μg/m³)': 'mean', 'us_aqi (USAQI)': 'mean'
    }
    agg = {k: v for k, v in agg.items() if k in df_weekly.columns}
    df_weekly = df_weekly.set_index('time').resample('W-MON').agg(agg)
    new_cols = [f"{c[0]}_{c[1]}" if isinstance(c, tuple) and c[1] != '' else c[0] for c in df_weekly.columns]
    df_weekly.columns = ['temp_mean', 'sunshine_total', 'precip_total', 'humidity_mean', 'surface_pressure_mean',
                         'wind_speed_mean',
                         'wind_speed_max', 'pressure_msl_mean', 'wet_bulb_temp_mean', 'apparent_temp_mean', 'temp_max',
                         'temp_min',
                         'leaf_wetness_mean', 'cloud_cover_mean', 'pm10_mean', 'pm2_5_mean', 'no2_mean', 'ozone_mean',
                         'aqi_mean']

    df_ili = df_ili[['YEAR', 'WEEK', '%UNWEIGHTED ILIS', 'REGION']]
    df_ili['time'] = pd.to_datetime(df_ili['YEAR'].astype(str) + '-W' + df_ili['WEEK'].astype(str) + '-1',
                                    format='%Y-W%W-%w')
    df = pd.merge(df_ili, df_weekly, on='time', how='inner')
    df.drop(['YEAR', 'WEEK'], axis=1, inplace=True)
    df = pd.get_dummies(df, columns=['REGION'])
    df = df.sort_values('time').reset_index(drop=True)

    # --- 严格无泄露的数据处理流程 ---

    # 1. 在所有预处理之前划分数据集
    print(f"\n步骤1: 在所有预处理之前划分原始数据集 (测试集大小: {test_size}周)...")
    if len(df) <= test_size:
        print("错误: 数据集太小，无法划分训练集和测试集。")
        return
    train_df_raw = df.iloc[:-test_size].copy()
    test_df_raw = df.iloc[-test_size:].copy()

    print(
        f"原始训练集: {len(train_df_raw)}行, 时间范围: {train_df_raw['time'].min().date()} - {train_df_raw['time'].max().date()}")
    print(
        f"原始测试集: {len(test_df_raw)}行, 时间范围: {test_df_raw['time'].min().date()} - {test_df_raw['time'].max().date()}")

    # 2. 计算只依赖训练集的参数 (如cap_limit)
    cap_limit = train_df_raw['%UNWEIGHTED ILIS'].quantile(0.90)  # 使用95分位数可能更稳健
    print(f"峰值截断阈值 (仅基于训练集计算): {cap_limit:.4f}")

    # 3. 创建一个用于特征工程的完整DataFrame，以确保测试集lag计算的连续性
    train_df_raw['source'] = 'train'
    test_df_raw['source'] = 'test'
    full_df_for_processing = pd.concat([train_df_raw, test_df_raw], ignore_index=True)

    # 4. 对拼接后的完整数据进行统一的、无未来泄露的特征工程
    print("\n步骤2: 对完整数据进行统一的特征工程 (创建时间/衍生/滞后特征)...")
    # a. 创建不依赖于目标变量的特征
    processed_df = create_safe_features(full_df_for_processing)
    # b. 对目标变量进行预处理（如截断）
    processed_df['%UNWEIGHTED ILIS'] = processed_df['%UNWEIGHTED ILIS'].clip(upper=cap_limit)
    # c. 创建依赖于历史目标变量的特征（现在是安全的）
    processed_df = create_time_series_features(processed_df)

    # 5. 丢弃任何因无法计算完整特征而包含NaN的行 (主要是数据开头的行)
    rows_before_drop = len(processed_df)
    processed_df.dropna(inplace=True)
    rows_after_drop = len(processed_df)
    print(f"特征工程后，丢弃包含NaN的行: 共丢弃 {rows_before_drop - rows_after_drop} 行")

    # 6. 将处理好的数据重新分离为训练集和测试集
    print("\n步骤3: 将处理干净的数据重新分离为训练集和测试集...")
    train_df = processed_df[processed_df['source'] == 'train'].copy().drop(columns=['source']).reset_index(drop=True)
    test_df = processed_df[processed_df['source'] == 'test'].copy().drop(columns=['source']).reset_index(drop=True)

    # 记录用于滚动评估的起始索引
    train_end_index_in_full = len(train_df)

    print(
        f"最终干净的训练集: {len(train_df)}行, 时间范围: {train_df['time'].min().date()} - {train_df['time'].max().date()}")
    print(
        f"最终干净的测试集: {len(test_df)}行, 时间范围: {test_df['time'].min().date()} - {test_df['time'].max().date()}")

    if len(train_df) < input_steps + output_steps or len(test_df) < output_steps:
        print("错误：处理后的训练集或测试集过小，无法继续。")
        return

        # 7. 对目标变量进行变换 (在分离后的数据集上操作)
    train_df['target'] = np.log1p(train_df['%UNWEIGHTED ILIS'])
    test_df['target'] = np.log1p(test_df['%UNWEIGHTED ILIS'])

    def inverse_transform(x):
        return np.expm1(x)

    # --- HA序列处理 ---
    print("\n步骤4: 处理HA序列...")
    ha_processed = pd.DataFrame()
    try:
        # ... (与您原代码相同的HA序列文件解析逻辑) ...
        ha_records = list(SeqIO.parse("../data/sequences01.fasta", "fasta"))
        ha_data = []
        for record in ha_records:
            desc_parts = [part.strip() for part in record.description.split("|")]
            sample_date = pd.NaT
            for part in desc_parts:
                sample_date = parse_fasta_date(part)
                if not pd.isna(sample_date): break
            if pd.isna(sample_date): sample_date = parse_fasta_date(record.description)
            if pd.isna(sample_date): continue
            if is_valid_sequence(str(record.seq), "amino_acid"):
                ha_data.append({"sequence": str(record.seq), "sample_date": sample_date})

        ha_df = pd.DataFrame(ha_data)
        if not ha_df.empty:
            ha_processed = process_ha_sequences_with_plm(ha_df)
            if not ha_processed.empty:
                ha_processed = augment_ha_sequences(ha_processed, augmentation_factor=2)
    except FileNotFoundError:
        print("警告: HA序列文件未找到。")
    check_ha_data_quality(ha_processed)

    plm_dim = ha_processed['embedding'].iloc[0].shape[
        0] if not ha_processed.empty and 'embedding' in ha_processed.columns and len(ha_processed) > 0 else 320

    # --- 特征选择与标准化 (严格在训练集上fit) ---
    print("\n步骤5: 特征选择与标准化 (严格在训练集上fit)...")
    ignore_cols = ['time', '%UNWEIGHTED ILIS', 'target', 'is_outlier', 'ha_embeds', 'ha_time_diffs', 'ha_count']
    feat_cols = [c for c in train_df.columns if c not in ignore_cols]

    # KNN Imputer (只在训练集上fit)
    imputer = KNNImputer(n_neighbors=5)
    train_df[feat_cols] = imputer.fit_transform(train_df[feat_cols])
    test_df[feat_cols] = imputer.transform(test_df[feat_cols])  # 用训练集的imputer变换测试集

    # 特征选择 (只在训练集上fit)
    selector = SelectKBest(score_func=mutual_info_regression, k=min(20, len(feat_cols)))
    selector.fit(train_df[feat_cols], train_df['target'])
    top_feats = [feat_cols[i] for i in selector.get_support(indices=True)]
    print(f"选中的Top-{len(top_feats)}特征: {top_feats}")

    # 标准化 (只在训练集上fit)
    scaler = StandardScaler()
    train_df[top_feats] = scaler.fit_transform(train_df[top_feats])
    test_df[top_feats] = scaler.transform(test_df[top_feats])  # 用训练集的scaler变换测试集
    print("特征已在训练集上fit，并transform到训练集和测试集。")

    # --- 准备最终的、包含所有信息的完整数据集 ---
    full_processed_df = pd.concat([train_df, test_df], ignore_index=True)

    # --- 为完整数据集安全地匹配HA序列 ---
    print("\n步骤6: 为完整数据集安全地匹配HA序列...")
    full_processed_df['ha_embeds'] = [[] for _ in range(len(full_processed_df))]
    full_processed_df['ha_time_diffs'] = [[] for _ in range(len(full_processed_df))]
    full_processed_df['ha_count'] = 0  # ← 添加这行

    if not ha_processed.empty:
        # 这一步非常重要：确保ha_processed按时间排序
        ha_processed = ha_processed.sort_values('sample_date').reset_index(drop=True)

        for idx, row in tqdm(full_processed_df.iterrows(), total=len(full_processed_df), desc="匹配HA序列"):
            # collect_ha_sequences_enhanced 内部已修复，只看过去
            embeds, diffs, cnt = collect_ha_sequences_enhanced(row['time'], ha_processed, max_seqs=24)
            full_processed_df.at[idx, 'ha_embeds'] = embeds
            full_processed_df.at[idx, 'ha_time_diffs'] = diffs
            full_processed_df.at[idx, 'ha_count'] = cnt  # ← 添加这行

        diagnose_ha_processing(
            full_processed_df.iloc[:train_end_index_in_full],
            full_processed_df.iloc[train_end_index_in_full:]
        )

    max_ha = full_processed_df['ha_embeds'].apply(len).max()
    if max_ha == 0: max_ha = 1  # 至少为1，防止维度错误
    print(f"最大HA序列长度为: {max_ha}")

    # --- 创建训练序列 (只使用训练集部分的数据) ---
    print("\n步骤7: 创建多步序列样本 (用于训练和验证)...")
    training_data_source = full_processed_df.iloc[:train_end_index_in_full]

    # 这里生成的包含了所有的训练数据
    X_env_all, X_plm_all, X_time_all, seq_len_all, y_seq_all, dates_seq_all = create_multi_step_seq(
        training_data_source, 'target', top_feats, max_ha, plm_dim, input_steps, output_steps
    )

    if len(X_env_all) == 0:
        print("错误：未能创建任何训练样本，程序终止。")
        return

        # --- 新增：划分训练集和验证集 ---
    val_size = 52  # 验证集大小：取训练数据的最后 1 年 (52周)

    # 确保数据量足够划分
    if len(X_env_all) > val_size + 50:
        train_size = len(X_env_all) - val_size
        print(f"将训练数据划分为: 训练集 {train_size} 样本, 验证集 {val_size} 样本")

        # 切分数组
        X_env_train = X_env_all[:train_size]
        X_plm_train = X_plm_all[:train_size]
        X_time_train = X_time_all[:train_size]
        seq_len_train = seq_len_all[:train_size]
        y_seq_train = y_seq_all[:train_size]
        dates_seq_train = dates_seq_all[:train_size]

        X_env_val = X_env_all[train_size:]
        X_plm_val = X_plm_all[train_size:]
        X_time_val = X_time_all[train_size:]
        seq_len_val = seq_len_all[train_size:]
        y_seq_val = y_seq_all[train_size:]
        dates_seq_val = dates_seq_all[train_size:]
    else:
        print("警告：样本数量不足以划分验证集，将使用全部数据进行训练（无早停）。")
        X_env_train, X_plm_train, X_time_train, seq_len_train, y_seq_train, dates_seq_train = \
            X_env_all, X_plm_all, X_time_all, seq_len_all, y_seq_all, dates_seq_all
        X_env_val = None

        # --- 模型训练 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n步骤8: 在设备 {device} 上训练模型 (启用早停和验证)...")
    trained_models = []
    training_histories = []

    # 准备阶段一的 DataLoader
    val_loader = None
    if X_env_val is not None:
        val_ds = MultiStepInfluenzaDataset(X_env_val, X_plm_val, X_time_val, seq_len_val, y_seq_val, dates_seq_val)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

        # 准备全量数据 (Train + Val) 的 DataLoader，用于阶段二
    all_ds = MultiStepInfluenzaDataset(X_env_all, X_plm_all, X_time_all, seq_len_all, y_seq_all, dates_seq_all)
    all_loader = DataLoader(all_ds, batch_size=8, shuffle=True)

    for i in range(num_ensemble_models):
        print(f"\n--- 训练模型 {i + 1}/{num_ensemble_models} (Seed: {42 + i}) ---")
        set_seed(42 + i)

        # >>> 阶段一：在训练集上训练，验证集上早停，寻找最佳 Epoch <<<
        print("  [阶段一] 寻找最佳 Epoch...")
        train_ds = MultiStepInfluenzaDataset(X_env_train, X_plm_train, X_time_train, seq_len_train, y_seq_train,
                                             dates_seq_train)
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

        # 初始化一个临时模型用于搜索
        search_model = StableMultiStepTransformer(
            n_env=X_env_train.shape[2], plm_dim=plm_dim, max_ha=max_ha,
            input_steps=input_steps, output_steps=output_steps,
            **MODEL_CONFIG  # ← 使用统一配置
        )

        # 注意：接收 best_epoch
        _, _, best_epoch = conservative_training_strategy(
            search_model,
            train_loader,
            val_loader=val_loader,
            device=device,
            epochs=100,
            patience=15
        )

        print(f"  => 模型 {i + 1} 确定的最佳训练轮数为: {best_epoch}")

        # >>> 阶段二：使用最佳 Epoch 在全量数据 (Train + Val) 上重训练 <<<
        print(f"  [阶段二] 使用全量数据重训练 {best_epoch} 轮...")

        final_model = StableMultiStepTransformer(
            n_env=X_env_train.shape[2], plm_dim=plm_dim, max_ha=max_ha,
            input_steps=input_steps, output_steps=output_steps,
            **MODEL_CONFIG  # ← 使用相同配置
        )

        # 这里不传 val_loader，因为我们要用全量数据训练固定轮数
        # 我们可以稍微多训练一点点 (比如 +20%) 因为数据量变大了，或者保持原样。保持原样通常最安全。
        final_model, history_final, _ = conservative_training_strategy(
            final_model,
            all_loader,  # 使用全量数据加载器
            val_loader=None,  # 不再验证
            device=device,
            epochs=best_epoch,  # 强制训练到这个轮数
            patience=9999  # 禁用早停
        )

        trained_models.append(final_model)
        training_histories.append(history_final)  # ← 收集历史
        torch.save(final_model.state_dict(), f'checkpoints/ensemble_model_{i}.pth')

        # --- 模型评估与可视化 ---
        # --- 模型评估与可视化 ---
    print("\n步骤9: 模型训练完成，开始滚动评估...")
    final_preds, final_dates, final_true = evaluate_ensemble(
        trained_models, full_processed_df, train_end_index_in_full, top_feats,
        max_ha, plm_dim, input_steps, output_steps, device, inverse_transform
    )

    if final_preds:
        step_metrics, overall_metrics = calculate_multi_step_metrics(final_true, final_preds)

        print("\n===== 最终测试结果 (集成模型) =====")
        for key, value in overall_metrics.items():
            print(f"{key.upper()}: {value:.4f}")

        print("\n--- 分步评估指标 ---")
        for step, metrics in step_metrics.items():
            print(f"  预测未来第 {step + 1} 周 - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")

        save_multi_step_results(train_df, test_df, overall_metrics, step_metrics)  # ← 添加

            # ====================  开始补充可视化代码 ====================
        print("\n生成最终结果图表...")
        try:
            # 假设您的绘图函数都封装在一个名为 plot.py 的文件中
            # 我们需要导入它
            import plot as plot_utils

            # 1. 绘制最终的测试集预测结果图 (这个函数您已经提供了)
            # plot_multi_step_test_results(test_df, final_preds, final_dates)

            # 您原始代码中其他的绘图函数调用
            # 我们需要准备绘图用的数据

            # a. 准备用于展示数据预处理过程的数据
            # 使用最开始加载的原始 df DataFrame
            plot_df_for_fig1 = df.copy()

            # b. 准备用于绘制相关性散点图和误差分析图的数据
            flat_pred = [item for sublist in final_preds for item in sublist]
            flat_true = [item for sublist in final_true for item in sublist]
            min_len = min(len(flat_pred), len(flat_true))
            flat_pred = np.array(flat_pred[:min_len])
            flat_true = np.array(flat_true[:min_len])

            # 调用绘图函数
            # 注意：请确保您的 plot.py 文件存在，并且包含以下函数
            # 如果函数名不同，请相应修改

            # 绘制数据预处理效果图
            if hasattr(plot_utils, 'plot_ili_preprocessing'):
                plot_utils.plot_ili_preprocessing(plot_df_for_fig1, cap_limit, train_df, test_df)

                # 绘制特征重要性图
            if hasattr(plot_utils, 'plot_feature_importance'):
                # 注意：feat_cols 在您的代码中包含了所有特征，我们需要确保传递的是用于选择的特征列表
                features_for_selection = [c for c in train_df.columns if
                                          c not in ['time', '%UNWEIGHTED ILIS', 'target', 'is_outlier', 'ha_embeds',
                                                    'ha_time_diffs', 'ha_count']]
                plot_utils.plot_feature_importance(selector, features_for_selection)

                # 绘制训练历史曲线 (假设您保存了 history)
            # `conservative_training_strategy` 返回了 history，但您在主循环中没有收集它
            # 我们可以在训练循环中收集它
            # *** 请在 main 函数的模型训练循环中添加 `training_histories.append(history)` ***
            if training_histories and hasattr(plot_utils, 'plot_training_history'):
                plot_utils.plot_training_history(training_histories, num_ensemble_models)

            # 绘制测试集多步预测结果 (这个函数您已经有了，这里作为示例)
            plot_multi_step_test_results(test_df, final_preds, final_dates)

            # 绘制预测值与真实值的相关性散点图
            if hasattr(plot_utils, 'plot_correlation_scatterplot'):
                plot_utils.plot_correlation_scatterplot(flat_true, flat_pred, overall_metrics.get('r2', 0))

                # 绘制MAPE误差分析图
            if hasattr(plot_utils, 'plot_mape_error_analysis'):
                plot_utils.plot_mape_error_analysis(flat_true, flat_pred, overall_metrics.get('mape', 0))

                # 绘制多步预测能力衰减图
            if step_metrics and hasattr(plot_utils, 'plot_multistep_decay'):
                plot_utils.plot_multistep_decay(step_metrics)

        except ImportError:
            print("警告: 未找到 'plot.py' 或 'plot_utils' 文件，跳过附加的可视化。")

            # ====================  可视化代码补充结束 ====================

    else:
        print("评估失败，未能生成任何预测结果。")

    print("\n程序结束。")


if __name__ == "__main__":
    main()