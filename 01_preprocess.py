#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_preprocess.py
功能：加载原始数据，清洗缺失值，统一列名，并进行标准化处理。
输出：result/penguins_processed.csv (包含原始数据 + 标准化后的特征列)
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 配置 ---
# 原始数据路径 (请根据实际情况调整)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'penguins_size.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'result')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 需要使用的特征
WANTED_FEATURES = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
ALIAS_MAP = {
    'culmen_length_mm': 'bill_length_mm',
    'culmen_depth_mm': 'bill_depth_mm'
}

def main():
    print(f"[预处理] 正在加载数据: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"找不到文件: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # 1. 统一列名
    df = df.rename(columns=ALIAS_MAP)

    # 2. 检查必要特征是否存在
    available_feats = [f for f in WANTED_FEATURES if f in df.columns]
    if len(available_feats) < len(WANTED_FEATURES):
        raise ValueError(f"缺少特征列，当前只有: {available_feats}")
    
    # 3. 缺失值处理：删除特征或标签(species)缺失的行
    # 确保 species 存在，方便后续验证
    if 'species' not in df.columns:
        df['species'] = 'Unknown' 
    
    initial_len = len(df)
    df_clean = df.dropna(subset=available_feats + ['species']).reset_index(drop=True)
    print(f"[预处理] 清洗缺失值: {initial_len} -> {len(df_clean)} 行")

    # 4. 异常值处理：使用 IQR 方法裁剪极端值（减少 DBSCAN 噪声）
    print("[预处理] 正在处理异常值...")
    for col in available_feats:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        # 裁剪而非删除，保留更多数据
        df_clean[col] = df_clean[col].clip(lower, upper)
    
    # 5. 标准化 (StandardScaler)
    # 我们将标准化后的数据存为新列，后缀加 _std，保留原数据方便查看
    scaler = StandardScaler()
    X = df_clean[available_feats].values
    X_scaled = scaler.fit_transform(X)
    
    scaled_cols = [f"{col}_std" for col in available_feats]
    df_scaled = pd.DataFrame(X_scaled, columns=scaled_cols)
    
    # 合并原始数据和标准化数据
    df_final = pd.concat([df_clean, df_scaled], axis=1)

    # 6. 保存结果
    out_path = os.path.join(OUTPUT_DIR, 'penguins_processed.csv')
    df_final.to_csv(out_path, index=False)
    print(f"[预处理] 完成。已保存至: {out_path}")
    print(f"[预处理] 标准化特征列名: {scaled_cols}")

if __name__ == "__main__":
    main()