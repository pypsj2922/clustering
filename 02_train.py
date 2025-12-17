#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train.py
功能：读取处理后的数据，训练 KMeans/DBSCAN/Agglomerative，计算指标。
输出：result/penguins_labeled.csv (新增了聚类标签列)
      result/clustering_metrics.csv (各模型得分汇总)
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# --- 配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, 'result', 'penguins_processed.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'result')
RANDOM_STATE = 42

# 这里的特征列名要对应 01_preprocess.py 生成的 _std 列
FEATURE_COLS = [
    'bill_length_mm_std', 'bill_depth_mm_std', 
    'flipper_length_mm_std', 'body_mass_g_std'
]

def compute_metrics(X, labels, true_labels=None):
    """计算轮廓系数(Silhouette)和外部指标(ARI, NMI)"""
    res = {'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)}
    if res['n_clusters'] > 1:
        # Silhouette (忽略噪声点 -1 对于 DBSCAN)
        mask = labels != -1
        if np.sum(mask) > 1:
             res['silhouette'] = silhouette_score(X[mask], labels[mask])
        else:
             res['silhouette'] = -1
    else:
        res['silhouette'] = -1
    
    # 外部指标（如果有真实标签）
    if true_labels is not None:
        res['ARI'] = adjusted_rand_score(true_labels, labels)
        res['NMI'] = normalized_mutual_info_score(true_labels, labels)
    return res

def main():
    print(f"[训练] 正在加载数据: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError("请先运行 01_preprocess.py 生成数据文件！")
        
    df = pd.read_csv(INPUT_FILE)
    X = df[FEATURE_COLS].values
    y_true = df['species'].values # 用于评估

    results_list = []
    
    # ---------------- KMeans ----------------
    print("[训练] 正在运行 KMeans (搜索最佳 K)...")
    best_k = 2
    best_sil = -1
    best_km_labels = None
    
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        lbls = km.fit_predict(X)
        score = silhouette_score(X, lbls)
        if score > best_sil:
            best_sil = score
            best_k = k
            best_km_labels = lbls
            
    # 记录最佳 KMeans
    m_km = compute_metrics(X, best_km_labels, y_true)
    m_km['method'] = f'KMeans(K={best_k})'
    results_list.append(m_km)
    df['kmeans_label'] = best_km_labels # 保存标签到 dataframe

    # ---------------- DBSCAN ----------------
    print("[训练] 正在运行 DBSCAN (网格搜索)...")
    best_db_score = -2
    best_db_labels = np.full(len(df), -1) # 默认全是噪声
    best_params = "N/A"
    
    eps_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    min_samples_list = [3, 5, 10]
    
    for eps in eps_list:
        for ms in min_samples_list:
            db = DBSCAN(eps=eps, min_samples=ms)
            lbls = db.fit_predict(X)
            unique = set(lbls)
            if len(unique) > 1: # 必须至少有一个簇
                # 计算轮廓系数
                mask = lbls != -1
                if np.sum(mask) > 2: # 除去噪声点至少有2个点
                    s = silhouette_score(X[mask], lbls[mask])
                    if s > best_db_score:
                        best_db_score = s
                        best_db_labels = lbls
                        best_params = f"eps={eps},ms={ms}"

    m_db = compute_metrics(X, best_db_labels, y_true)
    m_db['method'] = f'DBSCAN({best_params})'
    results_list.append(m_db)
    df['dbscan_label'] = best_db_labels

    # ---------------- Agglomerative ----------------
    print("[训练] 正在运行 层次聚类 (Agglomerative)...")
    # 为了简化，直接使用 KMeans 找到的最佳 K，或者也可以重新搜索
    # 这里我们重新搜索一下
    best_agg_k = 2
    best_agg_sil = -1
    best_agg_labels = None
    
    for k in range(2, 9):
        agg = AgglomerativeClustering(n_clusters=k)
        lbls = agg.fit_predict(X)
        s = silhouette_score(X, lbls)
        if s > best_agg_sil:
            best_agg_sil = s
            best_agg_k = k
            best_agg_labels = lbls
            
    m_agg = compute_metrics(X, best_agg_labels, y_true)
    m_agg['method'] = f'Agglomerative(K={best_agg_k})'
    results_list.append(m_agg)
    df['agglomerative_label'] = best_agg_labels

    # --- 保存结果 ---
    # 1. 保存指标汇总
    metrics_df = pd.DataFrame(results_list)
    metrics_path = os.path.join(OUTPUT_DIR, 'clustering_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print("\n>>> 评估结果汇总:")
    print(metrics_df[['method', 'n_clusters', 'silhouette', 'ARI']])

    # 2. 保存带有标签的数据集
    out_file = os.path.join(OUTPUT_DIR, 'penguins_labeled.csv')
    df.to_csv(out_file, index=False)
    print(f"\n[训练] 完成。带标签数据已保存至: {out_file}")

if __name__ == "__main__":
    main()