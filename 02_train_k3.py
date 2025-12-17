#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train_v2.py
功能：
1. 训练 KMeans, Agglomerative, GMM。
2. 新增：恢复 DBSCAN 算法（网格搜索最佳参数）。
3. 解决"K=2 vs K=3"问题：强制保存 K=3 的模型结果。
4. 使用 AIC/BIC 为 GMM 选择参数。
输出：result/penguins_labeled_full.csv
"""

import os
# 限制线程数，避免某些环境下的警告
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score

# --- 配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, 'result', 'penguins_processed.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'result')
RANDOM_STATE = 42

FEATURE_COLS = [
    'bill_length_mm_std', 'bill_depth_mm_std', 
    'flipper_length_mm_std', 'body_mass_g_std'
]

def compute_metrics(X, labels, method_name, true_labels=None):
    """计算指标并打印"""
    unique_labels = np.unique(labels)
    # DBSCAN 的噪声点标记为 -1，计算簇数量时要排除
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # 基础指标：轮廓系数 (Silhouette)
    # 如果只有一个簇或者全是噪声，无法计算轮廓系数
    sil = -1
    if n_clusters > 1:
        # 通常计算轮廓系数时不包含噪声点，或者将噪声点视为单独的一类
        # 这里我们只对非噪声点进行计算，以评价核心簇的质量
        mask = labels != -1
        if np.sum(mask) > 2: # 至少要有3个点才能算
            try:
                sil = silhouette_score(X[mask], labels[mask])
            except:
                sil = -1
    
    # 外部指标 (ARI) - 代表"不仅仅分得开，而且分得对不对"
    ari = -1
    if true_labels is not None:
        ari = adjusted_rand_score(true_labels, labels)

    print(f"[{method_name}] K={n_clusters} | Silhouette={sil:.4f} | ARI(准确度)={ari:.4f}")
    return labels, sil, ari

def main():
    print(f"[训练] 加载数据: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError("找不到预处理文件，请先运行 01_preprocess.py")

    df = pd.read_csv(INPUT_FILE)
    X = df[FEATURE_COLS].values
    y_true = df['species'].values

    # -------------------------------------------------------
    # 1. KMeans (对比 自动K vs 强制K=3)
    # -------------------------------------------------------
    print("\n>>> 1. KMeans 实验")
    
    # A. 自动搜索最佳 K (基于 Silhouette)
    best_k_sil = -1
    best_k_val = 0
    
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        lbls = km.fit_predict(X)
        score = silhouette_score(X, lbls)
        if score > best_k_sil:
            best_k_sil = score
            best_k_val = k
            
    print(f"   自动搜索建议: K={best_k_val} (Silhouette={best_k_sil:.4f})")
    
    # B. 保存 K=Auto 的结果
    km_auto = KMeans(n_clusters=best_k_val, random_state=RANDOM_STATE, n_init=10)
    df['kmeans_auto'] = km_auto.fit_predict(X)
    
    # C. 强制保存 K=3 的结果
    km_force = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    df['kmeans_k3'] = km_force.fit_predict(X)
    compute_metrics(X, df['kmeans_k3'], "KMeans(Force K=3)", y_true)

    # -------------------------------------------------------
    # 2. GMM (高斯混合模型)
    # -------------------------------------------------------
    print("\n>>> 2. Gaussian Mixture Model (GMM) 实验")
    best_bic = np.inf
    best_gmm_k = 0
    
    for k in range(2, 6):
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=5)
        gmm.fit(X)
        bic = gmm.bic(X) 
        if bic < best_bic:
            best_bic = bic
            best_gmm_k = k
            
    print(f"   BIC建议: K={best_gmm_k} (BIC={best_bic:.2f})")
    
    # 强制 K=3 的 GMM
    gmm_3 = GaussianMixture(n_components=3, random_state=RANDOM_STATE)
    df['gmm_k3'] = gmm_3.fit_predict(X)
    compute_metrics(X, df['gmm_k3'], "GMM(Force K=3)", y_true)

    # -------------------------------------------------------
    # 3. 层次聚类 (Agglomerative)
    # -------------------------------------------------------
    print("\n>>> 3. Agglomerative (层次聚类) 实验")
    agg_3 = AgglomerativeClustering(n_clusters=3)
    df['agg_k3'] = agg_3.fit_predict(X)
    compute_metrics(X, df['agg_k3'], "Agg(Force K=3)", y_true)

    # -------------------------------------------------------
    # 4. DBSCAN (密度聚类) - 新增部分
    # -------------------------------------------------------
    print("\n>>> 4. DBSCAN (密度聚类) 实验")
    best_db_score = -2
    best_db_labels = None
    best_params = "N/A"
    
    # 网格搜索
    eps_candidates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
    min_samples_candidates = [3, 5, 10]

    for eps in eps_candidates:
        for ms in min_samples_candidates:
            db = DBSCAN(eps=eps, min_samples=ms)
            lbls = db.fit_predict(X)
            
            # 检查是否有意义：至少要有2个簇(含噪声算一种的话)，或者有大于1个真实簇
            unique = set(lbls)
            # 如果只有噪声(-1)或者只有一类(0)，则跳过
            if len(unique) < 2:
                continue
                
            # 计算轮廓系数 (忽略噪声点)
            mask = lbls != -1
            if np.sum(mask) > 5: # 至少有一些点不是噪声
                try:
                    score = silhouette_score(X[mask], lbls[mask])
                except:
                    score = -2
                
                if score > best_db_score:
                    best_db_score = score
                    best_db_labels = lbls
                    best_params = f"eps={eps}, ms={ms}"

    if best_db_labels is not None:
        print(f"   DBSCAN 最佳参数: {best_params}")
        df['dbscan_best'] = best_db_labels
        compute_metrics(X, df['dbscan_best'], f"DBSCAN({best_params})", y_true)
    else:
        print("   DBSCAN 未找到有效参数（数据可能过于密集或参数范围不合适）。")
        # 如果失败，填入全 -1 防止报错
        df['dbscan_best'] = -1

    # -------------------------------------------------------
    # 保存
    # -------------------------------------------------------
    out_file = os.path.join(OUTPUT_DIR, 'penguins_labeled_full.csv')
    df.to_csv(out_file, index=False)
    print(f"\n[完成] 结果已保存: {out_file}")

if __name__ == "__main__":
    main()