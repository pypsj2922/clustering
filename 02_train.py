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
    has_noise = -1 in unique_labels
    
    # 基础指标：轮廓系数 (Silhouette)
    sil = -1
    if n_clusters > 1:
        mask = labels != -1
        if np.sum(mask) > 2:
            try:
                sil = silhouette_score(X[mask], labels[mask])
            except:
                sil = -1
    
    # 外部指标 (ARI)
    ari = -1
    ari_no_noise = -1
    if true_labels is not None:
        ari = adjusted_rand_score(true_labels, labels)
        # 如果有噪声点，额外计算去除噪声后的 ARI
        if has_noise:
            mask = labels != -1
            ari_no_noise = adjusted_rand_score(true_labels[mask], labels[mask])

    if has_noise and ari_no_noise != -1:
        noise_count = np.sum(labels == -1)
        noise_ratio = noise_count / len(labels) * 100
        print(f"[{method_name}] K={n_clusters} | Silhouette={sil:.4f} | ARI={ari:.4f} | ARI(去噪)={ari_no_noise:.4f} | 噪声={noise_count}({noise_ratio:.1f}%)")
    else:
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
    
    # A. 自动选择 K 的 GMM (基于 BIC)
    gmm_auto = GaussianMixture(n_components=best_gmm_k, random_state=RANDOM_STATE)
    df['gmm_auto'] = gmm_auto.fit_predict(X)
    compute_metrics(X, df['gmm_auto'], f"GMM(Auto K={best_gmm_k})", y_true)
    
    # B. 强制 K=3 的 GMM
    gmm_3 = GaussianMixture(n_components=3, random_state=RANDOM_STATE)
    df['gmm_k3'] = gmm_3.fit_predict(X)
    compute_metrics(X, df['gmm_k3'], "GMM(Force K=3)", y_true)

    # -------------------------------------------------------
    # 3. 层次聚类 (Agglomerative)
    # -------------------------------------------------------
    print("\n>>> 3. Agglomerative (层次聚类) 实验")
    
    # A. 自动选择 K (基于 Silhouette)
    best_agg_sil = -1
    best_agg_k = 2
    for k in range(2, 6):
        agg = AgglomerativeClustering(n_clusters=k)
        lbls = agg.fit_predict(X)
        score = silhouette_score(X, lbls)
        if score > best_agg_sil:
            best_agg_sil = score
            best_agg_k = k
    
    print(f"   Silhouette建议: K={best_agg_k} (Silhouette={best_agg_sil:.4f})")
    agg_auto = AgglomerativeClustering(n_clusters=best_agg_k)
    df['agg_auto'] = agg_auto.fit_predict(X)
    compute_metrics(X, df['agg_auto'], f"Agg(Auto K={best_agg_k})", y_true)
    
    # B. 强制 K=3
    agg_3 = AgglomerativeClustering(n_clusters=3)
    df['agg_k3'] = agg_3.fit_predict(X)
    compute_metrics(X, df['agg_k3'], "Agg(Force K=3)", y_true)

    # -------------------------------------------------------
    # 4. DBSCAN (密度聚类) 
    # -------------------------------------------------------
    print("\n>>> 4. DBSCAN (密度聚类) 实验")
    best_db_score = -2
    best_db_labels = None
    best_params = "N/A"
    best_noise_ratio = 100.0
    best_n_clusters = 0
    
    # 扩大参数搜索范围以降低噪声率，同时保证3分类
    # eps: 扩大范围，增加更大的值以降低噪声率
    eps_candidates = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    # min_samples: 减小最小值，尝试更小的值以降低噪声率
    min_samples_candidates = [2, 3, 4, 5, 8]
    
    # 目标：噪声率30%以下，簇数=3
    MAX_NOISE_RATIO = 30.0
    TARGET_CLUSTERS = 3

    print("   正在搜索参数（优先选择K=3且噪声率≤30%的组合）...")
    
    # 第一轮：优先寻找K=3且噪声率≤30%的组合
    candidates_k3 = []
    
    for eps in eps_candidates:
        for ms in min_samples_candidates:
            db = DBSCAN(eps=eps, min_samples=ms)
            lbls = db.fit_predict(X)
            
            # 检查是否有意义：至少要有2个簇
            unique = set(lbls)
            if len(unique) < 2:
                continue
            
            # 计算簇数（排除噪声点）
            n_clusters = len(unique) - (1 if -1 in unique else 0)
            noise_count = np.sum(lbls == -1)
            noise_ratio = noise_count / len(lbls) * 100
            
            # 计算轮廓系数和ARI (忽略噪声点)
            mask = lbls != -1
            if np.sum(mask) > 5:
                try:
                    sil_score = silhouette_score(X[mask], lbls[mask])
                except:
                    sil_score = -2
                
                # 计算ARI（正确率）
                try:
                    ari_score = adjusted_rand_score(y_true, lbls)
                except:
                    ari_score = -1
                
                # 记录所有K=3且噪声率≤30%的候选
                if n_clusters == TARGET_CLUSTERS and noise_ratio <= MAX_NOISE_RATIO:
                    candidates_k3.append({
                        'eps': eps,
                        'min_samples': ms,
                        'labels': lbls,
                        'n_clusters': n_clusters,
                        'noise_ratio': noise_ratio,
                        'silhouette': sil_score,
                        'ari': ari_score,
                        'params': f"eps={eps}, ms={ms}"
                    })
    
    # 从K=3的候选中选择最佳参数
    if candidates_k3:
        # 优先选择ARI（正确率）最高的，在噪声率≤30%的前提下
        candidates_k3.sort(key=lambda c: c['ari'], reverse=True)
        best_candidate = candidates_k3[0]
        
        best_db_labels = best_candidate['labels']
        best_params = best_candidate['params']
        best_noise_ratio = best_candidate['noise_ratio']
        best_n_clusters = best_candidate['n_clusters']
        best_db_score = best_candidate['silhouette']
        best_ari = best_candidate['ari']
        
        print(f"   找到 {len(candidates_k3)} 个K=3且噪声率≤30%的参数组合")
        print(f"   最佳选择（ARI最高）: {best_params}")
        print(f"   簇数={best_n_clusters}, 噪声率={best_noise_ratio:.1f}% (目标≤30%), ARI={best_ari:.4f}, Silhouette={best_db_score:.4f}")
    else:
        # 如果找不到噪声率≤30%的K=3组合，放宽条件：寻找所有K=3的组合，选择噪声率最低的
        print("   未找到噪声率≤30%的K=3组合，放宽条件搜索所有K=3组合...")
        candidates_k3_all = []
        
        for eps in eps_candidates:
            for ms in min_samples_candidates:
                db = DBSCAN(eps=eps, min_samples=ms)
                lbls = db.fit_predict(X)
                
                unique = set(lbls)
                if len(unique) < 2:
                    continue
                
                n_clusters = len(unique) - (1 if -1 in unique else 0)
                noise_count = np.sum(lbls == -1)
                noise_ratio = noise_count / len(lbls) * 100
                
                mask = lbls != -1
                if np.sum(mask) > 5:
                    try:
                        sil_score = silhouette_score(X[mask], lbls[mask])
                    except:
                        sil_score = -2
                    
                    # 计算ARI（正确率）
                    try:
                        ari_score = adjusted_rand_score(y_true, lbls)
                    except:
                        ari_score = -1
                    
                    # 记录所有K=3的候选（不管噪声率）
                    if n_clusters == TARGET_CLUSTERS:
                        candidates_k3_all.append({
                            'eps': eps,
                            'min_samples': ms,
                            'labels': lbls,
                            'n_clusters': n_clusters,
                            'noise_ratio': noise_ratio,
                            'silhouette': sil_score,
                            'ari': ari_score,
                            'params': f"eps={eps}, ms={ms}"
                        })
        
        if candidates_k3_all:
            # 优先选择ARI最高的K=3组合（在噪声率≤30%的前提下）
            # 如果噪声率都>30%，则选择ARI最高的
            candidates_within_limit = [c for c in candidates_k3_all if c['noise_ratio'] <= MAX_NOISE_RATIO]
            if candidates_within_limit:
                candidates_within_limit.sort(key=lambda c: c['ari'], reverse=True)
                best_candidate = candidates_within_limit[0]
            else:
                # 如果都没有≤30%的，选择ARI最高的
                candidates_k3_all.sort(key=lambda c: c['ari'], reverse=True)
                best_candidate = candidates_k3_all[0]
            
            best_db_labels = best_candidate['labels']
            best_params = best_candidate['params']
            best_noise_ratio = best_candidate['noise_ratio']
            best_n_clusters = best_candidate['n_clusters']
            best_db_score = best_candidate['silhouette']
            best_ari = best_candidate['ari']
            
            print(f"   找到 {len(candidates_k3_all)} 个K=3的参数组合")
            if best_noise_ratio <= MAX_NOISE_RATIO:
                print(f"   最佳选择（噪声率≤30%且ARI最高）: {best_params}")
            else:
                print(f"   最佳选择（ARI最高，但噪声率>{MAX_NOISE_RATIO}%）: {best_params}")
            print(f"   簇数={best_n_clusters}, 噪声率={best_noise_ratio:.1f}% (目标≤30%), ARI={best_ari:.4f}, Silhouette={best_db_score:.4f}")
        else:
            # 如果还是找不到K=3的，回退到原来的策略（但优先K>=3）
            print("   未找到K=3的参数组合，使用最佳轮廓系数策略（优先K≥3）...")
            for eps in eps_candidates:
                for ms in min_samples_candidates:
                    db = DBSCAN(eps=eps, min_samples=ms)
                    lbls = db.fit_predict(X)
                    
                    unique = set(lbls)
                    if len(unique) < 2:
                        continue
                    
                    n_clusters = len(unique) - (1 if -1 in unique else 0)
                    mask = lbls != -1
                    if np.sum(mask) > 5:
                        try:
                            score = silhouette_score(X[mask], lbls[mask])
                        except:
                            score = -2
                        
                        # 优先选择K>=3的（避免K=2）
                        if (n_clusters >= 3 and score > best_db_score) or \
                           (best_n_clusters < 3 and n_clusters >= 3):
                            best_db_score = score
                            best_db_labels = lbls
                            best_params = f"eps={eps}, ms={ms}"
                            best_n_clusters = n_clusters
                            best_noise_ratio = np.sum(lbls == -1) / len(lbls) * 100

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