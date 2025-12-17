#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进阶实验1：Adelie企鹅岛屿聚类分析
探索不同岛屿上的Adelie企鹅是否存在形态差异
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

# --- 配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), 'data', 'penguins_size.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'result')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
RANDOM_STATE = 42

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_pca_scatter(X_2d, labels, title, filename):
    """绘制PCA散点图"""
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = sns.color_palette("Set2", n_colors=len(unique_labels))
    
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        if lbl == -1:  # DBSCAN噪声点
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c='black', marker='x', 
                       label='Noise', alpha=0.6, s=40)
        else:
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], color=colors[i], 
                       label=f'{lbl}', alpha=0.7, s=60)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_confusion_matrix_aligned(y_true, y_pred, title, filename):
    """绘制对角线对齐的混淆矩阵"""
    true_labels = sorted(set(y_true))
    pred_labels = sorted(set(y_pred))
    
    # 匈牙利算法匹配
    cost_matrix = np.zeros((len(true_labels), len(pred_labels)))
    for i, tl in enumerate(true_labels):
        for j, pl in enumerate(pred_labels):
            cost_matrix[i, j] = -np.sum((np.array(y_true) == tl) & (np.array(y_pred) == pl))
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    ordered_pred = [pred_labels[j] for j in col_ind]
    
    cm_data = []
    for tl in true_labels:
        row = [np.sum((np.array(y_true) == tl) & (np.array(y_pred) == pl)) for pl in ordered_pred]
        cm_data.append(row)
    
    cm_df = pd.DataFrame(cm_data, index=true_labels, columns=ordered_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', square=True, cbar=False)
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Label')
    plt.title(title, fontsize=14)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def main():
    print("=" * 60)
    print("进阶实验1：Adelie企鹅岛屿聚类分析")
    print("=" * 60)
    
    # 加载数据
    df = pd.read_csv(DATA_PATH)
    
    # 只保留Adelie企鹅
    df_adelie = df[df['species'] == 'Adelie'].copy()
    df_adelie = df_adelie.dropna(subset=FEATURE_COLS + ['island'])
    print(f"\nAdelie企鹅样本数: {len(df_adelie)}")
    print(f"岛屿分布:\n{df_adelie['island'].value_counts()}")
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(df_adelie[FEATURE_COLS])
    y_island = df_adelie['island'].values
    
    # PCA降维
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
    
    # 绘制真实岛屿分布
    plot_pca_scatter(X_pca, y_island, "Adelie企鹅 - 真实岛屿分布", "adelie_00_truth.png")
    
    # 特征分布对比
    print("\n各岛屿Adelie企鹅特征均值:")
    print(df_adelie.groupby('island')[FEATURE_COLS].mean().round(2))
    
    # 聚类实验
    print("\n>>> 聚类实验 (K=3，对应3个岛屿)")
    
    # KMeans
    km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    labels_km = km.fit_predict(X)
    sil_km = silhouette_score(X, labels_km)
    ari_km = adjusted_rand_score(y_island, labels_km)
    print(f"[KMeans] Silhouette={sil_km:.4f} | ARI={ari_km:.4f}")
    plot_pca_scatter(X_pca, labels_km, f"Adelie - KMeans (ARI={ari_km:.3f})", "adelie_01_kmeans.png")
    plot_confusion_matrix_aligned(y_island, labels_km, "Adelie KMeans vs Island", "adelie_01_kmeans_cm.png")
    
    # GMM
    gmm = GaussianMixture(n_components=3, random_state=RANDOM_STATE)
    labels_gmm = gmm.fit_predict(X)
    sil_gmm = silhouette_score(X, labels_gmm)
    ari_gmm = adjusted_rand_score(y_island, labels_gmm)
    print(f"[GMM] Silhouette={sil_gmm:.4f} | ARI={ari_gmm:.4f}")
    plot_pca_scatter(X_pca, labels_gmm, f"Adelie - GMM (ARI={ari_gmm:.3f})", "adelie_02_gmm.png")
    plot_confusion_matrix_aligned(y_island, labels_gmm, "Adelie GMM vs Island", "adelie_02_gmm_cm.png")
    
    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=3)
    labels_agg = agg.fit_predict(X)
    sil_agg = silhouette_score(X, labels_agg)
    ari_agg = adjusted_rand_score(y_island, labels_agg)
    print(f"[Agglomerative] Silhouette={sil_agg:.4f} | ARI={ari_agg:.4f}")
    plot_pca_scatter(X_pca, labels_agg, f"Adelie - Agglomerative (ARI={ari_agg:.3f})", "adelie_03_agg.png")
    plot_confusion_matrix_aligned(y_island, labels_agg, "Adelie Agg vs Island", "adelie_03_agg_cm.png")
    
    # DBSCAN
    print("\n>>> DBSCAN 实验")
    best_db_score = -2
    best_db_labels = None
    best_params = "N/A"
    for eps in [0.3, 0.5, 0.7, 1.0, 1.5]:
        for ms in [3, 5, 10]:
            db = DBSCAN(eps=eps, min_samples=ms)
            lbls = db.fit_predict(X)
            unique = set(lbls)
            n_clusters = len(unique) - (1 if -1 in unique else 0)
            if n_clusters >= 2:
                mask = lbls != -1
                if np.sum(mask) > 5:
                    try:
                        score = silhouette_score(X[mask], lbls[mask])
                        if score > best_db_score:
                            best_db_score = score
                            best_db_labels = lbls
                            best_params = f"eps={eps}, ms={ms}"
                    except:
                        pass
    
    if best_db_labels is not None:
        mask = best_db_labels != -1
        ari_db = adjusted_rand_score(y_island, best_db_labels)
        ari_db_no_noise = adjusted_rand_score(y_island[mask], best_db_labels[mask])
        noise_ratio = np.sum(~mask) / len(best_db_labels) * 100
        print(f"[DBSCAN({best_params})] Silhouette={best_db_score:.4f} | ARI={ari_db:.4f} | ARI(去噪)={ari_db_no_noise:.4f} | 噪声={noise_ratio:.1f}%")
        plot_pca_scatter(X_pca, best_db_labels, f"Adelie - DBSCAN (ARI={ari_db:.3f})", "adelie_04_dbscan.png")
    else:
        print("[DBSCAN] 未找到有效参数")
    
    # 特征pairplot
    print("\n>>> 绘制特征Pairplot...")
    df_plot = df_adelie[FEATURE_COLS + ['island']].copy()
    df_plot.columns = ['Culmen Length', 'Culmen Depth', 'Flipper Length', 'Body Mass', 'Island']
    penguin_colors = {'Biscoe': '#FF8C00', 'Dream': '#A034F0', 'Torgersen': '#057076'}
    g = sns.pairplot(df_plot, hue='Island', diag_kind='hist', palette=penguin_colors, 
                     height=2.2, plot_kws={'alpha': 0.7}, diag_kws={'alpha': 0.6})
    g.fig.suptitle("Adelie企鹅 - 特征与岛屿散点可视化", fontsize=14, y=1.02)
    plt.savefig(os.path.join(OUTPUT_DIR, 'adelie_pairplot.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: adelie_pairplot.png")
    
    # 结论
    print("\n" + "=" * 60)
    print("结论分析:")
    print("=" * 60)
    if max(ari_km, ari_gmm, ari_agg) < 0.1:
        print("ARI值很低，说明不同岛屿的Adelie企鹅在形态特征上没有明显差异。")
        print("聚类算法无法仅凭身体测量数据区分来自不同岛屿的Adelie企鹅。")
    else:
        print("存在一定的岛屿差异，聚类结果与岛屿分布有一定相关性。")


if __name__ == "__main__":
    main()

