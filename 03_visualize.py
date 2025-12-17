#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_visualize.py
功能：读取带标签数据，使用 PCA 降维至 2D，绘制散点图和混淆矩阵。
输出：result/*.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- 配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, 'result', 'penguins_labeled.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'result')

# 必须与训练时使用的特征一致
FEATURE_COLS = [
    'bill_length_mm_std', 'bill_depth_mm_std', 
    'flipper_length_mm_std', 'body_mass_g_std'
]

def plot_pca_scatter(X_2d, labels, title, filename, save_dir):
    """通用散点图绘制函数"""
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    
    # 专门为噪声点(DBSCAN -1)设置样式
    for lbl in unique_labels:
        mask = labels == lbl
        if lbl == -1:
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c='black', marker='x', label='Noise', alpha=0.5)
        else:
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f'Cluster {lbl}', alpha=0.8)
            
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot: {filename}")

def plot_confusion_matrix(y_true, y_pred, title, filename, save_dir):
    """通用混淆矩阵绘制函数"""
    cm = pd.crosstab(y_true, y_pred, rownames=['True Species'], colnames=['Cluster'])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    path = os.path.join(save_dir, filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {filename}")

def main():
    print(f"[可视化] 正在加载数据: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError("请先运行 02_train.py 生成带标签数据！")
        
    df = pd.read_csv(INPUT_FILE)
    X = df[FEATURE_COLS].values
    y_true = df['species']

    # 1. 再次执行 PCA (仅用于可视化坐标)
    # 注意：虽然可以在预处理阶段做，但在这里做更灵活，随时可以改 PCA 参数
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"[可视化] PCA 解释方差比: {pca.explained_variance_ratio_}")

    # 2. 绘制 真实标签 (Ground Truth)
    plot_pca_scatter(X_pca, y_true, "Ground Truth (Species)", "viz_pca_true_species.png", OUTPUT_DIR)

    # 3. 绘制 各模型聚类结果
    # 检查列是否存在（防止 02_train.py 没跑完整）
    if 'kmeans_label' in df.columns:
        plot_pca_scatter(X_pca, df['kmeans_label'], "KMeans Clustering", "viz_pca_kmeans.png", OUTPUT_DIR)
        plot_confusion_matrix(y_true, df['kmeans_label'], "Confusion: Species vs KMeans", "viz_cm_kmeans.png", OUTPUT_DIR)

    if 'dbscan_label' in df.columns:
        plot_pca_scatter(X_pca, df['dbscan_label'], "DBSCAN Clustering", "viz_pca_dbscan.png", OUTPUT_DIR)
        plot_confusion_matrix(y_true, df['dbscan_label'], "Confusion: Species vs DBSCAN", "viz_cm_dbscan.png", OUTPUT_DIR)

    if 'agglomerative_label' in df.columns:
        plot_pca_scatter(X_pca, df['agglomerative_label'], "Agglomerative Clustering", "viz_pca_agglomerative.png", OUTPUT_DIR)
        plot_confusion_matrix(y_true, df['agglomerative_label'], "Confusion: Species vs Agglomerative", "viz_cm_agglomerative.png", OUTPUT_DIR)

    print(f"[可视化] 全部完成。请查看 {OUTPUT_DIR} 目录。")

if __name__ == "__main__":
    main()