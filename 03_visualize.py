#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_visualize.py
功能：
1. 读取训练结果 (penguins_labeled_full.csv)。
2. 绘制 5 种特定模型结果 (KMeans Auto/K3, GMM K3, Agg K3, DBSCAN)。
3. 生成 PCA 散点图和混淆矩阵。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- 配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, 'result', 'penguins_labeled_full.csv') 
OUTPUT_DIR = os.path.join(BASE_DIR, 'result')

FEATURE_COLS = [
    'bill_length_mm_std', 'bill_depth_mm_std', 
    'flipper_length_mm_std', 'body_mass_g_std'
]

def plot_pca_scatter(X_2d, labels, title, filename, save_dir):
    """通用散点图绘制函数"""
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    
    # 颜色盘
    colors = sns.color_palette("tab10", n_colors=len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        # 单独处理 DBSCAN 可能产生的噪声点 -1
        if lbl == -1:
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c='black', marker='x', label='Noise', alpha=0.6, s=40)
        else:
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], color=colors[i], label=f'Cluster {lbl}', alpha=0.8, s=60)
            
    plt.title(title, fontsize=14)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # 图例放外面，防止遮挡
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved scatter plot: {filename}")

def plot_confusion_matrix(y_true, y_pred, title, filename, save_dir):
    """通用混淆矩阵绘制函数"""
    cm = pd.crosstab(y_true, y_pred, rownames=['True Species'], colnames=['Cluster'])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, cbar=False)
    plt.title(title, fontsize=14)
    path = os.path.join(save_dir, filename)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {filename}")

def main():
    print(f"[可视化] 正在加载数据: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"找不到文件 {INPUT_FILE}。请确保先运行了 02_train_v2.py！")
             
    df = pd.read_csv(INPUT_FILE)
    
    if not all(col in df.columns for col in FEATURE_COLS):
        raise ValueError(f"数据中缺少特征列: {FEATURE_COLS}")
        
    X = df[FEATURE_COLS].values
    y_true = df['species']

    # 1. 执行 PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"[可视化] PCA 解释方差比: {pca.explained_variance_ratio_}")

    # 2. 绘制 真实标签
    plot_pca_scatter(X_pca, y_true, "Ground Truth (Species)", "viz_00_truth.png", OUTPUT_DIR)

    # 3. 绘制 02_train_v2.py 生成的模型结果
    # 格式: (CSV列名, 图表标题, 输出文件前缀)
    plot_tasks = [
        ('kmeans_auto', 'KMeans (Auto Search)', 'viz_01_kmeans_auto'),
        ('kmeans_k3',   'KMeans (Force K=3)',   'viz_02_kmeans_k3'),
        ('gmm_k3',      'GMM (Force K=3)',      'viz_03_gmm_k3'),
        ('agg_k3',      'Agglomerative (K=3)',  'viz_04_agg_k3'),
        ('dbscan_best', 'DBSCAN (Best Param)',  'viz_05_dbscan'), # 新增 DBSCAN
    ]

    for col, title, fname_prefix in plot_tasks:
        if col in df.columns:
            print(f"--> 处理任务: {title}")
            plot_pca_scatter(
                X_pca, 
                df[col], 
                f"{title} - PCA Projection", 
                f"{fname_prefix}_scatter.png", 
                OUTPUT_DIR
            )
            plot_confusion_matrix(
                y_true, 
                df[col], 
                f"Confusion Matrix: {title}", 
                f"{fname_prefix}_cm.png", 
                OUTPUT_DIR
            )
        else:
            print(f"警告: 数据中未找到列 '{col}'，可能是 DBSCAN 没有找到合适的参数。")

    print(f"\n[可视化] 全部完成。请查看 {OUTPUT_DIR} 目录。")
    print("提示: DBSCAN 图中黑色的叉(x)代表噪声点(Noise)。")

if __name__ == "__main__":
    main()