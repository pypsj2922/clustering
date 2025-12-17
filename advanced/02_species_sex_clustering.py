#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进阶实验2：物种+性别六分类聚类分析
探索能否将企鹅分成6类（3物种 × 2性别）
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


def plot_pca_scatter(X_2d, labels, title, filename, markers=None):
    """绘制PCA散点图"""
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", n_colors=len(unique_labels))
    
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        if lbl == -1:  # DBSCAN噪声点
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c='black', marker='x', 
                       label='Noise', alpha=0.6, s=40)
        else:
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], color=colors[i], 
                       label=f'{lbl}', alpha=0.7, s=50)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_confusion_matrix_aligned(y_true, y_pred, title, filename):
    """绘制对角线对齐的混淆矩阵"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_labels = sorted(set(y_true))
    pred_labels = sorted(set(y_pred))
    
    # 匈牙利算法匹配
    n = max(len(true_labels), len(pred_labels))
    cost_matrix = np.zeros((n, n))
    for i, tl in enumerate(true_labels):
        for j, pl in enumerate(pred_labels):
            cost_matrix[i, j] = -np.sum((y_true == tl) & (y_pred == pl))
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    ordered_pred = [pred_labels[j] if j < len(pred_labels) else f'C{j}' for j in col_ind[:len(true_labels)]]
    
    cm_data = []
    for tl in true_labels:
        row = [np.sum((y_true == tl) & (y_pred == pl)) if pl in pred_labels else 0 for pl in ordered_pred]
        cm_data.append(row)
    
    cm_df = pd.DataFrame(cm_data, index=true_labels, columns=ordered_pred)
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', square=True, cbar=False)
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Label')
    plt.title(title, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def plot_sex_comparison(df, filename):
    """绘制性别对特征的影响"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    features = FEATURE_COLS
    
    for ax, feat in zip(axes.flatten(), features):
        sns.boxplot(data=df, x='species', y=feat, hue='sex', ax=ax, palette='Set2')
        ax.set_title(f'{feat} by Species and Sex')
        ax.legend(title='Sex', loc='upper right')
    
    plt.suptitle('性别对企鹅身体测量的影响', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def main():
    print("=" * 60)
    print("进阶实验2：物种+性别六分类聚类分析")
    print("=" * 60)
    
    # 加载数据
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURE_COLS + ['species', 'sex'])
    # 过滤掉 '.' 等无效性别值
    df = df[df['sex'].isin(['FEMALE','MALE'])]
    
    # 创建6分类标签
    df['species_sex'] = df['species'] + '_' + df['sex']
    print(f"\n样本总数: {len(df)}")
    print(f"\n六分类分布:\n{df['species_sex'].value_counts()}")
    
    # 绘制性别对特征的影响
    plot_sex_comparison(df, 'sex_00_feature_comparison.png')
    
    # 统计分析：性别差异
    print("\n各物种雌雄体重差异:")
    for species in df['species'].unique():
        female_mass = df[(df['species'] == species) & (df['sex'] == 'FEMALE')]['body_mass_g'].mean()
        male_mass = df[(df['species'] == species) & (df['sex'] == 'MALE')]['body_mass_g'].mean()
        diff = male_mass - female_mass
        print(f"  {species}: 雌性 {female_mass:.0f}g, 雄性 {male_mass:.0f}g, 差异 {diff:.0f}g ({diff/female_mass*100:.1f}%)")
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS])
    y_species = df['species'].values
    y_sex = df['sex'].values
    y_species_sex = df['species_sex'].values
    
    # PCA降维
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
    
    # 绘制真实分布
    plot_pca_scatter(X_pca, y_species_sex, "真实分布 (物种+性别)", "sex_01_truth.png")
    plot_pca_scatter(X_pca, y_species, "真实分布 (仅物种)", "sex_02_species_only.png")
    
    # ========== 聚类实验 ==========
    print("\n" + "=" * 60)
    print("聚类实验")
    print("=" * 60)
    
    # --- 实验A: K=3 (仅区分物种) ---
    print("\n>>> 实验A: K=3 聚类 (目标：区分3个物种)")
    
    km3 = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    labels_km3 = km3.fit_predict(X)
    ari_species_km3 = adjusted_rand_score(y_species, labels_km3)
    print(f"[KMeans K=3] ARI(vs物种)={ari_species_km3:.4f}")
    plot_pca_scatter(X_pca, labels_km3, f"KMeans K=3 (ARI vs Species={ari_species_km3:.3f})", "sex_03_kmeans_k3.png")
    plot_confusion_matrix_aligned(y_species, labels_km3, "KMeans K=3 vs Species", "sex_03_kmeans_k3_cm.png")
    
    # --- 实验B: K=6 (区分物种+性别) ---
    print("\n>>> 实验B: K=6 聚类 (目标：区分6类)")
    
    km6 = KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=10)
    labels_km6 = km6.fit_predict(X)
    ari_6_km = adjusted_rand_score(y_species_sex, labels_km6)
    sil_km6 = silhouette_score(X, labels_km6)
    print(f"[KMeans K=6] Silhouette={sil_km6:.4f} | ARI(vs 6类)={ari_6_km:.4f}")
    plot_pca_scatter(X_pca, labels_km6, f"KMeans K=6 (ARI={ari_6_km:.3f})", "sex_04_kmeans_k6.png")
    plot_confusion_matrix_aligned(y_species_sex, labels_km6, "KMeans K=6 vs Species+Sex", "sex_04_kmeans_k6_cm.png")
    
    # GMM K=6
    gmm6 = GaussianMixture(n_components=6, random_state=RANDOM_STATE, n_init=5)
    labels_gmm6 = gmm6.fit_predict(X)
    ari_6_gmm = adjusted_rand_score(y_species_sex, labels_gmm6)
    sil_gmm6 = silhouette_score(X, labels_gmm6)
    print(f"[GMM K=6] Silhouette={sil_gmm6:.4f} | ARI(vs 6类)={ari_6_gmm:.4f}")
    plot_pca_scatter(X_pca, labels_gmm6, f"GMM K=6 (ARI={ari_6_gmm:.3f})", "sex_05_gmm_k6.png")
    plot_confusion_matrix_aligned(y_species_sex, labels_gmm6, "GMM K=6 vs Species+Sex", "sex_05_gmm_k6_cm.png")
    
    # Agglomerative K=6
    agg6 = AgglomerativeClustering(n_clusters=6)
    labels_agg6 = agg6.fit_predict(X)
    ari_6_agg = adjusted_rand_score(y_species_sex, labels_agg6)
    sil_agg6 = silhouette_score(X, labels_agg6)
    print(f"[Agglomerative K=6] Silhouette={sil_agg6:.4f} | ARI(vs 6类)={ari_6_agg:.4f}")
    plot_pca_scatter(X_pca, labels_agg6, f"Agglomerative K=6 (ARI={ari_6_agg:.3f})", "sex_06_agg_k6.png")
    plot_confusion_matrix_aligned(y_species_sex, labels_agg6, "Agg K=6 vs Species+Sex", "sex_06_agg_k6_cm.png")
    
    # DBSCAN 实验
    # 注意：DBSCAN 无法指定簇数，它根据密度自动发现簇
    print("\n>>> DBSCAN 实验 (注意：DBSCAN无法指定簇数)")
    best_db_ari = -1
    best_db_labels = None
    best_params = "N/A"
    best_n_clusters = 0
    
    # 更细粒度的参数搜索，直接优化ARI
    for eps in np.arange(0.15, 1.2, 0.05):
        for ms in [2, 3, 4, 5, 6, 8, 10]:
            db = DBSCAN(eps=eps, min_samples=ms)
            lbls = db.fit_predict(X)
            unique = set(lbls)
            n_clusters = len(unique) - (1 if -1 in unique else 0)
            noise_count = np.sum(lbls == -1)
            noise_pct = noise_count / len(lbls) * 100
            
            # 至少2个簇，噪声不超过30%
            if n_clusters >= 2 and noise_pct < 30:
                ari = adjusted_rand_score(y_species_sex, lbls)
                if ari > best_db_ari:
                    best_db_ari = ari
                    best_db_labels = lbls
                    best_params = f"eps={eps:.2f}, ms={ms}"
                    best_n_clusters = n_clusters
    
    ari_6_db = -1
    if best_db_labels is not None:
        mask = best_db_labels != -1
        ari_6_db = best_db_ari
        ari_db_no_noise = adjusted_rand_score(y_species_sex[mask], best_db_labels[mask])
        noise_ratio = np.sum(~mask) / len(best_db_labels) * 100
        print(f"[DBSCAN({best_params})] 发现{best_n_clusters}个簇 | ARI={ari_6_db:.4f} | ARI(去噪)={ari_db_no_noise:.4f} | 噪声={noise_ratio:.1f}%")
        plot_pca_scatter(X_pca, best_db_labels, f"DBSCAN ({best_n_clusters}簇, ARI={ari_6_db:.3f})", "sex_07_dbscan.png")
        
        # 为DBSCAN单独绘制混淆矩阵（重新编号簇，排除噪声）
        # 将簇编号重新映射为连续的 0,1,2,...
        db_labels_clean = best_db_labels.copy()
        unique_clusters = sorted([c for c in set(db_labels_clean) if c != -1])
        cluster_map = {old: new for new, old in enumerate(unique_clusters)}
        cluster_map[-1] = -1  # 噪声保持-1
        db_labels_renamed = np.array([cluster_map[c] for c in db_labels_clean])
        
        # 只对非噪声点绘制混淆矩阵
        plot_confusion_matrix_aligned(y_species_sex[mask], db_labels_renamed[mask], 
                                      f"DBSCAN vs Species+Sex (去噪后, {best_n_clusters}簇)", "sex_07_dbscan_cm.png")
    else:
        print("[DBSCAN] 未找到有效参数")
    
    # --- 实验C: 自动选择K ---
    print("\n>>> 实验C: 自动选择最佳K")
    best_k = 2
    best_sil = -1
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        lbls = km.fit_predict(X)
        sil = silhouette_score(X, lbls)
        if sil > best_sil:
            best_sil = sil
            best_k = k
    print(f"Silhouette最佳K={best_k} (score={best_sil:.4f})")
    
    # 特征pairplot
    print("\n>>> 绘制特征Pairplot...")
    df_plot = df[FEATURE_COLS + ['species_sex']].copy()
    df_plot.columns = ['Culmen Length', 'Culmen Depth', 'Flipper Length', 'Body Mass', 'Species_Sex']
    g = sns.pairplot(df_plot, hue='Species_Sex', diag_kind='hist', 
                     height=2.0, plot_kws={'alpha': 0.6}, diag_kws={'alpha': 0.5})
    g.fig.suptitle("企鹅 - 特征与物种+性别散点可视化", fontsize=14, y=1.02)
    plt.savefig(os.path.join(OUTPUT_DIR, 'sex_pairplot.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: sex_pairplot.png")
    
    # 结论
    print("\n" + "=" * 60)
    print("结论分析:")
    print("=" * 60)
    print(f"1. K=3聚类 vs 物种: ARI={ari_species_km3:.4f}")
    print(f"2. K=6聚类 vs 物种+性别:")
    print(f"   - KMeans: ARI={ari_6_km:.4f}")
    print(f"   - GMM: ARI={ari_6_gmm:.4f}")
    print(f"   - Agglomerative: ARI={ari_6_agg:.4f}")
    print(f"   - DBSCAN: ARI={ari_6_db:.4f}")
    print(f"3. 自动选择的最佳K={best_k}")
    
    if ari_6_km > 0.5:
        print("\n性别因素对聚类有显著影响，可以较好地区分6类。")
    elif ari_6_km > 0.3:
        print("\n性别因素有一定影响，但区分效果一般。")
    else:
        print("\n仅凭身体测量数据难以同时区分物种和性别，性别差异在物种内部不够明显。")


if __name__ == "__main__":
    main()

