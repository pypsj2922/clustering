#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_statistical_test.py
统计检验：为聚类分析提供理论依据
1. t-test: 检验性别差异（2组）
2. ANOVA: 检验岛屿差异（3组）、物种+性别差异（6组）
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# --- 配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), 'data', 'penguins_size.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'result')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
FEATURE_NAMES = ['喙长(mm)', '喙深(mm)', '鳍长(mm)', '体重(g)']

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def ttest_by_sex(df, species=None):
    """t检验：比较雌雄差异"""
    if species:
        df_sub = df[df['species'] == species]
        title = f"{species}企鹅"
    else:
        df_sub = df
        title = "所有企鹅"
    
    print(f"\n>>> {title} - 性别差异 t-test")
    print("-" * 60)
    
    male = df_sub[df_sub['sex'] == 'MALE']
    female = df_sub[df_sub['sex'] == 'FEMALE']
    
    results = []
    for feat, name in zip(FEATURE_COLS, FEATURE_NAMES):
        t_stat, p_value = stats.ttest_ind(male[feat].dropna(), female[feat].dropna())
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        male_mean = male[feat].mean()
        female_mean = female[feat].mean()
        diff_pct = (male_mean - female_mean) / female_mean * 100
        
        print(f"  {name:12s}: 雌={female_mean:7.1f}, 雄={male_mean:7.1f}, "
              f"差异={diff_pct:+5.1f}%, t={t_stat:6.2f}, p={p_value:.4f} {sig}")
        results.append({'feature': name, 't_stat': t_stat, 'p_value': p_value, 'significant': p_value < 0.05})
    
    return pd.DataFrame(results)


def anova_test(df, group_col, group_name):
    """ANOVA检验：比较多组差异"""
    print(f"\n>>> {group_name} - ANOVA 检验")
    print("-" * 60)
    
    groups = df[group_col].unique()
    print(f"  分组: {list(groups)} (共{len(groups)}组)")
    
    results = []
    for feat, name in zip(FEATURE_COLS, FEATURE_NAMES):
        group_data = [df[df[group_col] == g][feat].dropna() for g in groups]
        f_stat, p_value = stats.f_oneway(*group_data)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        # 计算效应量 eta-squared
        grand_mean = df[feat].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_data)
        ss_total = sum((df[feat] - grand_mean)**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        
        print(f"  {name:12s}: F={f_stat:8.2f}, p={p_value:.6f}, η²={eta_sq:.3f} {sig}")
        results.append({
            'feature': name, 'f_stat': f_stat, 'p_value': p_value, 
            'eta_squared': eta_sq, 'significant': p_value < 0.05
        })
    
    return pd.DataFrame(results)


def plot_group_comparison(df, group_col, title, filename):
    """绘制分组对比箱线图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for ax, feat, name in zip(axes.flatten(), FEATURE_COLS, FEATURE_NAMES):
        sns.boxplot(data=df, x=group_col, y=feat, hue=group_col, ax=ax, palette='Set2', legend=False)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def main():
    print_section("统计检验分析 - 为聚类提供理论依据")
    
    # 加载数据
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURE_COLS + ['species', 'sex', 'island'])
    df = df[df['sex'].isin(['MALE', 'FEMALE'])]
    df['species_sex'] = df['species'] + '_' + df['sex']
    
    print(f"\n数据概况: {len(df)} 样本")
    print(f"物种分布: {dict(df['species'].value_counts())}")
    print(f"性别分布: {dict(df['sex'].value_counts())}")
    print(f"岛屿分布: {dict(df['island'].value_counts())}")
    
    # =========================================================
    # 实验1：Adelie企鹅岛屿差异检验
    # =========================================================
    print_section("实验1：Adelie企鹅 - 不同岛屿是否有显著差异？")
    
    df_adelie = df[df['species'] == 'Adelie']
    print(f"\nAdelie样本数: {len(df_adelie)}")
    print(f"岛屿分布: {dict(df_adelie['island'].value_counts())}")
    
    anova_adelie = anova_test(df_adelie, 'island', 'Adelie企鹅按岛屿分组')
    plot_group_comparison(df_adelie, 'island', 'Adelie企鹅 - 不同岛屿特征对比', 'stat_adelie_island.png')
    
    # 结论
    sig_count = anova_adelie['significant'].sum()
    print(f"\n结论: {sig_count}/{len(FEATURE_COLS)} 个特征在岛屿间存在显著差异(p<0.05)")
    if sig_count == 0:
        print("  → 不同岛屿的Adelie企鹅在形态上无显著差异，聚类可能难以区分岛屿")
    elif sig_count < len(FEATURE_COLS):
        print("  → 部分特征存在差异，聚类可能有一定区分能力")
    else:
        print("  → 所有特征都有显著差异，聚类应该能较好区分岛屿")
    
    # =========================================================
    # 实验2：性别差异检验
    # =========================================================
    print_section("实验2：性别差异检验 (t-test)")
    
    # 全体企鹅性别差异
    ttest_all = ttest_by_sex(df)
    
    # 各物种内部性别差异
    for species in df['species'].unique():
        ttest_by_sex(df, species)
    
    plot_group_comparison(df, 'sex', '所有企鹅 - 性别特征对比', 'stat_sex_comparison.png')
    
    sig_count = ttest_all['significant'].sum()
    print(f"\n结论: {sig_count}/{len(FEATURE_COLS)} 个特征在性别间存在显著差异(p<0.05)")
    
    # =========================================================
    # 实验3：物种+性别 6分类差异检验
    # =========================================================
    print_section("实验3：物种+性别 6分类 - ANOVA检验")
    
    anova_6class = anova_test(df, 'species_sex', '6分类(物种×性别)')
    plot_group_comparison(df, 'species_sex', '企鹅 - 物种×性别 特征对比', 'stat_species_sex.png')
    
    # 对比：仅物种分组
    print("\n>>> 对比：仅按物种分组")
    anova_species = anova_test(df, 'species', '3分类(仅物种)')
    
    # 效应量对比
    print("\n>>> 效应量(η²)对比：物种 vs 物种+性别")
    print("-" * 60)
    for i, name in enumerate(FEATURE_NAMES):
        eta_species = anova_species.iloc[i]['eta_squared']
        eta_6class = anova_6class.iloc[i]['eta_squared']
        improvement = (eta_6class - eta_species) / eta_species * 100 if eta_species > 0 else 0
        print(f"  {name:12s}: 物种η²={eta_species:.3f}, 6分类η²={eta_6class:.3f}, 提升={improvement:+.1f}%")
    
    # =========================================================
    # 总结
    # =========================================================
    print_section("总结与聚类建议")
    
    print("""
1. Adelie企鹅岛屿差异:
   - 如果ANOVA不显著，说明不同岛屿的Adelie企鹅形态相似
   - 聚类算法难以仅凭身体测量区分岛屿来源

2. 性别差异:
   - 雄性通常比雌性体型更大（体重、鳍长等）
   - 但性别差异小于物种间差异

3. 6分类(物种×性别):
   - 如果η²提升明显，说明加入性别因素能解释更多变异
   - 聚类K=6可能比K=3更合理
   - 但如果提升不大，说明性别差异在物种内部不够明显

4. 聚类建议:
   - 先看ANOVA结果决定是否值得做细分类聚类
   - η² > 0.14 为大效应，0.06-0.14 为中等效应，< 0.06 为小效应
""")
    
    # 保存统计结果
    results_df = pd.DataFrame({
        '分析': ['Adelie岛屿', '性别(全体)', '6分类', '3分类(物种)'],
        '显著特征数': [
            anova_adelie['significant'].sum(),
            ttest_all['significant'].sum(),
            anova_6class['significant'].sum(),
            anova_species['significant'].sum()
        ],
        '平均η²/效应量': [
            anova_adelie['eta_squared'].mean(),
            '-',
            anova_6class['eta_squared'].mean(),
            anova_species['eta_squared'].mean()
        ]
    })
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'stat_summary.csv'), index=False)
    print(f"\n统计结果已保存至: stat_summary.csv")


if __name__ == "__main__":
    main()

