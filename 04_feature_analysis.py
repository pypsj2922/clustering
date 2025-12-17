#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_feature_analysis.py
功能：生成特征成对散点图矩阵 (Pairplot)，模仿用户上传的图表风格。
说明：
    使用原始数据进行可视化，以便保留真实的物理单位 (mm, g)。
    支持中文标题显示。
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 这里我们读取【原始数据】，因为可视化通常需要看真实的物理含义（毫米、克）
DATA_PATH = os.path.join(BASE_DIR, 'data', 'penguins_size.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'result')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 解决 Matplotlib 中文显示问题 (根据你的系统选择字体)
# Windows 通常用 'SimHei' (黑体) 或 'Microsoft YaHei' (微软雅黑)
# Mac 通常用 'Arial Unicode MS' 或 'PingFang SC'
# Linux 可能需要安装相应字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

def main():
    print(f"[绘图] 正在加载数据: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"找不到原始数据文件: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # 1. 数据清洗：Pairplot 不能处理空值，先删除空行
    df_clean = df.dropna().copy()

    # 2. 列名美化
    # 为了让图表坐标轴跟你的截图一样显示 "Culmen Length (mm)" 这种漂亮的格式
    # 我们建立一个映射字典
    rename_map = {
        'culmen_length_mm': 'Culmen Length (mm)',
        'bill_length_mm': 'Culmen Length (mm)',  # 兼容不同版本的列名
        'culmen_depth_mm': 'Culmen Depth (mm)',
        'bill_depth_mm': 'Culmen Depth (mm)',
        'flipper_length_mm': 'Flipper Length (mm)',
        'body_mass_g': 'Body Mass (g)',
        'species': 'Species'
    }
    
    # 只保留需要的列并重命名
    # 注意：这里过滤掉了 'island' 和 'sex'，只保留数值特征用于画散点图
    cols_to_use = [col for col in rename_map.keys() if col in df_clean.columns]
    df_plot = df_clean[cols_to_use].rename(columns=rename_map)

    print("[绘图] 正在生成 Pairplot (这可能需要几秒钟)...")

    # 3. 使用 Seaborn 绘制 Pairplot
    # hue='Species': 按照物种上色
    # diag_kind='hist': 对角线画直方图 (你的截图中是直方图)
    # height=2.5: 控制每张小图的大小
    # 使用 palmerpenguins 官方风格的配色（带透明度，重叠时不会完全变灰）
    penguin_colors = {
        'Adelie': '#FF8C00',      # 橙色
        'Chinstrap': '#A034F0',   # 紫色  
        'Gentoo': '#057076'       # 青色
    }
    
    g = sns.pairplot(
        df_plot, 
        hue='Species', 
        diag_kind='hist',
        markers=["o", "s", "D"], # 设置不同物种点的形状
        palette=penguin_colors,  # 使用官方风格配色
        height=2.5,
        corner=False,            # 设为 True 可以只画左下角的一半
        plot_kws={'alpha': 0.5}, # 散点图透明度
        diag_kws={'alpha': 0.3}  # 直方图透明度
    )

    # 4. 添加总标题
    # y=1.02 是为了把标题稍微往上提一点，避免遮挡图表
    g.fig.suptitle("特征与标签组合的散点可视化", fontsize=16, y=1.02)

    # 5. 保存图片
    out_path = os.path.join(OUTPUT_DIR, 'feature_pairplot.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"[绘图] 完成。图片已保存至: {out_path}")
    
    # 也可以显示出来(如果在交互式环境下)
    # plt.show()

if __name__ == "__main__":
    main()