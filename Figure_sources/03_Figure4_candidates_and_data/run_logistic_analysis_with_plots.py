# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 13:51:53 2025

@author: xiang

Note:
This version preserves ALL fitting logic, parameters, and outputs of the original script,
and only adds per-word plot saving without changing the returned results or the CSV schema.
"""

import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def logistic_model(t, L, k, t0):
    """
    逻辑增长模型函数
    L: 曲线的最大值 (承载量)
    k: 增长速率
    t0: S型曲线中点的时间
    """
    try:
        return L / (1 + np.exp(-k * (t - t0)))
    except OverflowError:
        # 如果计算溢出，返回一个巨大的数值，让拟合算法知道这个方向是错误的
        return np.full_like(t, 1e12)


def analyze_word_logistic(file_path):
    """
    对单个词汇的CSV数据进行逻辑增长模型拟合与分析
    （与原版完全相同的拟合流程和参数设置；仅在成功拟合后额外保存一张叠加图）
    """
    word_name = os.path.basename(file_path).replace('.csv', '')
    
    try:
        df = pd.read_csv(file_path)
        df['Time'] = pd.to_datetime(df['Time'])
        
        # 将数据转换为累计词频，因为逻辑增长模型是描述累计数量的
        df.sort_values('Time', inplace=True)
        df['cumulative_count'] = df['count'].cumsum()
        
        # 按月取样，获得更平滑的累计曲线（与原脚本一致：'M' 月末采样 + ffill）
        df.set_index('Time', inplace=True)
        monthly_cumulative = df['cumulative_count'].resample('M').max().ffill()
        
        if monthly_cumulative.empty or monthly_cumulative.isnull().all():
            print(f"[{word_name}] 数据为空或全是无效值，已跳过。")
            return None
        
        monthly_cumulative.dropna(inplace=True)
        
        y_data = monthly_cumulative.values
        x_data = np.arange(len(y_data)) # 使用简单的数字序列作为时间输入
        
        if len(y_data) < 4:
            print(f"[{word_name}] 有效数据点不足4个，无法进行可靠拟合，已跳过。")
            return None
            
    except Exception as e:
        print(f"[{word_name}] 处理数据时发生错误: {e}")
        return None

    # --- 执行拟合（保持原参数设置与边界不变） ---
    try:
        # 为参数提供合理的初始猜测值和边界
        max_val = y_data[-1]
        # [L, k, t0]
        p0 = [max_val, 0.1, len(x_data) / 2] 
        bounds = ([max_val*0.8, 0.001, 0], [max_val*5, 2, len(x_data)*2])
        
        popt, pcov = curve_fit(logistic_model, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError:
        print(f"[{word_name}] 拟合失败，数据可能不符合逻辑增长模式。")
        return None

    # --- 计算R²并准备结果（保持原返回不变，不新增列） ---
    y_predicted = logistic_model(x_data, *popt)
    r2 = r2_score(y_data, y_predicted)
    
    L_fit, k_fit, t0_fit = popt
    
    results = {
        'word': word_name,
        'logistic_r_squared': r2,
        'L_fit (capacity)': L_fit,
        'k_fit (growth_rate)': k_fit,
        't0_fit (midpoint_time)': t0_fit
    }

    # --- 仅新增：保存叠加图（不影响任何计算结果/CSV列） ---
    try:
        plots_dir = os.path.join(os.getcwd(), "logistic_plots")
        os.makedirs(plots_dir, exist_ok=True)

        x_dates = monthly_cumulative.index.to_pydatetime()
        plt.figure(figsize=(7.6, 4.6), dpi=200)
        # 观测：累计
        plt.plot(x_dates, y_data, lw=1.5, label="Observed (cumulative)")
        # 拟合：logistic
        plt.plot(x_dates, y_predicted, lw=2.2, label="Logistic fit")
        # 标题含R²
        try:
            r2_text = f" (R$^2$={r2:.2f})"
        except Exception:
            r2_text = ""
        plt.title(f"{word_name} — Logistic overlay{r2_text}")
        plt.xlabel("Time")
        plt.ylabel("Cumulative total")
        plt.grid(True, ls="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(plots_dir, f"{word_name}_logistic_overlay.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
    except Exception as e:
        # 绘图失败不影响拟合主流程
        print(f"[{word_name}] 绘图保存失败：{e}")

    return results

# --- 主程序开始（保持原逻辑与输出完全一致） ---
if __name__ == "__main__":
    folder_path = '.' 
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f != 'logistic_fit_summary_results.csv' and not f.startswith('sir_')]
    batch_results = []
    
    print(f"--- 开始批量处理逻辑增长模型, 共找到 {len(all_files)} 个原始数据文件 ---")

    for file in all_files:
        file_path = os.path.join(folder_path, file)
        result = analyze_word_logistic(file_path)
        if result:
            batch_results.append(result)

    if batch_results:
        summary_df = pd.DataFrame(batch_results)
        summary_df.sort_values('logistic_r_squared', ascending=False, inplace=True)
        
        print("\n--- 所有文件处理完毕，生成汇总报告 ---")
        print(summary_df.head())
        
        summary_filename = 'logistic_fit_summary_results.csv'
        summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
        
        print(f"\n批量分析完成！汇总结果已保存到 '{summary_filename}' 文件中。")
        print("拟合叠加图已保存到子文件夹 'logistic_plots'。")
    else:
        print("\n--- 未成功处理任何文件 ---")
