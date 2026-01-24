import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def collect_summaries(results_dir, experiment_names):
    """지정된 실험들의 summary.csv를 수집하고 통합"""
    all_dfs = []
    
    for exp_name in experiment_names:
        exp_path = os.path.join(results_dir, exp_name)
        summary_path = os.path.join(exp_path, 'summary.csv')
        
        if not os.path.exists(summary_path):
            print(f"Warning: {summary_path} not found. Skipping {exp_name}")
            continue
        
        df = pd.read_csv(summary_path)
        df['experiment'] = exp_name
        all_dfs.append(df)
    
    if len(all_dfs) == 0:
        raise ValueError("No valid summary.csv files found!")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def calculate_averages(df):
    """각 실험별로 폴드 평균 계산"""
    avg_df = df.groupby('experiment').agg({
        'test_auc': 'mean',
        'val_auc': 'mean',
        'test_acc': 'mean',
        'val_acc': 'mean'
    }).reset_index()
    
    # 표준편차도 계산 (에러바용)
    std_df = df.groupby('experiment').agg({
        'test_auc': 'std',
        'val_auc': 'std',
        'test_acc': 'std',
        'val_acc': 'std'
    }).reset_index()
    
    # 평균과 표준편차 합치기
    for col in ['test_auc', 'val_auc', 'test_acc', 'val_acc']:
        avg_df[f'{col}_std'] = std_df[col]
    
    return avg_df

def create_barplot(df, metric, output_path, title_suffix=''):
    """막대 그래프 생성 (수치값 표시)"""
    plt.figure(figsize=(10, 6))
    
    experiments = df['experiment'].values
    values = df[metric].values
    stds = df[f'{metric}_std'].values if f'{metric}_std' in df.columns else None
    
    # 막대 그래프 (error bar 제거 - 숫자와 겹침 방지)
    bars = plt.bar(range(len(experiments)), values, 
                        alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # 막대 위에 수치값 표시 (error bar 위로 충분히 올림)
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        # error bar가 있다면 그 위로, 없다면 막대 위에 여유있게
        y_offset = stds[i] * 1.2 if stds is not None and not np.isnan(stds[i]) else height * 0.05
        text_y = height + y_offset
        
        plt.text(bar.get_x() + bar.get_width()/2., text_y,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Experiment', fontsize=12, fontweight='bold')
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    plt.title(f'{metric.replace("_", " ").title()} Comparison{title_suffix}', 
              fontsize=14, fontweight='bold')
    # 실험명 가로로 배치
    plt.xticks(range(len(experiments)), experiments, rotation=0, ha='center')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare multiple experiment results')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Base results directory (default: ./results)')
    parser.add_argument('--experiments', type=str, nargs='+', required=True,
                        help='Experiment names (subdirectories in results_dir)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for comparison results (default: results_dir/comparison)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        output_dir = os.path.join(args.results_dir, 'comparison')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Summary 수집
    print(f"\nCollecting summaries from {len(args.experiments)} experiments...")
    combined_df = collect_summaries(args.results_dir, args.experiments)
    
    # 통합 CSV 저장
    combined_csv_path = os.path.join(output_dir, 'combined_summary.csv')
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Saved combined summary: {combined_csv_path}")
    
    # 평균 계산
    print("\nCalculating averages...")
    avg_df = calculate_averages(combined_df)
    
    # 평균 결과 CSV 저장
    avg_csv_path = os.path.join(output_dir, 'averaged_summary.csv')
    avg_df.to_csv(avg_csv_path, index=False)
    print(f"Saved averaged summary: {avg_csv_path}")
    
    # 시각화
    print("\nGenerating visualizations...")
    create_barplot(avg_df, 'test_auc', 
                  os.path.join(output_dir, 'test_auc_comparison.png'),
                  ' - Test AUC')
    create_barplot(avg_df, 'test_acc', 
                  os.path.join(output_dir, 'test_acc_comparison.png'),
                  ' - Test Accuracy')
    
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")
    print(avg_df[['experiment', 'test_auc', 'test_acc']].to_string(index=False))
    print(f"{'='*60}")
    print(f"\nAll results saved to: {output_dir}")

if __name__ == '__main__':
    main()
