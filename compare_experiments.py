import os
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

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


def positive_probability(value):
    probs = np.asarray(value, dtype=float).reshape(-1)
    if probs.size != 2:
        raise ValueError(f'Expected binary probabilities, got shape {np.asarray(value).shape}')
    return float(probs[1])


def collect_prediction_rows(results_dir, experiment_names):
    """Read held-out bag probabilities saved by main.py for binary experiments."""
    rows = []
    for experiment in experiment_names:
        exp_dir = Path(results_dir) / experiment
        for result_path in sorted(exp_dir.glob('split_*_results.pkl')):
            fold = int(result_path.stem.split('_')[1])
            with result_path.open('rb') as handle:
                split_results = pickle.load(handle)
            for slide_id, result in split_results.items():
                rows.append({
                    'experiment': experiment,
                    'fold': fold,
                    'slide_id': str(slide_id),
                    'label': int(result['label']),
                    'prob_positive': positive_probability(result['prob']),
                })
    return pd.DataFrame(rows)


def calculate_binary_metrics(frame, threshold, scope, fold):
    labels = frame['label'].to_numpy(dtype=int)
    probs = frame['prob_positive'].to_numpy(dtype=float)
    predictions = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        'experiment': frame['experiment'].iloc[0],
        'scope': scope,
        'fold': fold,
        'n_bags': len(frame),
        'positive_bags': int(labels.sum()),
        'positive_prevalence': float(labels.mean()),
        'threshold': threshold,
        'auroc': roc_auc_score(labels, probs),
        'average_precision': average_precision_score(labels, probs),
        'balanced_accuracy': balanced_accuracy_score(labels, predictions),
        'sensitivity': recall_score(labels, predictions, zero_division=0),
        'specificity': tn / (tn + fp) if tn + fp else float('nan'),
        'precision': precision_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
        'mcc': matthews_corrcoef(labels, predictions) if len(np.unique(predictions)) > 1 else 0.0,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def calculate_imbalance_metrics(results_dir, experiment_names, threshold):
    """Return fold and average metrics to merge into the existing summaries."""
    predictions = collect_prediction_rows(results_dir, experiment_names)
    if predictions.empty:
        print('No split_*_results.pkl files found; skipping imbalance-aware metrics.')
        return None, None

    predictions = predictions.sort_values(['experiment', 'fold', 'slide_id'])
    metric_rows = []
    for (experiment, fold), frame in predictions.groupby(['experiment', 'fold'], sort=True):
        metric_rows.append(calculate_binary_metrics(frame, threshold, 'outer_fold', int(fold)))
    metrics = pd.DataFrame(metric_rows)
    fold_metrics = metrics.loc[metrics['scope'] == 'outer_fold'].copy()
    metric_cols = [
        'auroc', 'average_precision', 'balanced_accuracy', 'sensitivity',
        'specificity', 'precision', 'f1', 'mcc',
    ]
    averages = fold_metrics.groupby('experiment')[metric_cols].agg(['mean', 'std'])
    averages.columns = ['_'.join(column) for column in averages.columns]
    averages = averages.reset_index()
    fold_metrics = fold_metrics.drop(columns=['scope'])
    print('Added PR-AUC, balanced accuracy, sensitivity, specificity, F1, and MCC to existing summaries.')
    return fold_metrics, averages

def create_barplot(df, metric, output_path, title_suffix=''):
    """막대 그래프 생성 (수치값 표시)"""
    plt.figure(figsize=(10, 6))
    
    experiments = df['experiment'].values
    values = df[metric].values
    stds = df[f'{metric}_std'].values if f'{metric}_std' in df.columns else None
    
    # 실험마다 다른 색상 지정 (더 명확한 색상 팔레트)
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                     '#aec7e8', '#ffbb78']
    colors = [color_palette[i % len(color_palette)] for i in range(len(experiments))]
    
    # 막대 그래프 (각 막대를 개별적으로 그려서 색상 확실히 적용)
    bars = []
    for i, (val, color) in enumerate(zip(values, colors)):
        bar = plt.bar(i, val, color=color, alpha=0.8, 
                     edgecolor='black', linewidth=1.5)
        bars.extend(bar)
    
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
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Fixed positive-class threshold for sensitivity/specificity metrics (default: 0.5)')
    
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        parser.error('--threshold must be strictly between 0 and 1')
    
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
    
    # 평균 계산
    print("\nCalculating averages...")
    avg_df = calculate_averages(combined_df)

    fold_metrics, metric_averages = calculate_imbalance_metrics(
        args.results_dir, args.experiments, args.threshold)
    if fold_metrics is not None:
        combined_df = combined_df.merge(
            fold_metrics,
            left_on=['experiment', 'folds'],
            right_on=['experiment', 'fold'],
            how='left',
        ).drop(columns=['fold'])
        avg_df = avg_df.merge(metric_averages, on='experiment', how='left')

    # 통합 CSV 저장
    combined_csv_path = os.path.join(output_dir, 'combined_summary.csv')
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Saved combined summary: {combined_csv_path}")

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
