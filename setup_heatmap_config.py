#!/usr/bin/env python
"""
자동으로 heatmap 설정 파일과 CSV를 생성하는 스크립트
"""
import os
import yaml
import pandas as pd
import glob
import argparse

def find_wsi_files(data_dir, slide_ext='.svs'):
    """WSI 파일 찾기"""
    extensions = ['.svs', '.ndpi', '.tiff', '.tif']
    if slide_ext:
        extensions = [slide_ext]
    
    wsi_files = []
    for ext in extensions:
        pattern = os.path.join(data_dir, f'*{ext}')
        files = glob.glob(pattern)
        wsi_files.extend(files)
    
    return sorted(wsi_files)

def find_checkpoint(results_dir, exp_code='task3_s1', fold=0):
    """체크포인트 파일 찾기"""
    ckpt_path = os.path.join(results_dir, exp_code, f's_{fold}_checkpoint.pt')
    if os.path.exists(ckpt_path):
        return ckpt_path
    
    # 다른 fold 찾기
    pattern = os.path.join(results_dir, exp_code, 's_*_checkpoint.pt')
    checkpoints = glob.glob(pattern)
    if checkpoints:
        return checkpoints[0]
    
    return None

def create_heatmap_list_csv(wsi_files, output_path, label_dict=None):
    """Heatmap용 CSV 파일 생성"""
    slide_ids = []
    labels = []
    
    for wsi_file in wsi_files:
        slide_id = os.path.splitext(os.path.basename(wsi_file))[0]
        slide_ids.append(slide_id)
        
        # label이 없으면 첫 번째 클래스로 설정
        if label_dict:
            labels.append(list(label_dict.keys())[0])
        else:
            labels.append('normal')
    
    df = pd.DataFrame({'slide_id': slide_ids, 'label': labels})
    df.to_csv(output_path, index=False)
    print(f"Created CSV file: {output_path}")
    print(f"Total slides: {len(slide_ids)}")
    return output_path

def create_config_file(config_path, data_dir, ckpt_path, process_list_path, 
                       n_classes=2, label_dict=None, model_name='resnet50_trunc',
                       embed_dim=1024, slide_ext='.svs'):
    """설정 파일 생성"""
    
    if label_dict is None:
        label_dict = {'normal': 0, 'tumor': 1}
    
    config = {
        'exp_arguments': {
            'n_classes': n_classes,
            'save_exp_code': 'task3_heatmap',
            'raw_save_dir': 'heatmaps/heatmap_raw_results',
            'production_save_dir': 'heatmaps/heatmap_production_results',
            'batch_size': 256
        },
        'data_arguments': {
            'data_dir': data_dir,
            'data_dir_key': 'source',
            'process_list': os.path.basename(process_list_path),
            'preset': 'presets/bwh_biopsy.csv',
            'slide_ext': slide_ext,
            'label_dict': label_dict
        },
        'patching_arguments': {
            'patch_size': 256,
            'overlap': 0.5,
            'patch_level': 0,
            'custom_downsample': 1
        },
        'encoder_arguments': {
            'model_name': model_name,
            'target_img_size': 224
        },
        'model_arguments': {
            'ckpt_path': ckpt_path,
            'model_type': 'clam_sb',
            'initiate_fn': 'initiate_model',
            'model_size': 'small',
            'drop_out': 0.25,
            'embed_dim': embed_dim
        },
        'heatmap_arguments': {
            'vis_level': 1,
            'alpha': 0.4,
            'blank_canvas': False,
            'save_orig': True,
            'save_ext': 'jpg',
            'use_ref_scores': True,
            'blur': False,
            'use_center_shift': True,
            'use_roi': False,
            'calc_heatmap': True,
            'binarize': False,
            'binary_thresh': -1,
            'custom_downsample': 1,
            'cmap': 'jet'
        },
        'sample_arguments': {
            'samples': [
                {
                    'name': 'topk_high_attention',
                    'sample': True,
                    'seed': 1,
                    'k': 15,
                    'mode': 'topk'
                }
            ]
        }
    }
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created config file: {config_path}")
    return config_path

def main():
    parser = argparse.ArgumentParser(description='Heatmap 설정 파일 자동 생성')
    parser.add_argument('--data_dir', type=str, default='image_team/projects/CLAM/data',
                        help='WSI 파일이 있는 디렉토리')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='결과 디렉토리')
    parser.add_argument('--exp_code', type=str, default='task3_s1',
                        help='실험 코드 (체크포인트 찾기용)')
    parser.add_argument('--fold', type=int, default=0,
                        help='사용할 fold 번호')
    parser.add_argument('--slide_ext', type=str, default='.svs',
                        help='WSI 파일 확장자 (.svs, .ndpi, .tiff 등)')
    parser.add_argument('--model_name', type=str, default='resnet50_trunc',
                        help='Encoder 모델 이름')
    parser.add_argument('--embed_dim', type=int, default=1024,
                        help='Feature embedding 차원')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='클래스 수')
    parser.add_argument('--max_slides', type=int, default=5,
                        help='최대 시각화할 slide 수 (테스트용)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Heatmap 설정 파일 자동 생성")
    print("=" * 60)
    
    # 1. WSI 파일 찾기
    print(f"\n1. WSI 파일 찾는 중: {args.data_dir}")
    wsi_files = find_wsi_files(args.data_dir, args.slide_ext)
    
    if not wsi_files:
        print(f"경고: {args.data_dir}에서 WSI 파일을 찾을 수 없습니다.")
        print("다른 확장자를 시도합니다...")
        wsi_files = find_wsi_files(args.data_dir, None)
    
    if not wsi_files:
        print(f"에러: WSI 파일을 찾을 수 없습니다. 경로를 확인해주세요: {args.data_dir}")
        return
    
    # 테스트용으로 제한
    if args.max_slides > 0:
        wsi_files = wsi_files[:args.max_slides]
    
    print(f"   찾은 WSI 파일: {len(wsi_files)}개")
    for i, f in enumerate(wsi_files[:5], 1):
        print(f"   {i}. {os.path.basename(f)}")
    if len(wsi_files) > 5:
        print(f"   ... 외 {len(wsi_files)-5}개")
    
    # 2. 체크포인트 찾기
    print(f"\n2. 체크포인트 찾는 중: {args.results_dir}/{args.exp_code}")
    ckpt_path = find_checkpoint(args.results_dir, args.exp_code, args.fold)
    
    if not ckpt_path:
        print(f"경고: 체크포인트를 찾을 수 없습니다.")
        print(f"   경로: {args.results_dir}/{args.exp_code}/s_{args.fold}_checkpoint.pt")
        ckpt_path = input("체크포인트 경로를 직접 입력하세요 (또는 Enter로 건너뛰기): ").strip()
        if not ckpt_path:
            print("체크포인트를 찾을 수 없어 설정 파일 생성을 중단합니다.")
            return
    else:
        print(f"   찾은 체크포인트: {ckpt_path}")
    
    # 3. CSV 파일 생성
    print(f"\n3. CSV 파일 생성 중...")
    os.makedirs('heatmaps/process_lists', exist_ok=True)
    csv_path = 'heatmaps/process_lists/task3_heatmap_list.csv'
    
    label_dict = {'normal': 0, 'tumor': 1} if args.n_classes == 2 else None
    create_heatmap_list_csv(wsi_files, csv_path, label_dict)
    
    # 4. 설정 파일 생성
    print(f"\n4. 설정 파일 생성 중...")
    config_path = 'heatmaps/configs/task3_heatmap_config.yaml'
    create_config_file(
        config_path=config_path,
        data_dir=args.data_dir,
        ckpt_path=ckpt_path,
        process_list_path=csv_path,
        n_classes=args.n_classes,
        label_dict=label_dict,
        model_name=args.model_name,
        embed_dim=args.embed_dim,
        slide_ext=args.slide_ext
    )
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"\n생성된 파일:")
    print(f"  - 설정 파일: {config_path}")
    print(f"  - CSV 파일: {csv_path}")
    print(f"\n실행 명령어:")
    print(f"  CUDA_VISIBLE_DEVICES=1 python create_heatmaps.py --config task3_heatmap_config.yaml")
    print()

if __name__ == '__main__':
    main()
