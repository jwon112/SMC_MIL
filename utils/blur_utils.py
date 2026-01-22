import os
import argparse
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import openslide
import cv2
from tqdm import tqdm

def blur_score_laplacian(pil_img, downsample=2):
    """Calculate blur score using Laplacian variance.
    
    Args:
        pil_img: PIL Image
        downsample: Downsample factor for faster computation
        
    Returns:
        float: Blur score (higher = sharper)
    """
    img = np.array(pil_img.convert("RGB"))
    if downsample and downsample > 1:
        img = cv2.resize(img, (img.shape[1]//downsample, img.shape[0]//downsample))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def sample_blur_scores(
    slide_path, coords_h5_path, patch_level=0, patch_size=256,
    max_patches=2000, seed=42
):
    """Sample blur scores from a slide.
    
    Args:
        slide_path: Path to WSI file
        coords_h5_path: Path to H5 file with coordinates
        patch_level: Patch level
        patch_size: Patch size
        max_patches: Maximum number of patches to sample
        seed: Random seed
        
    Returns:
        list: List of blur scores
    """
    rng = np.random.default_rng(seed)
    slide = openslide.OpenSlide(slide_path)

    with h5py.File(coords_h5_path, "r") as f:
        coords = f["coords"][:]  # (N, 2) top-left

    n = len(coords)
    if n == 0:
        return []

    idx = rng.choice(n, size=min(max_patches, n), replace=False)
    scores = []
    for i in idx:
        x, y = coords[i]
        region = slide.read_region((int(x), int(y)), patch_level, (patch_size, patch_size)).convert("RGB")
        scores.append(blur_score_laplacian(region, downsample=2))
    slide.close()
    return scores

def profile_blur_scores(
    csv_path, data_slide_dir, slide_ext,
    data_h5_dir, patch_level=0, patch_size=256,
    max_patches_per_slide=2000, out_csv="blur_scores.csv"
):
    """Profile blur scores across multiple slides.
    
    Args:
        csv_path: Path to CSV with slide IDs
        data_slide_dir: Directory containing WSI files
        slide_ext: Slide file extension
        data_h5_dir: Directory containing H5 coordinate files
        patch_level: Patch level
        patch_size: Patch size
        max_patches_per_slide: Maximum patches to sample per slide
        out_csv: Output CSV path
    """
    df = pd.read_csv(csv_path)
    # CLAM process_list_autogen.csv는 보통 "slide_id" 혹은 "slide" 컬럼을 가짐
    slide_ids = df["slide_id"].astype(str).tolist() if "slide_id" in df.columns else df["slide"].astype(str).tolist()

    print(f"\n{'='*60}")
    print("Configuration Check")
    print(f"{'='*60}")
    print(f"CSV file: {csv_path}")
    print(f"Total slides in CSV: {len(slide_ids)}")
    print(f"Slide directory: {data_slide_dir}")
    print(f"H5 directory: {os.path.join(data_h5_dir, 'patches')}")
    print(f"Slide extension: {slide_ext}")
    print(f"{'='*60}\n")

    rows = []
    missing_slide_files = []
    missing_h5_files = []
    empty_coords = []
    processed_slides = []
    
    for sid in tqdm(slide_ids, desc="Processing slides"):
        slide_path = os.path.join(data_slide_dir, sid + slide_ext)
        h5_path = os.path.join(data_h5_dir, "patches", sid + ".h5")
        
        # 1단계: 슬라이드 파일 확인
        if not os.path.exists(slide_path):
            missing_slide_files.append((sid, slide_path))
            continue
        
        # 2단계: H5 파일 확인
        if not os.path.exists(h5_path):
            missing_h5_files.append((sid, h5_path))
            continue
        
        # 3단계: H5 파일 내부 좌표 확인
        try:
            with h5py.File(h5_path, "r") as f:
                if "coords" not in f:
                    empty_coords.append((sid, "no 'coords' dataset"))
                    continue
                coords = f["coords"][:]
                if len(coords) == 0:
                    empty_coords.append((sid, "empty coords"))
                    continue
        except Exception as e:
            empty_coords.append((sid, f"error reading H5: {e}"))
            continue
        
        # 4단계: blur score 계산
        try:
            scores = sample_blur_scores(
                slide_path, h5_path, patch_level=patch_level, patch_size=patch_size,
                max_patches=max_patches_per_slide
            )
            if len(scores) == 0:
                empty_coords.append((sid, "sample_blur_scores returned empty"))
                continue
            
            processed_slides.append(sid)
            for s in scores:
                rows.append({"slide_id": sid, "blur_score": s})
        except Exception as e:
            print(f"\nERROR processing {sid}: {e}")
            continue

    # 상세한 진단 결과 출력
    print(f"\n{'='*60}")
    print("Processing Summary")
    print(f"{'='*60}")
    print(f"✓ Successfully processed: {len(processed_slides)} slides")
    print(f"✗ Missing slide files: {len(missing_slide_files)}")
    print(f"✗ Missing H5 files: {len(missing_h5_files)}")
    print(f"✗ Empty/invalid H5 files: {len(empty_coords)}")
    print(f"✓ Total blur scores calculated: {len(rows)}")
    
    if len(rows) == 0:
        print(f"\n{'='*60}")
        print("ERROR: No blur scores were calculated!")
        print(f"{'='*60}")
        
        if missing_slide_files:
            print(f"\n❌ Missing slide files (first 5 examples):")
            for sid, path in missing_slide_files[:5]:
                print(f"   - {sid}")
                print(f"     Expected: {path}")
            if len(missing_slide_files) > 5:
                print(f"   ... and {len(missing_slide_files) - 5} more")
        
        if missing_h5_files:
            print(f"\n❌ Missing H5 files (first 5 examples):")
            for sid, path in missing_h5_files[:5]:
                print(f"   - {sid}")
                print(f"     Expected: {path}")
            if len(missing_h5_files) > 5:
                print(f"   ... and {len(missing_h5_files) - 5} more")
        
        if empty_coords:
            print(f"\n❌ Empty/invalid H5 files (first 5 examples):")
            for sid, reason in empty_coords[:5]:
                print(f"   - {sid}: {reason}")
            if len(empty_coords) > 5:
                print(f"   ... and {len(empty_coords) - 5} more")
        
        print(f"\n{'='*60}")
        print("Troubleshooting Steps:")
        print(f"{'='*60}")
        print(f"1. Check if slide files exist:")
        print(f"   ls {data_slide_dir}/*{slide_ext} | head")
        print(f"2. Check if H5 files exist:")
        print(f"   ls {os.path.join(data_h5_dir, 'patches')}/*.h5 | head")
        print(f"3. Verify slide IDs in CSV match file names (without extension)")
        print(f"4. Check if H5 files contain 'coords' dataset with data")
        print(f"{'='*60}")
        return

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"\n✓ Saved: {out_csv}")
    print(f"\n{'='*60}")
    print("Blur Score Statistics")
    print(f"{'='*60}")
    print(out["blur_score"].describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99]))
    print("\nRecommended blur threshold values:")
    print(f"  5th percentile: {out['blur_score'].quantile(0.05):.2f}")
    print(f"  10th percentile: {out['blur_score'].quantile(0.10):.2f}")
    print(f"  25th percentile: {out['blur_score'].quantile(0.25):.2f}")
    print("\nUse these values with --blur_thr in extract_features_fp.py")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile blur scores across multiple slides')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file with slide IDs')
    parser.add_argument('--data_slide_dir', type=str, required=True,
                        help='Directory containing WSI files')
    parser.add_argument('--data_h5_dir', type=str, required=True,
                        help='Directory containing H5 coordinate files')
    parser.add_argument('--slide_ext', type=str, default='.svs',
                        help='Slide file extension (default: .svs)')
    parser.add_argument('--patch_level', type=int, default=0,
                        help='Patch level (default: 0)')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Patch size (default: 256)')
    parser.add_argument('--max_patches_per_slide', type=int, default=2000,
                        help='Maximum patches to sample per slide (default: 2000)')
    parser.add_argument('--out_csv', type=str, default='blur_scores.csv',
                        help='Output CSV path (default: blur_scores.csv)')
    
    args = parser.parse_args()
    
    profile_blur_scores(
        csv_path=args.csv_path,
        data_slide_dir=args.data_slide_dir,
        slide_ext=args.slide_ext,
        data_h5_dir=args.data_h5_dir,
        patch_level=args.patch_level,
        patch_size=args.patch_size,
        max_patches_per_slide=args.max_patches_per_slide,
        out_csv=args.out_csv
    )
