import time
import os
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
import h5py
import openslide
from tqdm import tqdm
import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder
from utils.blur_utils import blur_score_laplacian  # 기존 util 그대로 사용

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def custom_collate_fn(batch):
	"""Custom collate function to handle PIL Images."""
	from torch.utils.data._utils.collate import default_collate

	img_pils = [item['img_pil'] for item in batch]
	other_data = [{'img': item['img'], 'coord': item['coord']} for item in batch]
	collated = default_collate(other_data)
	collated['img_pil'] = img_pils
	return collated


def compute_w_loader(
	output_path,
	loader,
	model,
	verbose=0,
	blur_mode='none',
	blur_thr=None,
	blur_alpha=0.2,
	blur_downsample=2
):
	"""
	args:
		output_path: path to save computed features (.h5)
		model: pytorch encoder model
		verbose: 0/1+
		blur_mode: 'none' | 'drop' | 'weight'
			- none: no filtering
			- drop: drop patches with blur_score < blur_thr
			- weight: keep all patches, but save weights (blur < thr -> blur_alpha else 1.0)
		blur_thr: threshold used in drop/weight
		blur_alpha: weight assigned to blurry patches in weight mode
		blur_downsample: downsample factor for blur computation
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches')
		if blur_mode in ['drop', 'weight'] and blur_thr is not None:
			print(f'Blur mode = {blur_mode}, threshold = {blur_thr}, downsample = {blur_downsample}')
			if blur_mode == 'weight':
				print(f'  weight for blurry patches = {blur_alpha}')

	mode = 'w'
	total_patches = 0
	total_dropped = 0
	skipped_batches = 0

	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():
			batch = data['img']  # Tensor (B,C,H,W)
			coords = data['coord'].numpy().astype(np.int32)
			img_pils = data['img_pil']  # list[PIL.Image]

			weights = None

			# --------- blur score 계산 (필요할 때만) ----------
			if blur_mode in ['drop', 'weight'] and blur_thr is not None:
				blur_scores = np.array(
					[blur_score_laplacian(pil, downsample=blur_downsample) for pil in img_pils],
					dtype=np.float32
				)

				original_count = len(blur_scores)
				total_patches += original_count

				if blur_mode == 'drop':
					keep_mask = blur_scores >= blur_thr
					dropped_count = int((~keep_mask).sum())
					total_dropped += dropped_count

					if keep_mask.sum() == 0:
						skipped_batches += 1
						if verbose > 0 and count % 10 == 0:
							tqdm.write(f"  Batch {count}: All patches dropped (blurry)")
						continue

					if dropped_count > 0 and verbose > 0 and count % 10 == 0:
						tqdm.write(f"  Batch {count}: Dropped {dropped_count}/{original_count} blurry patches")

					batch = batch[keep_mask]
					coords = coords[keep_mask]

				elif blur_mode == 'weight':
					# weight는 drop하지 않고 "저장"만
					weights = np.ones(original_count, dtype=np.float32)
					weights[blur_scores < blur_thr] = float(blur_alpha)

			# --------- feature extraction ----------
			batch = batch.to(device, non_blocking=True)
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}

			# weight 모드에서만 weights 저장
			if blur_mode == 'weight' and weights is not None:
				asset_dict['weights'] = weights

			save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
			mode = 'a'

	# --------- stats ----------
	if verbose > 0 and blur_mode == 'drop' and blur_thr is not None:
		print(f"\n{'='*60}")
		print("Blur Filtering Statistics (DROP)")
		print(f"{'='*60}")
		print(f"Total patches processed: {total_patches}")
		print(f"Total patches dropped: {total_dropped} ({100*total_dropped/max(total_patches,1):.2f}%)")
		print(f"Total patches kept: {total_patches - total_dropped} ({100*(total_patches-total_dropped)/max(total_patches,1):.2f}%)")
		print(f"Skipped batches (all blurry): {skipped_batches}")
		print(f"{'='*60}")

	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)

parser.add_argument(
	'--model_name',
	type=str,
	default='resnet50_trunc',
	choices=['resnet50_trunc', 'uni_v1', 'uni_v2', 'uni_v2_l', 'conch_v1', 'conch_v1_5']
)

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)

# --- blur options (통합) ---
parser.add_argument(
	'--blur_mode',
	type=str,
	default='none',
	choices=['none', 'drop', 'weight'],
	help='none: no filtering, drop: remove blurry patches, weight: save per-patch weights'
)
parser.add_argument('--blur_thr', type=float, default=None, help='threshold for blur score')
parser.add_argument('--blur_alpha', type=float, default=0.2, help='weight for blurry patches when blur_mode=weight')
parser.add_argument('--blur_downsample', type=int, default=2, help='downsample factor for blur computation')

args = parser.parse_args()


if __name__ == '__main__':
	print('initializing dataset')
	if args.csv_path is None:
		raise NotImplementedError("csv_path is required")

	bags_dataset = Dataset_All_Bags(args.csv_path)

	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
	model.eval()
	model = model.to(device)

	total = len(bags_dataset)
	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id + '.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)

		print(f'\nprogress: {bag_candidate_idx}/{total}')
		print(slide_id)

		if not args.no_auto_skip and slide_id + '.pt' in dest_files:
			print(f'skipped {slide_id}')
			continue

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)

		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)

		dataset = Whole_Slide_Bag_FP(
			file_path=h5_file_path,
			wsi=wsi,
			img_transforms=img_transforms
		)

		# ✅ PIL 안전 처리 위해 collate_fn 유지
		loader = DataLoader(
			dataset=dataset,
			batch_size=args.batch_size,
			collate_fn=custom_collate_fn,
			**loader_kwargs
		)

		output_file_path = compute_w_loader(
			output_path=output_path,
			loader=loader,
			model=model,
			verbose=1,
			blur_mode=args.blur_mode,
			blur_thr=args.blur_thr,
			blur_alpha=args.blur_alpha,
			blur_downsample=args.blur_downsample
		)

		time_elapsed = time.time() - time_start
		print(f'\ncomputing features for {output_file_path} took {time_elapsed:.2f} s')

		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)
			if 'weights' in file:
				print('weights size: ', file['weights'].shape)

		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))
