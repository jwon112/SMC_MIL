import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder
from utils.blur_utils import blur_score_laplacian

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def custom_collate_fn(batch):
	"""Custom collate function to handle PIL Images."""
	from torch.utils.data._utils.collate import default_collate
	
	# Separate PIL images from other data
	img_pils = [item['img_pil'] for item in batch]
	
	# Use default collate for the rest
	other_data = [{'img': item['img'], 'coord': item['coord']} for item in batch]
	collated = default_collate(other_data)
	
	# Add PIL images as a list (not collated)
	collated['img_pil'] = img_pils
	
	return collated

def compute_w_loader(output_path, loader, model, verbose = 0, blur_mode='none', blur_thr=None, blur_downsample=2):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
		blur_mode: 'none' or 'drop' - blur filtering mode
		blur_thr: blur threshold (patches with score < threshold will be dropped)
		blur_downsample: downsample factor for blur computation
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))
		if blur_mode == 'drop' and blur_thr is not None:
			print(f'Blur filtering enabled: threshold = {blur_thr}')

	mode = 'w'
	total_patches = 0
	total_dropped = 0
	skipped_batches = 0
	
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			img_pils = data['img_pil']  # PIL images for blur computation

			# Blur filtering (drop mode)
			if blur_mode == 'drop' and blur_thr is not None:
				blur_scores = []
				for pil in img_pils:
					score = blur_score_laplacian(pil, downsample=blur_downsample)
					blur_scores.append(score)

				blur_scores = np.array(blur_scores, dtype=np.float32)
				keep_mask = blur_scores >= blur_thr
				
				original_count = len(blur_scores)
				dropped_count = (~keep_mask).sum()
				total_patches += original_count
				total_dropped += dropped_count
				
				# Skip batch if all patches are blurry
				if keep_mask.sum() == 0:
					skipped_batches += 1
					if verbose > 0 and count % 10 == 0:
						tqdm.write(f"  Batch {count}: All patches dropped (blurry)")
					continue

				# Filter out blurry patches
				if dropped_count > 0 and verbose > 0 and count % 10 == 0:
					tqdm.write(f"  Batch {count}: Dropped {dropped_count}/{original_count} blurry patches")
				
				batch = batch[keep_mask]
				coords = coords[keep_mask]

			batch = batch.to(device, non_blocking=True)
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	# Print blur filtering statistics
	if blur_mode == 'drop' and blur_thr is not None and verbose > 0:
		print(f"\n{'='*60}")
		print("Blur Filtering Statistics")
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
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'uni_v2', 'uni_v2_l', 'conch_v1', 'conch_v1_5'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--blur_mode', type=str, default='none', choices=['none', 'drop'],
					help='Blur filtering mode: none (no filtering) or drop (remove blurry patches)')
parser.add_argument('--blur_thr', type=float, default=None,
					help='Blur threshold. Patches with blur score < threshold will be dropped (only used when --blur_mode=drop)')
parser.add_argument('--blur_downsample', type=int, default=2,
					help='Downsample factor for blur computation (default: 2)')
args = parser.parse_args()


if __name__ == '__main__':
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
			
	_ = model.eval()
	model = model.to(device)
	total = len(bags_dataset)

	loader_kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == "cuda" else {}

	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
							   		 wsi=wsi, 
									 img_transforms=img_transforms)

		# Use custom collate function to handle PIL Images (dataset always returns img_pil)
		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn, **loader_kwargs)
		output_file_path = compute_w_loader(
			output_path, 
			loader=loader, 
			model=model, 
			verbose=1,
			blur_mode=args.blur_mode,
			blur_thr=args.blur_thr,
			blur_downsample=args.blur_downsample
		)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)

		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))



