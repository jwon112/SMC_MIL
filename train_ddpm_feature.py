"""
Train feature-space DDPM (epsilon predictor) with L_ddpm + L_id (identity-on-clean).
Loads patch features from H5 files, samples at patch level, saves checkpoint for CLAM.
"""
from __future__ import print_function

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

from models.ddpm_feature import DDPMFeatureDenoiser, get_ddpm_schedule, noisify


def get_slide_ids_from_csv(csv_path, data_dir):
    """Return list of slide_id that have an existing h5 file."""
    df = pd.read_csv(csv_path)
    if 'slide_id' not in df.columns:
        raise ValueError("csv must have 'slide_id' column")
    slide_ids = df['slide_id'].astype(str).tolist()
    kept = []
    for sid in slide_ids:
        path = os.path.join(data_dir, 'h5_files', '{}.h5'.format(sid))
        if os.path.isfile(path):
            kept.append(sid)
    return kept


def get_slide_ids_from_dir(h5_dir):
    """Scan h5_files folder for .h5 files."""
    sub = os.path.join(h5_dir, 'h5_files')
    if not os.path.isdir(sub):
        sub = h5_dir
    ids = []
    for f in os.listdir(sub):
        if f.endswith('.h5'):
            ids.append(f[:-3])
    return ids


def get_slide_ids_in_both(clean_dir, noisy_dir):
    """Return slide_ids that have both clean and noisy h5 with same slide set."""
    clean_ids = set(get_slide_ids_from_dir(clean_dir))
    noisy_sub = os.path.join(noisy_dir, 'h5_files') if os.path.isdir(os.path.join(noisy_dir, 'h5_files')) else noisy_dir
    noisy_ids = set(os.path.splitext(f)[0] for f in os.listdir(noisy_sub) if f.endswith('.h5'))
    return sorted(clean_ids & noisy_ids)


class PatchFeatureDataset(Dataset):
    """Dataset of patch-level features from H5. Loads all into memory at init."""

    def __init__(self, data_dir, slide_ids, embed_dim=None):
        self.data_dir = data_dir
        self.slide_ids = slide_ids
        self.patches = []  # list of [N_i, D] tensors
        self.cumlen = [0]
        for sid in slide_ids:
            path = os.path.join(data_dir, 'h5_files', '{}.h5'.format(sid))
            with h5py.File(path, 'r') as f:
                feats = f['features'][:]
            feats = torch.from_numpy(feats.astype(np.float32))
            if embed_dim is not None and feats.shape[1] != embed_dim:
                raise ValueError("slide {} features dim {} != embed_dim {}".format(
                    sid, feats.shape[1], embed_dim))
            self.patches.append(feats)
            self.cumlen.append(self.cumlen[-1] + len(feats))
        self.total = self.cumlen[-1]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # which slide: cumlen[j] <= idx < cumlen[j+1]
        j = np.searchsorted(self.cumlen, idx, side='right') - 1
        j = min(j, len(self.patches) - 1)
        local = idx - self.cumlen[j]
        return self.patches[j][local].clone()


class PairedPatchFeatureDataset(Dataset):
    """Paired (clean, noisy) patch features. Same slide_id and patch index = same patch."""

    def __init__(self, clean_dir, noisy_dir, slide_ids, embed_dim=None):
        self.clean_patches = []
        self.noisy_patches = []
        self.cumlen = [0]
        for sid in slide_ids:
            cpath = os.path.join(clean_dir, 'h5_files', '{}.h5'.format(sid))
            npath = os.path.join(noisy_dir, 'h5_files', '{}.h5'.format(sid))
            if not os.path.isfile(cpath) or not os.path.isfile(npath):
                continue
            with h5py.File(cpath, 'r') as f:
                cfeat = f['features'][:]
            with h5py.File(npath, 'r') as f:
                nfeat = f['features'][:]
            if len(cfeat) != len(nfeat):
                raise ValueError("slide {} clean patches {} != noisy patches {}".format(
                    sid, len(cfeat), len(nfeat)))
            cfeat = torch.from_numpy(cfeat.astype(np.float32))
            nfeat = torch.from_numpy(nfeat.astype(np.float32))
            if embed_dim is not None:
                if cfeat.shape[1] != embed_dim or nfeat.shape[1] != embed_dim:
                    raise ValueError("slide {} feature dim != embed_dim {}".format(sid, embed_dim))
            self.clean_patches.append(cfeat)
            self.noisy_patches.append(nfeat)
            self.cumlen.append(self.cumlen[-1] + len(cfeat))
        self.total = self.cumlen[-1]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        j = np.searchsorted(self.cumlen, idx, side='right') - 1
        j = min(j, len(self.clean_patches) - 1)
        local = idx - self.cumlen[j]
        return self.clean_patches[j][local].clone(), self.noisy_patches[j][local].clone()


def main():
    parser = argparse.ArgumentParser(description='Train DDPM feature denoiser (L_ddpm + L_id)')
    parser.add_argument('--data_root_dir', type=str, default=None,
                        help='Root directory containing h5_files/')
    parser.add_argument('--h5_dir', type=str, default=None,
                        help='Same as data_root_dir; overrides if both set')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='CSV with slide_id column to restrict slides (optional)')
    parser.add_argument('--split_dir', type=str, default=None,
                        help='If set, use train split from splits in this dir (e.g. split_0)')
    parser.add_argument('--split_fold', type=int, default=0,
                        help='Fold index when using split_dir (default 0)')
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_id', type=float, default=0.2,
                        help='Weight for identity-on-clean loss L_id')
    parser.add_argument('--frac_id', type=float, default=0.2,
                        help='Fraction of batch used for t=0 (L_id) each step')
    parser.add_argument('--noisy_dir', type=str, default=None,
                        help='Path to noisy features (h5_files/). If set, train with paired (clean, noisy) + L_pair; same slide_id/patch index = pair.')
    parser.add_argument('--lambda_pair', type=float, default=1.0,
                        help='Weight for paired loss L_pair = MSE(denoise(noisy), clean) when --noisy_dir is set')
    parser.add_argument('--ddpm_t_start', type=int, default=20,
                        help='t_start for denoise() in L_pair (default 20)')
    parser.add_argument('--ddpm_num_steps', type=int, default=20,
                        help='num_steps for denoise() in L_pair (default 20)')
    parser.add_argument('--ckpt_save', type=str, default='./ddpm_ckpt.pt',
                        help='Path to save checkpoint')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    data_dir = args.h5_dir if args.h5_dir else args.data_root_dir
    if not data_dir or not os.path.isdir(data_dir):
        raise SystemExit("Provide --data_root_dir or --h5_dir with existing directory")

    use_paired = args.noisy_dir and os.path.isdir(args.noisy_dir)
    if use_paired:
        # Restrict to slides that exist in both clean and noisy
        both_ids = get_slide_ids_in_both(data_dir, args.noisy_dir)
        if not both_ids:
            raise SystemExit("No slides found in both clean dir and noisy_dir")
        if args.split_dir and os.path.isfile(os.path.join(args.split_dir, 'splits_{}.csv'.format(args.split_fold))):
            split_df = pd.read_csv(os.path.join(args.split_dir, 'splits_{}.csv'.format(args.split_fold)))
            if 'train' in split_df.columns:
                train_ids = set(split_df['train'].dropna().astype(str).unique().tolist())
            else:
                train_ids = set(split_df.iloc[:, 0].dropna().astype(str).unique().tolist())
            slide_ids = [s for s in both_ids if s in train_ids]
        elif args.csv_path and os.path.isfile(args.csv_path):
            csv_ids = set(get_slide_ids_from_csv(args.csv_path, data_dir))
            slide_ids = [s for s in both_ids if s in csv_ids]
        else:
            slide_ids = both_ids
        if not slide_ids:
            raise SystemExit("No slide IDs in both clean/noisy after split/csv filter")
        print("Paired training: {} slides (clean + noisy)".format(len(slide_ids)))
        dataset = PairedPatchFeatureDataset(data_dir, args.noisy_dir, slide_ids, args.embed_dim)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    else:
        if args.split_dir and os.path.isfile(os.path.join(args.split_dir, 'splits_{}.csv'.format(args.split_fold))):
            split_df = pd.read_csv(os.path.join(args.split_dir, 'splits_{}.csv'.format(args.split_fold)))
            if 'train' in split_df.columns:
                slide_ids = split_df['train'].dropna().astype(str).unique().tolist()
            else:
                slide_ids = split_df.iloc[:, 0].dropna().astype(str).unique().tolist()
            slide_ids = [s for s in slide_ids if os.path.isfile(os.path.join(data_dir, 'h5_files', '{}.h5'.format(s)))]
        elif args.csv_path and os.path.isfile(args.csv_path):
            slide_ids = get_slide_ids_from_csv(args.csv_path, data_dir)
        else:
            slide_ids = get_slide_ids_from_dir(data_dir)
        if not slide_ids:
            raise SystemExit("No slide IDs found")
        dataset = PatchFeatureDataset(data_dir, slide_ids, args.embed_dim)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    denoiser = DDPMFeatureDenoiser(embed_dim=args.embed_dim, T=args.T).to(device)
    schedule = get_ddpm_schedule(args.T, device=device)
    optim = torch.optim.Adam(denoiser.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        denoiser.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            if use_paired:
                h_clean, h_noisy = batch
                h_clean = h_clean.to(device)
                h_noisy = h_noisy.to(device)
            else:
                h_clean = batch.to(device)

            B = h_clean.size(0)
            n_id = max(1, int(B * args.frac_id))
            n_ddpm = B - n_id

            # L_id: t=0, target epsilon=0
            if n_id > 0:
                h_id = h_clean[:n_id]
                t_id = torch.zeros(n_id, dtype=torch.long, device=device)
                eps_pred_id = denoiser.predict_epsilon(h_id, t_id)
                L_id = (eps_pred_id ** 2).mean()
            else:
                L_id = torch.tensor(0.0, device=device)

            # L_ddpm: t ~ Uniform(1, T-1), noising, predict epsilon
            if n_ddpm > 0:
                h_ddpm = h_clean[n_id:]
                t_ddpm = torch.randint(1, args.T, (n_ddpm,), device=device)
                h_t, eps_target = noisify(h_ddpm, t_ddpm, schedule)
                eps_pred_ddpm = denoiser.predict_epsilon(h_t, t_ddpm)
                L_ddpm = ((eps_pred_ddpm - eps_target) ** 2).mean()
            else:
                L_ddpm = torch.tensor(0.0, device=device)

            loss = L_ddpm + args.lambda_id * L_id

            # L_pair: denoise(noisy) -> clean (when paired data)
            if use_paired:
                h_denoised = denoiser.denoise(h_noisy, t_start=args.ddpm_t_start, num_steps=args.ddpm_num_steps)
                L_pair = ((h_denoised - h_clean) ** 2).mean()
                loss = loss + args.lambda_pair * L_pair

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batches += 1

        print("Epoch {} loss {:.4f}".format(epoch + 1, total_loss / n_batches))

    ckpt_dir = os.path.dirname(args.ckpt_save)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = {
        'state_dict': denoiser.state_dict(),
        'embed_dim': args.embed_dim,
        'T': args.T,
    }
    torch.save(ckpt, args.ckpt_save)
    print("Saved checkpoint to", args.ckpt_save)


if __name__ == '__main__':
    main()
