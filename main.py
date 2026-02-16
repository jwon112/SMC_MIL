from __future__ import print_function

import argparse
import pdb
import os
import math

# Patch topk.utils.delta so labels are on same device as y (fixes DDP + SmoothTop1SVM device mismatch).
# Must run before any code that imports topk (e.g. core_utils which uses SmoothTop1SVM).
try:
    import topk.utils as _topk_utils
    _topk_delta_orig = getattr(_topk_utils, 'delta', None)
    if _topk_delta_orig is not None:
        def _topk_delta_patch(y, labels, alpha):
            labels = labels.to(y.device)
            return _topk_delta_orig(y, labels, alpha)
        _topk_utils.delta = _topk_delta_patch
        import topk.functional as _topk_fn
        if hasattr(_topk_fn, 'delta'):
            _topk_fn.delta = _topk_delta_patch
except Exception:
    pass

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(args, rank=0, world_size=1, local_rank=0):
    # create results directory if necessary (only rank 0 in DDP)
    if rank == 0 and not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if world_size > 1:
        torch.distributed.barrier()

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, args, rank=rank, world_size=world_size, local_rank=local_rank)
        if rank == 0:
            all_test_auc.append(test_auc)
            all_val_auc.append(val_auc)
            all_test_acc.append(test_acc)
            all_val_acc.append(val_acc)
            filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
            save_pkl(filename, results)
        if world_size > 1:
            torch.distributed.barrier()

    if rank == 0:
        final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
            'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc': all_val_acc})
        if len(folds) != args.k:
            save_name = 'summary_partial_{}_{}.csv'.format(start, end)
        else:
            save_name = 'summary.csv'
        final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'task_3_camelyon16_binary', 'task_4_camelyon16_multiclass'])
parser.add_argument('--csv_path', type=str, default=None, help='custom csv path (overrides task default)')
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--use_maqw', action='store_true', default=False,
                    help='enable M-AQW (Meta-Parametric Asymmetric Quality-Aware Weighting); requires H5 with quality metrics')
parser.add_argument('--maqw_metrics', type=str, default='laplacian',
                    help='comma-separated quality metrics for M-AQW. 1 metric = single-indicator; 2+ = multi-indicator. '
                         'Allowed: laplacian, tenengrad, vgm, wavelet, stain_saturation, color_entropy, contrast. Example: laplacian or laplacian,tenengrad,vgm')
parser.add_argument('--distributed', action='store_true', default=False,
                    help='enable DDP multi-GPU training (use with torchrun)')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping 

elif args.task == 'task_3_camelyon16_binary':
    args.n_classes=2
    csv_path = args.csv_path if args.csv_path else 'dataset_csv/camelyon16_binary.csv'
    dataset = Generic_MIL_Dataset(csv_path = csv_path,
                            data_dir= os.path.join(args.data_root_dir),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal':0, 'tumor':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_4_camelyon16_multiclass':
    args.n_classes=3
    csv_path = args.csv_path if args.csv_path else 'dataset_csv/camelyon16_multiclass.csv'
    dataset = Generic_MIL_Dataset(csv_path = csv_path,
                            data_dir= os.path.join(args.data_root_dir),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'negative':0, 'micro':1, 'macro':2},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping 
        
else:
    raise NotImplementedError

# drop slides that have no feature file (e.g. skipped during extraction)
dataset.filter_slides_by_available_files()

if args.use_maqw:
    dataset.load_from_h5(True)
    assert args.model_type in ['clam_sb', 'clam_mb'], 'M-AQW is only supported with CLAM models (clam_sb or clam_mb)'
if args.use_maqw:
    _metric_to_key = {
        'laplacian': 'laplacian_scores',
        'tenengrad': 'tenengrad',
        'vgm': 'vgm',
        'wavelet': 'wavelet_scores',
        'stain_saturation': 'stain_saturation',
        'color_entropy': 'color_entropy',
        'contrast': 'contrast',
    }
    _names = [s.strip() for s in args.maqw_metrics.split(',') if s.strip()]
    for m in _names:
        if m not in _metric_to_key:
            raise ValueError('maqw_metrics: unknown metric "{}". Allowed: {}'.format(m, list(_metric_to_key.keys())))
    _keys = [_metric_to_key[m] for m in _names]
    if len(_keys) == 1:
        dataset.maqw_metric_key = _keys[0]
        dataset.use_maqw_multi = False
    else:
        dataset.use_maqw_multi = True
        dataset.maqw_multi_keys = _keys
        args.maqw_multi_n_channels = len(_keys)

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})
if args.use_maqw:
    settings.update({'use_maqw': True, 'maqw_metrics': args.maqw_metrics})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world_size > 1:
        torch.distributed.init_process_group(backend='nccl')
    try:
        results = main(args, rank=rank, world_size=world_size, local_rank=local_rank)
        if rank == 0:
            print("finished!")
            print("end script")
    finally:
        if world_size > 1:
            torch.distributed.destroy_process_group()


