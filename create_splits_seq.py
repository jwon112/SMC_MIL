import pdb
import os
import pandas as pd
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping', 'task_3_camelyon16_binary', 'task_4_camelyon16_multiclass', 'task_smc_acr_binary_0r_vs_1r2r3r', 'task_smc_amr_binary_pamr0_vs_positive', 'task_smc_any_rejection_binary'])
parser.add_argument('--csv_path', type=str, default=None, help='custom csv path (overrides task default)')
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'task_3_camelyon16_binary':
    args.n_classes=2
    csv_path = args.csv_path if args.csv_path else 'dataset_csv/camelyon16_binary.csv'
    dataset = Generic_WSI_Classification_Dataset(csv_path = csv_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal':0, 'tumor':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_4_camelyon16_multiclass':
    args.n_classes=3
    csv_path = args.csv_path if args.csv_path else 'dataset_csv/camelyon16_multiclass.csv'
    dataset = Generic_WSI_Classification_Dataset(csv_path = csv_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'negative':0, 'micro':1, 'macro':2},
                            patient_strat= False,
                            ignore=[])

elif args.task == 'task_smc_acr_binary_0r_vs_1r2r3r':
    args.n_classes=2
    csv_path = args.csv_path if args.csv_path else 'dataset_csv/smc_acr_binary_0r_vs_1r2r3r.csv'
    dataset = Generic_WSI_Classification_Dataset(csv_path = csv_path,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {0:0, 1:1},
                            patient_strat=True,
                            patient_voting='max',
                            ignore=[])

elif args.task == 'task_smc_amr_binary_pamr0_vs_positive':
    args.n_classes=2
    csv_path = args.csv_path if args.csv_path else 'dataset_csv/smc_amr_binary_pamr0_vs_positive.csv'
    dataset = Generic_WSI_Classification_Dataset(csv_path = csv_path,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {0:0, 1:1},
                            patient_strat=True,
                            patient_voting='max',
                            ignore=[])

elif args.task == 'task_smc_any_rejection_binary':
    args.n_classes=2
    csv_path = args.csv_path if args.csv_path else 'dataset_csv/smc_any_rejection_binary.csv'
    dataset = Generic_WSI_Classification_Dataset(csv_path = csv_path,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {0:0, 1:1},
                            patient_strat=True,
                            patient_voting='max',
                            ignore=[])

else:
    raise NotImplementedError

if args.task in ['task_3_camelyon16_binary', 'task_4_camelyon16_multiclass']:
    # patient_strat=False이므로 slide_cls_ids 사용
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.slide_cls_ids])
else:
    # patient_strat=True이므로 patient_cls_ids 사용
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))
