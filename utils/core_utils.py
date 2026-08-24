import numpy as np
import torch
from utils.utils import *
import os
import csv
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _patch_topk_delta_for_device():
    """topk.utils.delta(y, labels, alpha)에서 labels가 CPU에 있으면 DDP에서 device 불일치 발생.
    delta를 wrapper로 교체해 labels를 y와 같은 device로 옮긴 뒤 원래 delta 호출.
    topk.functional이 이미 delta를 import해 둔 경우를 위해 functional.delta도 덮어씀."""
    try:
        import topk.utils as _tu
        _orig = getattr(_tu, 'delta', None)
        if _orig is None:
            return
        def _wrapped(y, labels, alpha):
            labels = labels.to(y.device)
            return _orig(y, labels, alpha)
        _tu.delta = _wrapped
        import topk.functional as _tf
        if hasattr(_tf, 'delta'):
            _tf.delta = _wrapped
    except Exception:
        pass


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False, rank=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
            rank (int): In DDP, only rank 0 saves checkpoints. Default 0.
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.rank = rank
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # Use np.inf for compatibility with NumPy >= 2.0
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease. DDP-safe: only rank 0 saves; saves model.module when wrapped.'''
        if self.rank != 0:
            self.val_loss_min = val_loss
            return
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state, ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args, rank=0, world_size=1, local_rank=0):
    """
    Train for a single fold. When world_size > 1, uses DDP for training only (train loader sharded, val/test on all ranks).
    """
    # DDP: set device per process
    if world_size > 1:
        import utils.core_utils as _cu
        from utils import utils as _utils
        _cu.device = torch.device('cuda:{}'.format(local_rank))
        _utils.device = _cu.device
        device = _cu.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if rank == 0:
        if not os.path.isdir(writer_dir):
            os.mkdir(writer_dir)
        if args.log_data:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(writer_dir, flush_secs=15)
        else:
            writer = None
    else:
        writer = None

    if rank == 0:
        print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    if rank == 0:
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
        print('Done!')
        print("Training on {} samples".format(len(train_split)))
        print("Validating on {} samples".format(0 if val_split is None else len(val_split)))
        print("Testing on {} samples".format(len(test_split)))

    if rank == 0:
        print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        _patch_topk_delta_for_device()
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.to(device)
    else:
        loss_fn = nn.CrossEntropyLoss()
    if rank == 0:
        print('Done!')

    if rank == 0:
        print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out,
                  'n_classes': args.n_classes,
                  "embed_dim": args.embed_dim}

    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        if getattr(args, 'use_maqw', False):
            model_dict.update({'use_maqw': True})
            n_c = getattr(args, 'maqw_multi_n_channels', 1)
            if n_c > 1:
                model_dict.update({'use_maqw_multi': True, 'maqw_multi_n_channels': n_c})
            else:
                model_dict.update({'use_maqw_multi': False})
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        if getattr(args, 'use_ddpm_denoise', False):
            model_dict.update({
                'use_ddpm_denoise': True,
                'ddpm_ckpt_path': getattr(args, 'ddpm_ckpt', None),
                'ddpm_t_start': getattr(args, 'ddpm_t_start', 20),
                'ddpm_num_steps': getattr(args, 'ddpm_num_steps', 20),
            })
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            _patch_topk_delta_for_device()
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.to(device)
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    else:
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    model = model.to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    if rank == 0:
        print('Done!')
        print_network(model.module if hasattr(model, 'module') else model)

    if rank == 0:
        print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    if rank == 0:
        print('Done!')

    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs, eta_min=args.min_lr)
        if rank == 0:
            print(f'Cosine LR schedule: {args.lr:.2e} -> {args.min_lr:.2e} over {args.max_epochs} epochs')
    else:
        scheduler = None

    if rank == 0:
        print('\nInit Loaders...', end=' ')
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_split, shuffle=True)
        train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=False, train_sampler=train_sampler)
    else:
        train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    if args.no_val and args.early_stopping:
        raise ValueError('--no_val and --early_stopping cannot be used together')
    val_loader = None if args.no_val else get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)
    if rank == 0:
        print('Done!')

    if rank == 0:
        print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True, rank=rank)
    else:
        early_stopping = None
    if rank == 0:
        print('Done!')

    ckpt_path = os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))
    for epoch in range(args.max_epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = False if args.no_val else validate_clam(
                cur, epoch, model, val_loader, args.n_classes,
                early_stopping, writer, loss_fn, args.results_dir)
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = False if args.no_val else validate(
                cur, epoch, model, val_loader, args.n_classes,
                early_stopping, writer, loss_fn, args.results_dir)
        if scheduler is not None:
            scheduler.step()
            if writer:
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
        if stop:
            break

    # Save / load checkpoint; only rank 0 runs summary and writes files
    if rank == 0:
        if args.early_stopping:
            state = torch.load(ckpt_path, map_location=device)
            (model.module if hasattr(model, 'module') else model).load_state_dict(state)
        else:
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state, ckpt_path)

        if args.no_val:
            val_error, val_auc = float('nan'), float('nan')
        else:
            _, val_error, val_auc, _ = summary(model, val_loader, args.n_classes, results_dir=args.results_dir, split_name='val')
            print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
        results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes, results_dir=args.results_dir, split_name='test')
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
        for i in range(args.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
            if writer:
                writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
        if writer:
            if not args.no_val:
                writer.add_scalar('final/val_error', val_error, 0)
                writer.add_scalar('final/val_auc', val_auc, 0)
            writer.add_scalar('final/test_error', test_error, 0)
            writer.add_scalar('final/test_auc', test_auc, 0)
            writer.close()
        return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error

    if world_size > 1:
        torch.distributed.barrier()
    return None, 0.0, 0.0, 0.0, 0.0 


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 4:
            data, label, coords, laplacian_scores = batch
        else:
            data, label = batch[0], batch[1]
            laplacian_scores = None
        data, label = data.to(device), label.to(device)
        if laplacian_scores is not None:
            laplacian_scores = laplacian_scores.to(device, non_blocking=True)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True, laplacian_scores=laplacian_scores)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 4:
            data, label, coords, laplacian_scores = batch
        else:
            data, label = batch[0], batch[1]
            laplacian_scores = None
        data, label = data.to(device), label.to(device)
        if laplacian_scores is not None:
            laplacian_scores = laplacian_scores.to(device, non_blocking=True)
        logits, Y_prob, Y_hat, _, _ = model(data, laplacian_scores=laplacian_scores)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    maqw_logs = {"tau_L": [], "k_L": [], "tau_R": [], "k_R": [], "w_mean": []}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 4:
                data, label, coords, laplacian_scores = batch
            else:
                data, label = batch[0], batch[1]
                laplacian_scores = None
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            if laplacian_scores is not None:
                laplacian_scores = laplacian_scores.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, results_dict = model(data, laplacian_scores=laplacian_scores)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

            maqw = results_dict.get('maqw', None) if isinstance(results_dict, dict) else None
            if maqw is not None:
                maqw_logs["tau_L"].append(float(maqw["tau_L"].detach().cpu()))
                maqw_logs["k_L"].append(float(maqw["k_L"].detach().cpu()))
                maqw_logs["tau_R"].append(float(maqw["tau_R"].detach().cpu()))
                maqw_logs["k_R"].append(float(maqw["k_R"].detach().cpu()))
                maqw_logs["w_mean"].append(float(maqw["w_mean"].detach().cpu()))
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        if len(maqw_logs["tau_L"]) > 0:
            writer.add_histogram('maqw/val/tau_L', np.array(maqw_logs["tau_L"]), epoch)
            writer.add_histogram('maqw/val/k_L', np.array(maqw_logs["k_L"]), epoch)
            writer.add_histogram('maqw/val/tau_R', np.array(maqw_logs["tau_R"]), epoch)
            writer.add_histogram('maqw/val/k_R', np.array(maqw_logs["k_R"]), epoch)
            writer.add_scalar('maqw/val/w_mean_mean', float(np.mean(maqw_logs["w_mean"])), epoch)

    if len(maqw_logs["tau_L"]) > 0:
        print('M-AQW val: tau_L={:.4f}, k_L={:.4f}, tau_R={:.4f}, k_R={:.4f}, w_mean={:.4f}'.format(
            float(np.mean(maqw_logs["tau_L"])), float(np.mean(maqw_logs["k_L"])),
            float(np.mean(maqw_logs["tau_R"])), float(np.mean(maqw_logs["k_R"])),
            float(np.mean(maqw_logs["w_mean"]))))
    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    _model = model.module if hasattr(model, 'module') else model
    sample_size = _model.k_sample
    maqw_logs = {
        "tau_L": [], "k_L": [], "tau_R": [], "k_R": [],
        "w_mean": [], "w_lt_0p1": [], "w_gt_0p9": []
    }
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 4:
                data, label, coords, laplacian_scores = batch
            else:
                data, label = batch[0], batch[1]
                laplacian_scores = None
            data, label = data.to(device), label.to(device)
            if laplacian_scores is not None:
                laplacian_scores = laplacian_scores.to(device, non_blocking=True)
            logits, Y_prob, Y_hat, _, results_dict = model(data, label=label, instance_eval=True, laplacian_scores=laplacian_scores)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = results_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = results_dict['inst_preds']
            inst_labels = results_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

            maqw = results_dict.get('maqw', None) if isinstance(results_dict, dict) else None
            if maqw is not None:
                maqw_logs["tau_L"].append(float(maqw["tau_L"].detach().cpu()))
                maqw_logs["k_L"].append(float(maqw["k_L"].detach().cpu()))
                maqw_logs["tau_R"].append(float(maqw["tau_R"].detach().cpu()))
                maqw_logs["k_R"].append(float(maqw["k_R"].detach().cpu()))
                maqw_logs["w_mean"].append(float(maqw["w_mean"].detach().cpu()))
                maqw_logs["w_lt_0p1"].append(float(maqw["w_lt_0p1"].detach().cpu()))
                maqw_logs["w_gt_0p9"].append(float(maqw["w_gt_0p9"].detach().cpu()))

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)
        if len(maqw_logs["tau_L"]) > 0:
            writer.add_histogram('maqw/val/tau_L', np.array(maqw_logs["tau_L"]), epoch)
            writer.add_histogram('maqw/val/k_L', np.array(maqw_logs["k_L"]), epoch)
            writer.add_histogram('maqw/val/tau_R', np.array(maqw_logs["tau_R"]), epoch)
            writer.add_histogram('maqw/val/k_R', np.array(maqw_logs["k_R"]), epoch)
            writer.add_scalar('maqw/val/tau_L_mean', float(np.mean(maqw_logs["tau_L"])), epoch)
            writer.add_scalar('maqw/val/tau_R_mean', float(np.mean(maqw_logs["tau_R"])), epoch)
            writer.add_scalar('maqw/val/k_L_mean', float(np.mean(maqw_logs["k_L"])), epoch)
            writer.add_scalar('maqw/val/k_R_mean', float(np.mean(maqw_logs["k_R"])), epoch)
            writer.add_scalar('maqw/val/w_mean_mean', float(np.mean(maqw_logs["w_mean"])), epoch)
            writer.add_scalar('maqw/val/w_lt_0p1_mean', float(np.mean(maqw_logs["w_lt_0p1"])), epoch)
            writer.add_scalar('maqw/val/w_gt_0p9_mean', float(np.mean(maqw_logs["w_gt_0p9"])), epoch)

    if len(maqw_logs["tau_L"]) > 0:
        print('M-AQW val: tau_L={:.4f}, k_L={:.4f}, tau_R={:.4f}, k_R={:.4f}, w_mean={:.4f}, w_lt_0p1={:.4f}, w_gt_0p9={:.4f}'.format(
            float(np.mean(maqw_logs["tau_L"])), float(np.mean(maqw_logs["k_L"])),
            float(np.mean(maqw_logs["tau_R"])), float(np.mean(maqw_logs["k_R"])),
            float(np.mean(maqw_logs["w_mean"])), float(np.mean(maqw_logs["w_lt_0p1"])),
            float(np.mean(maqw_logs["w_gt_0p9"]))))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes, results_dir=None, split_name='test'):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    maqw_rows = []

    for batch_idx, batch in enumerate(loader):
        if len(batch) == 4:
            data, label, coords, laplacian_scores = batch
        else:
            data, label = batch[0], batch[1]
            laplacian_scores = None
        data, label = data.to(device), label.to(device)
        if laplacian_scores is not None:
            laplacian_scores = laplacian_scores.to(device, non_blocking=True)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, results_dict = model(data, laplacian_scores=laplacian_scores)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

        maqw = results_dict.get('maqw', None) if isinstance(results_dict, dict) else None
        if maqw is not None:
            row = {
                "slide_id": str(slide_id),
                "label": int(label.item()),
                "Y_hat": int(Y_hat.item()) if hasattr(Y_hat, "item") else int(Y_hat),
            }
            # probs can be shape [C] or [1, C]
            if probs.ndim == 1:
                for c in range(probs.shape[0]):
                    row[f"prob_{c}"] = float(probs[c])
            else:
                for c in range(probs.shape[1]):
                    row[f"prob_{c}"] = float(probs[0, c])

            for k in ["tau_L", "k_L", "tau_R", "k_R",
                      "q_mean", "q_std", "q_min", "q_max", "q_p25", "q_p75",
                      "w_mean", "w_std", "w_lt_0p1", "w_gt_0p9"]:
                if k in maqw:
                    row[k] = float(maqw[k].detach().cpu())
            for k in maqw:
                if k in row:
                    continue
                v = maqw[k]
                if hasattr(v, "detach"):
                    v = v.detach().cpu()
                if hasattr(v, "numel") and v.numel() == 1:
                    row[k] = float(v)
                elif hasattr(v, "numpy") and hasattr(v, "shape") and len(v.shape) == 1:
                    row[k] = ",".join([f"{x:.6f}" for x in v.numpy().tolist()])
            maqw_rows.append(row)

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    if results_dir is not None and len(maqw_rows) > 0:
        out_path = os.path.join(results_dir, f"maqw_{split_name}_details.csv")
        fieldnames = list(maqw_rows[0].keys())
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(maqw_rows)

    return patient_results, test_error, auc, acc_logger
