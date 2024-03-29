import os
import shutil
import pickle
import torch
import numpy as np
import scipy
from scipy import stats
import pandas as pd

class ContModelScore:
    def __init__(self, best_epoch_indx, spearman_corr, pearson_corr):
        self.best_epoch_indx = best_epoch_indx
        self.spearman_corr = spearman_corr
        self.pearson_corr = pearson_corr

    def __repr__(self):
        desc = " best_epoch_indx:{}\n spearman_corr:{}\n pearson_corr:{}\n" \
               "".format(self.best_epoch_indx, self.spearman_corr, self.pearson_corr)
        return desc

def build_predictions_df(seq_ids, true_score, pred_score, y_ref_names):
         
    
    seqid_inpseq_df = pd.DataFrame(seq_ids)
    seqid_inpseq_df.columns = ['seq_id']
    
    target_names = ['averageedited', 'averageunedited', 'averageindel']

    assert (len(y_ref_names) > 0 and len(y_ref_names) <= 3), f'# of target outcomes should be > 0 and not exceed 3!. Possible outcome names are:\n {target_outcomes}'

    df_dict = {}


    if true_score is not None:
        true_score_arr = np.array(true_score)
        if len(true_score_arr.shape) == 1: # (nsamples,)
            true_score_arr = true_score_arr.reshape(-1,1)
        
        num_targets = true_score_arr.shape[-1]
        assert num_targets <= 3, 'number of targets should not exceed three outcomes: averageedited, averageunedtied, averageidnel!'
        true_scores_dict = {}
        for i in range (num_targets):
            target_name = y_ref_names[i]
            true_scores_dict[f'true_{target_name}'] = true_score_arr[:, i]
        df_dict.update(true_scores_dict)

    pred_score_arr = np.array(pred_score)
    if len(pred_score_arr.shape) == 1: # (nsamples,)
        pred_score_arr = pred_score_arr.reshape(-1, 1)
    
    num_targets = pred_score_arr.shape[-1]
    assert num_targets in {1,3}, '# of predicted outcomes should be 1 or 3'
    pred_scores_dict = {}
    if num_targets == 3:
        for i in range (num_targets):
            target_name = target_names[i]
            pred_scores_dict[f'pred_{target_name}'] = pred_score_arr[:, i]
    elif num_targets == 1:
        target_name = y_ref_names[0]
        pred_scores_dict[f'pred_{target_name}'] = pred_score_arr[:, 0]
 

    # print('true_score_arr.shape:', true_score_arr.shape)
    # print('pred_score_arr.shape:',pred_score_arr.shape)
    df_dict.update(pred_scores_dict)
    predictions_df = pd.concat([seqid_inpseq_df, pd.DataFrame(df_dict)], axis=1)
    return predictions_df


class ReaderWriter(object):
    """class for dumping, reading and logging data"""
    def __init__(self):
        pass

    @staticmethod
    def dump_data(data, file_name, mode="wb"):
        """dump data by pickling
           Args:
               data: data to be pickled
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            pickle.dump(data, f)

    @staticmethod
    def read_data(file_name, mode="rb"):
        """read dumped/pickled data
           Args:
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            data = pickle.load(f)
        return(data)
        
    @staticmethod
    def dump_tensor(data, file_name):
        """
        Dump a tensor using PyTorch's custom serialization. Enables re-loading the tensor on a specific gpu later.
        Args:
            data: Tensor
            file_name: file path where data will be dumped
        Returns:
        """
        torch.save(data, file_name)

    @staticmethod
    def read_tensor(file_name, device):
        """read dumped/pickled data
           Args:
               file_name: file path where data will be dumped
               device: the gpu to load the tensor on to
        """
        data = torch.load(file_name, map_location=device)
        return data

    @staticmethod
    def write_log(line, outfile, mode="a"):
        """write data to a file
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(outfile, mode) as f:
            f.write(line)

    @staticmethod
    def read_log(file_name, mode="r"):
        """write data to a file
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(file_name, mode) as f:
            for line in f:
                yield line

def switch_layer_to_traineval_mode(model, target_layer, activate_train=True):
    """
    Target a layer and switch to train or eval mode
    """
    for child_name, child in model.named_children():
        if isinstance(child, target_layer):
            print(child_name, '=>', target_layer, 'is switching training mode to ', activate_train)
            if activate_train:
                child.train()
            else:
                child.eval()
            
        else:
            switch_layer_to_traineval_mode(child, target_layer, activate_train=activate_train)

def grad_track_hook(tensor_name):
    def print_hook(grad):
        pass
        # print('grad for ', tensor_name, ' is computed with grad_shape:', grad.shape, ' and grad_nrom:', grad.norm())
        # print('*'*15)
    return print_hook

def require_nonleaf_grad(v, tensor_name):
    v.retain_grad()
    v.register_hook(grad_track_hook(tensor_name))


def create_directory(folder_name, directory="current"):
    """create directory/folder (if it does not exist) and returns the path of the directory
       Args:
           folder_name: string representing the name of the folder to be created
       Keyword Arguments:
           directory: string representing the directory where to create the folder
                      if `current` then the folder will be created in the current directory
    """
    if directory == "current":
        path_current_dir = os.path.dirname(__file__)  # __file__ refers to utilities.py
    else:
        path_current_dir = directory
    path_new_dir = os.path.join(path_current_dir, folder_name)
    if not os.path.exists(path_new_dir):
        os.makedirs(path_new_dir)
    return(path_new_dir)


def get_device(to_gpu, index=0):
    is_cuda = torch.cuda.is_available()
    if(is_cuda and to_gpu):
        target_device = 'cuda:{}'.format(index)
    else:
        target_device = 'cpu'
    return torch.device(target_device)


def report_available_cuda_devices():
    if(torch.cuda.is_available()):
        n_gpu = torch.cuda.device_count()
        print('number of GPUs available:', n_gpu)
        for i in range(n_gpu):
            print("cuda:{}, name:{}".format(i, torch.cuda.get_device_name(i)))
            device = torch.device('cuda', i)
            get_cuda_device_stats(device)
            print()
    else:
        print("no GPU devices available!!")

def get_cuda_device_stats(device):
    print('total memory available:', torch.cuda.get_device_properties(device).total_memory/(1024**3), 'GB')
    print('total memory allocated on device:', torch.cuda.memory_allocated(device)/(1024**3), 'GB')
    print('max memory allocated on device:', torch.cuda.max_memory_allocated(device)/(1024**3), 'GB')
    print('total memory cached on device:', torch.cuda.memory_cached(device)/(1024**3), 'GB')
    print('max memory cached  on device:', torch.cuda.max_memory_cached(device)/(1024**3), 'GB')

def compute_harmonic_mean(a, b):
    assert (a >= 0) & (b>=0), 'one (or both) of the arguments is negative!'
    if a==0 and b==0:
        return 0.
    return 2*a*b/(a+b)
    
def compute_spearman_corr(pred_score, ref_score):
    # return scipy.stats.kendalltau(pred_score, ref_score)
    return scipy.stats.spearmanr(pred_score, ref_score)

def compute_pearson_corr(pred_score, ref_score):
    # return scipy.stats.kendalltau(pred_score, ref_score)
    return scipy.stats.pearsonr(pred_score, ref_score)

def check_na(df):
    assert df.isna().any().sum() == 0

def delete_directory(directory):
    if(os.path.isdir(directory)):
        shutil.rmtree(directory)
        
def transform_genseq_upper(df, columns):
    for colname in columns:
        df[colname] = df[colname].str.upper()
    return df

