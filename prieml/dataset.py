import numpy as np
import torch
from torch.utils.data import Dataset

class PEDataTensor(Dataset):

    def __init__(self, 
                 X_init_nucl, 
                 X_init_proto, 
                 X_init_pbs,
                 X_init_rt,
                 X_mut_nucl,
                 X_mut_pbs,
                 X_mut_rt,
                 seqlevel_feat,
                 seqlevel_feat_colnames,
                 y_score, 
                 x_init_len,
                 x_mut_len,
                 indx_seqid_map):
        # B: batch elements; T: sequence length
        # tensor.float32, (B, T), (sequence characters are mapped to 0-3) and 4 for padded characters
        self.X_init_nucl = X_init_nucl
        self.X_init_proto = X_init_proto
        self.X_init_pbs = X_init_pbs
        self.X_init_rt = X_init_rt

        self.X_mut_nucl = X_mut_nucl
        self.X_mut_pbs = X_mut_pbs
        self.X_mut_rt = X_mut_rt
        
        self.seqlevel_feat = seqlevel_feat
        self.seqlevel_feat_colnames = seqlevel_feat_colnames
        # tensor.float32, (B,1), (editing score)
        self.y_score = y_score  
        # tensor.int32, (B,), (length of each sequence)
        self.x_init_len = x_init_len 
        self.x_mut_len = x_mut_len
        # dictionary {indx:seq_id}
        self.indx_seqid_map = indx_seqid_map
        self.num_samples = self.X_init_nucl.size(0)  # int, number of sequences

    def __getitem__(self, indx):
        if self.y_score is None:
            y_val = self.y_score
        else:
            y_val = self.y_score[indx]

        return(self.X_init_nucl[indx],
               self.X_init_proto[indx],
               self.X_init_pbs[indx],
               self.X_init_rt[indx],
               self.X_mut_nucl[indx],
               self.X_mut_pbs[indx],
               self.X_mut_rt[indx],
               self.x_init_len[indx],
               self.x_mut_len[indx],
               self.seqlevel_feat[indx],
               y_val,
               indx, 
               self.indx_seqid_map[indx])


    def __len__(self):
        return(self.num_samples)

class PartitionDataTensor(Dataset):

    def __init__(self, pe_datatensor, partition_ids, dsettype, run_num):
        self.pe_datatensor = pe_datatensor  # instance of :class:`PEDatatensor`
        self.partition_ids = partition_ids  # list of sequence indices
        self.dsettype = dsettype  # string, dataset type (i.e. train, validation, test)
        self.run_num = run_num  # int, run number
        self.num_samples = len(self.partition_ids[:])  # int, number of docs in the partition

    def __getitem__(self, indx):
        target_id = self.partition_ids[indx]
        return self.pe_datatensor[target_id]

    def __len__(self):
        return(self.num_samples)

def extend_matrix(mat, ext_mat_shape, fill_val):
    assert len(mat.shape) == 2
    ext_mat = torch.full(ext_mat_shape, fill_val)
    return torch.cat([mat, ext_mat], axis=1)

class MinMaxNormalizer:
    def __init__(self):
        self.length_norm = ['Correction_Length', 'RToverhangmatches', 'RToverhanglength', 'RTlength','PBSlength']
        self.mfe_norm = ['MFE_protospacer', 
                        'MFE_protospacer_scaffold', 
                        'MFE_extension', 
                        'MFE_extension_scaffold', 
                        'MFE_protospacer_extension_scaffold',
                        'MFE_rt',
                        'MFE_pbs']
        self.mt_norm = ['RTmt', 'RToverhangmt','PBSmt',
                        'protospacermt','extensionmt',
                        'original_base_mt','edited_base_mt']
        self.normalizer_info_minmax = [(self.length_norm, 0., 50.), 
                                       (self.mfe_norm, -120., 0.), 
                                       (self.mt_norm, 0., 200.)]
        self.normalizer_info_max = [(self.length_norm, 50.), 
                                    (self.mfe_norm, 120.), 
                                    (self.mt_norm, 200.)]
    def get_colnames(self):
        cont_colnames = []
        for featlst in [self.length_norm, self.mfe_norm, self.mt_norm]:
            cont_colnames.extend(featlst)
        return cont_colnames

    def normalize_cont_cols(self, df, normalize_opt = 'max', suffix=''):
        if normalize_opt == 'max':
            print('--- max normalization ---')
            return self.normalize_cont_cols_max(df, suffix=suffix)
        elif normalize_opt == 'minmax':
            print('--- minmax normalization ---')
            return self.normalize_cont_cols_minmax(df, suffix=suffix)

    def normalize_cont_cols_minmax(self, df, suffix=''):
        """inplace min-max normalization of columns"""
        normalizer_info = self.normalizer_info_minmax
        cont_colnames = []
        for colgrp in normalizer_info:
            colnames, min_val, max_val = colgrp
            for colname in colnames:
                df[colname+suffix] = ((df[colname] - min_val)/(max_val - min_val)).clip(lower=0., upper=1.)
                cont_colnames.append(colname+suffix)
        return cont_colnames

    def normalize_cont_cols_max(self, df, suffix=''):
        """inplace max normalization of columns"""
        normalizer_info = self.normalizer_info_max
        cont_colnames = []
        for colgrp in normalizer_info:
            colnames, max_val = colgrp
            for colname in colnames:
                df[colname+suffix] = df[colname]/max_val
                cont_colnames.append(colname+suffix)
        return cont_colnames

def get_seqlevel_featnames():
    minmax_normalizer = MinMaxNormalizer()
    cont_colnames = minmax_normalizer.get_colnames()
    norm_colnames =  [f'{fname}_norm' for fname in cont_colnames]
    seqfeat_cols = [norm_colnames[0]] + \
                   ['Correction_Deletion', 'Correction_Insertion', 'Correction_Replacement'] + \
                   norm_colnames[1:] + ['original_base_mt_nan', 'edited_base_mt_nan']
    return seqfeat_cols

def create_datatensor(data_df, proc_seq_init_df, num_init_cols,  proc_seq_mut_df, num_mut_cols, cont_cols, window=10, y_ref=[]):
    """create a instance of DataTensor from processeed/cleaned dataframe
    
    Args:
        data_df: pandas.DataFrame, dataset
    """
    
    print('--- create_datatensor ---')

    lower_thr = 0
    upper_thr = 150
    blank_nucl_token = 4
    blank_annot_token = 2

    # process initial sequences
    start_init_seqs = (proc_seq_init_df['start_seq'] - window).clip(lower=lower_thr, upper=None)
    st_init_colindx = start_init_seqs.min()
    
    # end of RT stretch
    end_init_seqs = proc_seq_init_df['end_seq']
    # print('end_init_seqs:\n', end_init_seqs.value_counts())
    # to put assert statment
    assert ((end_init_seqs - start_init_seqs) <= upper_thr).all(), f'Difference between end and start seuqnce should be at most {upper_thr} bp'
    end_init_colindx = end_init_seqs.max()
    upper_init_thr = np.min([st_init_colindx+upper_thr, num_init_cols-1, end_init_colindx + window]) # -1 to compensate for not including end value
    end_init_seqs = (end_init_seqs + window).clip(lower=None, upper=upper_init_thr)
    end_init_colindx = end_init_seqs.max()
    # print('updated end_init_seqs:\n', end_init_seqs.value_counts())
    # print('st_init_colindx:', st_init_colindx)
    # print('end_init_colindx:', end_init_colindx)
    # print('x_init_len_comp:\n', (end_init_seqs - start_init_seqs).value_counts())

    X_init_nucl = torch.from_numpy(proc_seq_init_df[[f'B{i}' for  i in range(st_init_colindx, end_init_colindx+1)]].values).long()
    X_init_proto = torch.from_numpy(proc_seq_init_df[[f'Protos{i}' for  i in range(st_init_colindx, end_init_colindx+1)]].values).long()
    X_init_pbs = torch.from_numpy(proc_seq_init_df[[f'PBS{i}' for  i in range(st_init_colindx, end_init_colindx+1)]].values).long()
    X_init_rt = torch.from_numpy(proc_seq_init_df[[f'RT{i}' for  i in range(st_init_colindx, end_init_colindx+1)]].values).long()
    

    # x_init_len = (X_init_nucl != 4).sum(axis=1).long()

    # TODO: we could use ().values for x_init_len and x_mut_len to keep them in array format rather pd.Series
    x_init_len  = (end_init_seqs - start_init_seqs)

    # use blank indicator when nucleotide is N (i.e. blank token)
    ntoken_cond = (X_init_nucl == blank_nucl_token)

    X_init_proto[ntoken_cond] = blank_annot_token
    X_init_pbs[ntoken_cond] = blank_annot_token
    X_init_rt[ntoken_cond] = blank_annot_token

    # print('X_init_nucl.unique():', X_init_nucl.unique())
    # print('ntoken_cond:', ntoken_cond.unique())
    # print('X_init_proto:', X_init_proto.unique())
    # print('X_init_pbs:', X_init_pbs.unique())
    # print('X_init_rt:', X_init_rt.unique())

    # process mutation sequences
    if 'start_seq' in proc_seq_mut_df:
        start_mut_seqs = (proc_seq_mut_df['start_seq'] - window).clip(lower=lower_thr, upper=None)
    else:
        start_mut_seqs = start_init_seqs
    
    st_mut_colindx = start_mut_seqs.min()

    end_mut_seqs = proc_seq_mut_df['end_seq']
    # print('end_mut_seqs:\n', end_mut_seqs.value_counts())

    assert ((end_mut_seqs - start_mut_seqs) <= upper_thr).all(), f'Difference between end and start seuqnce should be at most {upper_thr} bp'

    end_mut_colindx = end_mut_seqs.max()
    upper_mut_thr = np.min([st_mut_colindx+upper_thr, num_mut_cols-1, end_mut_colindx + window])
    end_mut_seqs = (end_mut_seqs + window).clip(lower=None, upper=upper_mut_thr)
    end_mut_colindx = end_mut_seqs.max()
    # print('updated end_mut_seqs:\n', end_mut_seqs.value_counts())

    # print('st_mut_colindx:', st_mut_colindx)
    # print('end_mut_colindx:', end_mut_colindx)
    # print('x_mut_len_comp:\n', (end_mut_seqs - start_mut_seqs).value_counts())

    X_mut_nucl = torch.from_numpy(proc_seq_mut_df[[f'B{i}' for  i in range(st_mut_colindx, end_mut_colindx+1)]].values).long()
    X_mut_pbs = torch.from_numpy(proc_seq_mut_df[[f'PBS{i}' for  i in range(st_mut_colindx, end_mut_colindx+1)]].values).long()
    X_mut_rt = torch.from_numpy(proc_seq_mut_df[[f'RT{i}' for  i in range(st_mut_colindx, end_mut_colindx+1)]].values).long()

    # x_mut_len = (X_mut_nucl != 4).sum(axis=1).long()
    x_mut_len  = (end_mut_seqs - start_mut_seqs)

    # use blank indicator when nucleotide is N (i.e. blank token)
    ntoken_cond = (X_mut_nucl == blank_nucl_token)
    X_mut_pbs[ntoken_cond] = blank_annot_token
    X_mut_rt[ntoken_cond] = blank_annot_token
    # print('X_mut_nucl.unique():', X_mut_nucl.unique())
    # print('ntoken_cond:', ntoken_cond.unique())
    # print('X_mut_pbs:', X_mut_pbs.unique())
    # print('X_mut_rt:', X_mut_rt.unique())

    # harmonize the size of matrices above
    max_num_cols = np.max([end_init_colindx+1, end_mut_colindx+1])
    # print('max_num_cols:', max_num_cols)
        
    annot_addendum_tok = blank_annot_token
    # annot_addendum_tok = 0
    if max_num_cols > (end_init_colindx+1):
        num_cols_toadd = max_num_cols - (end_init_colindx + 1)
        bsize = X_init_nucl.shape[0]
        ext_mat_shape = (bsize, num_cols_toadd)
        # mut_cols > init_cols
        X_init_nucl = extend_matrix(X_init_nucl, ext_mat_shape, blank_nucl_token)
        X_init_proto = extend_matrix(X_init_proto, ext_mat_shape, annot_addendum_tok)
        X_init_pbs = extend_matrix(X_init_pbs, ext_mat_shape, annot_addendum_tok)
        X_init_rt = extend_matrix(X_init_rt, ext_mat_shape, annot_addendum_tok)

    elif max_num_cols > (end_mut_colindx+1):
        num_cols_toadd = max_num_cols - (end_mut_colindx +1)
        bsize = X_mut_nucl.shape[0]
        ext_mat_shape = (bsize, num_cols_toadd)
        # init_cols > mut_cols
        X_mut_nucl = extend_matrix(X_mut_nucl, ext_mat_shape, blank_nucl_token)
        X_mut_pbs = extend_matrix(X_mut_pbs, ext_mat_shape, annot_addendum_tok)
        X_mut_rt = extend_matrix(X_mut_rt, ext_mat_shape, annot_addendum_tok)

    seq_ids = data_df['seq_id'].values
    indx_seqid_map = {i:seq_ids[i] for i in range(len(seq_ids))}

    if len(y_ref):
        # y_score = torch.from_numpy(data_df['y'].values).reshape(-1,1)
        ycols = [tcol for tcol in y_ref if tcol in data_df]
        # print('ycols:', ycols)
        y_score = torch.from_numpy(data_df[ycols].values)
    else:
        y_score = None

    # get computed features at sequence level
    seqfeat_cols = [cont_cols[0]] + ['Correction_Deletion', 'Correction_Insertion', 'Correction_Replacement'] + cont_cols[1:]

    # case of having indicator variables when melting temperature cannot be computed
    for colname in ('original_base_mt_nan', 'edited_base_mt_nan'):
        if colname in data_df:
            seqfeat_cols.append(colname)

    seqlevel_feat = torch.from_numpy(data_df[seqfeat_cols].values)

    dtensor = PEDataTensor(X_init_nucl, 
                            X_init_proto, 
                            X_init_pbs,
                            X_init_rt,
                            X_mut_nucl,
                            X_mut_pbs,
                            X_mut_rt,
                            seqlevel_feat,
                            seqfeat_cols,
                            y_score, 
                            x_init_len,
                            x_mut_len,
                            indx_seqid_map)
        
    return dtensor

# def create_datatensor(data_df, proc_seq_init_df, num_init_cols,  proc_seq_mut_df, num_mut_cols, cont_cols, window=10, y_ref=[]):
#     """create a instance of DataTensor from processeed/cleaned dataframe
    
#     Args:
#         data_df: pandas.DataFrame, dataset
#     """
    
#     print('--- create_datatensor ---')

#     lower_thr = 0
#     upper_thr = 150
#     blank_nucl_token = 4
#     blank_annot_token = 2

#     # process initial sequences
#     start_init_seqs = (proc_seq_init_df['start_seq'] - window).clip(lower=lower_thr, upper=None)
#     st_init_colindx = start_init_seqs.min()
    
#     # end of RT stretch
#     end_init_seqs = proc_seq_init_df['end_seq']
#     # print('end_init_seqs:\n', end_init_seqs.value_counts())
#     # to put assert statment
#     assert ((end_init_seqs - start_init_seqs) <= upper_thr).all(), f'Difference between end and start seuqnce should be at most {upper_thr} bp'
#     end_init_colindx = end_init_seqs.max()
#     upper_init_thr = np.min([st_init_colindx+upper_thr, num_init_cols-1, end_init_colindx + window]) # -1 to compensate for not including end value
#     end_init_seqs = (end_init_seqs + window).clip(lower=None, upper=upper_init_thr)
#     end_init_colindx = end_init_seqs.max()
#     # print('updated end_init_seqs:\n', end_init_seqs.value_counts())
#     # print('st_init_colindx:', st_init_colindx)
#     # print('end_init_colindx:', end_init_colindx)
#     # print('x_init_len_comp:\n', (end_init_seqs - start_init_seqs).value_counts())

#     X_init_nucl = torch.from_numpy(proc_seq_init_df[[f'B{i}' for  i in range(st_init_colindx, end_init_colindx+1)]].values).long()
#     X_init_proto = torch.from_numpy(proc_seq_init_df[[f'Protos{i}' for  i in range(st_init_colindx, end_init_colindx+1)]].values).long()
#     X_init_pbs = torch.from_numpy(proc_seq_init_df[[f'PBS{i}' for  i in range(st_init_colindx, end_init_colindx+1)]].values).long()
#     X_init_rt = torch.from_numpy(proc_seq_init_df[[f'RT{i}' for  i in range(st_init_colindx, end_init_colindx+1)]].values).long()
    

#     # x_init_len = (X_init_nucl != 4).sum(axis=1).long()
#     x_init_len  = (end_init_seqs - start_init_seqs)

#     # use blank indicator when nucleotide is N (i.e. blank token)
#     ntoken_cond = (X_init_nucl == blank_nucl_token)

#     X_init_proto[ntoken_cond] = blank_annot_token
#     X_init_pbs[ntoken_cond] = blank_annot_token
#     X_init_rt[ntoken_cond] = blank_annot_token

#     print('X_init_nucl.unique():', X_init_nucl.unique())
#     print('ntoken_cond:', ntoken_cond.unique())
#     print('X_init_proto:', X_init_proto.unique())
#     print('X_init_pbs:', X_init_pbs.unique())
#     print('X_init_rt:', X_init_rt.unique())

#     # process mutation sequences
#     if 'start_seq' in proc_seq_mut_df:
#         start_mut_seqs = (proc_seq_mut_df['start_seq'] - window).clip(lower=lower_thr, upper=None)
#     else:
#         start_mut_seqs = start_init_seqs
    
#     st_mut_colindx = start_mut_seqs.min()

#     end_mut_seqs = proc_seq_mut_df['end_seq']
#     # print('end_mut_seqs:\n', end_mut_seqs.value_counts())

#     assert ((end_mut_seqs - start_mut_seqs) <= upper_thr).all(), f'Difference between end and start seuqnce should be at most {upper_thr} bp'

#     end_mut_colindx = end_mut_seqs.max()
#     upper_mut_thr = np.min([st_mut_colindx+upper_thr, num_mut_cols-1, end_mut_colindx + window])
#     end_mut_seqs = (end_mut_seqs + window).clip(lower=None, upper=upper_mut_thr)
#     end_mut_colindx = end_mut_seqs.max()
#     # print('updated end_mut_seqs:\n', end_mut_seqs.value_counts())

#     # print('st_mut_colindx:', st_mut_colindx)
#     # print('end_mut_colindx:', end_mut_colindx)
#     # print('x_mut_len_comp:\n', (end_mut_seqs - start_mut_seqs).value_counts())

#     X_mut_nucl = torch.from_numpy(proc_seq_mut_df[[f'B{i}' for  i in range(st_mut_colindx, end_mut_colindx+1)]].values).long()
#     X_mut_pbs = torch.from_numpy(proc_seq_mut_df[[f'PBS{i}' for  i in range(st_mut_colindx, end_mut_colindx+1)]].values).long()
#     X_mut_rt = torch.from_numpy(proc_seq_mut_df[[f'RT{i}' for  i in range(st_mut_colindx, end_mut_colindx+1)]].values).long()

#     # x_mut_len = (X_mut_nucl != 4).sum(axis=1).long()
#     x_mut_len  = (end_mut_seqs - start_mut_seqs)

#     # use blank indicator when nucleotide is N (i.e. blank token)
#     ntoken_cond = (X_mut_nucl == blank_nucl_token)
#     X_mut_pbs[ntoken_cond] = blank_annot_token
#     X_mut_rt[ntoken_cond] = blank_annot_token
#     print('X_mut_nucl.unique():', X_mut_nucl.unique())
#     print('ntoken_cond:', ntoken_cond.unique())
#     print('X_mut_pbs:', X_mut_pbs.unique())
#     print('X_mut_rt:', X_mut_rt.unique())

#     # harmonize the size of matrices above
#     max_num_cols = np.max([end_init_colindx+1, end_mut_colindx+1])
#     # print('max_num_cols:', max_num_cols)
#     if max_num_cols > (end_init_colindx+1):
#         num_cols_toadd = max_num_cols - (end_init_colindx + 1)
#         bsize = X_init_nucl.shape[0]
#         ext_mat_shape = (bsize, num_cols_toadd)
#         # mut_cols > init_cols
#         X_init_nucl = extend_matrix(X_init_nucl, ext_mat_shape, 4)
#         X_init_proto = extend_matrix(X_init_proto, ext_mat_shape, 0)
#         X_init_pbs = extend_matrix(X_init_pbs, ext_mat_shape, 0)
#         X_init_rt = extend_matrix(X_init_rt, ext_mat_shape, 0)

#     elif max_num_cols > (end_mut_colindx+1):
#         num_cols_toadd = max_num_cols - (end_mut_colindx +1)
#         bsize = X_mut_nucl.shape[0]
#         ext_mat_shape = (bsize, num_cols_toadd)
#         # init_cols > mut_cols
#         X_mut_nucl = extend_matrix(X_mut_nucl, ext_mat_shape, 4)
#         X_mut_pbs = extend_matrix(X_mut_pbs, ext_mat_shape, 0)
#         X_mut_rt = extend_matrix(X_mut_rt, ext_mat_shape, 0)

#     seq_ids = data_df['seq_id'].values
#     indx_seqid_map = {i:seq_ids[i] for i in range(len(seq_ids))}

#     if len(y_ref):
#         ycols = [tcol for tcol in y_ref if tcol in data_df]
#         print('ycols:', ycols)
#         y_score = torch.from_numpy(data_df[ycols].values)
#     else:
#         y_score = None

#     seqfeat_cols = [cont_cols[0]] + ['Correction_Deletion', 'Correction_Insertion', 'Correction_Replacement'] + cont_cols[1:]

#     # case of haiving indicator variables when melting temperature cannot be computed
#     for colname in ('original_base_mt_nan', 'edited_base_mt_nan'):
#         if colname in data_df:
#             seqfeat_cols.append(colname)

#     seqlevel_feat = torch.from_numpy(data_df[seqfeat_cols].values)

#     dtensor = PEDataTensor(X_init_nucl, 
#                             X_init_proto, 
#                             X_init_pbs,
#                             X_init_rt,
#                             X_mut_nucl,
#                             X_mut_pbs,
#                             X_mut_rt,
#                             seqlevel_feat,
#                             seqfeat_cols,
#                             y_score, 
#                             x_init_len,
#                             x_mut_len,
#                             indx_seqid_map)
        
#     return dtensor