import pandas as pd
import numpy as np
from ..dataset import get_seqlevel_featnames

class SeqBaselineGenerator:
    def __init__(self):
        pass
    @classmethod
    def generate_df_from_fixed_baselines(clss, df, nsamples, feat_subs_opt='theoretical_min_val'):
        
        tindices = df.index[:nsamples]
        df = df.loc[tindices].copy()
        
        #### overview of sequence level features and corresponding max normalizer #####
        ####
        #         Correction_Length                     0*50.
        #         Correction_Deletion                   0
        #         Correction_Insertion                  0
        #         Correction_Replacement                0
        #         RToverhangmatches                     0*50.

        #         RToverhanglength                      0*50.
        #         RTlength                              0*50.
        #         PBSlength                             0*50.
        #         MFE_protospacer                      -1*120.
        #         MFE_protospacer_scaffold             -1*120.

        #         MFE_extension                        -1*120.
        #         MFE_extension_scaffold               -1*120.
        #         MFE_protospacer_extension_scaffold   -1*120.
        #         MFE_rt                               -1*120.
        #         MFE_pbs                              -1*120.

        #         RTmt                                  0*200.
        #         RToverhangmt                          0*200.
        #         PBSmt                                 0*200.
        #         protospacermt                         0*200.
        #         extensionmt                           0*200.

        #         original_base_mt                      0*200.
        #         edited_base_mt                        0*200.
        #         original_base_mt_nan                  0
        #         edited_base_mt_nan                    0

        # some minimal processing
        if 'correction_type_categ' not in df:
            print('--- creating correction type categories ---')
            correction_categs = ['Deletion', 
                                'Insertion', 
                                'Replacement']
            df['correction_type_categ'] = pd.Categorical(df['Correction_Type'], categories=correction_categs)
            correction_type_df = pd.get_dummies(df['correction_type_categ'], prefix='Correction', prefix_sep='_')
            df = pd.concat([df, correction_type_df], axis=1)
        

        if feat_subs_opt == 'theoretical_min_val':
            feat_val_arr = [0*50.   , 0.      , 0.      , 0.      ,    0*50., 
                            0*50.   , 0*50.   , 0*50.   , -1.*120., -1.*120.,
                            -1.*120., -1.*120., -1.*120., -1.*120., -1.*120.,
                            0.*200. , 0.*200. , 0.*200. , 0.*200. ,  0.*200.,
                            0.*200. , 0.*200. , 0.      , 0.]
        # based on experimental data distribution 
        elif feat_subs_opt == 'min_val':
            feat_val_arr = [ 0.02*50.       ,  0.             ,  0.             ,  0.             ,  0.02*50.       ,
                             0.06*50.       ,  0.08*50.       ,  0.26*50.       , -0.11583333*120., -0.44416666*120.,
                            -0.22166667*120., -0.52583332*120., -0.74583333*120., -0.18999999*120., -0.055*120.     ,
                             0.04*200.      ,  0.03*200.      ,  0.13*200.      ,  0.22*200.      ,  0.18*200.      ,
                             0.*200.        ,  0.*200.        ,  0.             ,  0.        ]
        elif feat_subs_opt == 'mean_val':
            feat_val_arr = [ 0.02876886*50.  ,  0.06626062     ,  0.30763395     ,  0.62610543     ,  0.02217379*50. ,
                             0.17397369*50.  ,  0.33210595*50. ,  0.26*50.       , -0.01507402*120 , -0.30927307*120 ,
                            -0.03474034*120  , -0.33915078*120 , -0.53745106*120 , -0.01018871*120 , -0.00220591*120 ,
                             0.2589881*200.  ,  0.13501203*200.,  0.19645158*200.,  0.30725496*200.,  0.45543968*200.,
                             0.01088532*200. ,  0.02019323*200.,  0.30763395     ,  0.06626062]
        if 'edited_base_mt_nan' not in df:
            seqfeat_stopindx = -2
            feat_val_arr = feat_val_arr[:-2] # case of not having indicator variables when melting temperature cannot be computed
        else:
            seqfeat_stopindx = len(feat_val_arr)

                    
        seqlen = max([df['wide_initial_target'].str.len().max(),
                      df['wide_mutated_target'].str.len().max()])
        
        # get a sequence to create a baseline
        selecindx = tindices[0]
        row = df.loc[selecindx].copy()

        row['wide_initial_target'] = 'N'*seqlen
        row['wide_mutated_target'] = 'N'*seqlen

        # substitute the values of continuous features
        seqfeat_colnames = get_seqlevel_featnames(suffix=None)
        seqfeat_colnames = seqfeat_colnames[:seqfeat_stopindx]
        row[seqfeat_colnames] = feat_val_arr

        row['Name'] = 'seq_fixedbaseline_theormin'
        row['Editing_Position'] = -1.
        row['seq_id'] = 'seq_fixedbaseline_0'
        # this creates a NaN issue because the type should be category
        # for now we can ignore it as it does not have any influence 
        # row['correction_type_categ'] = 'NotApp' 
        row['Correction_Type'] = 'NotApp'
        # comment this out in case of issues -- legacy support 😬
        # if 'Editing_Position_Ahmed' in row:
        #     row['Editing_Position_Ahmed'] = -1
            
        row_df = pd.DataFrame(row.values.reshape(1,-1), columns=row.index)
        row_df = row_df.astype(df.dtypes.to_dict())
        upd_df = pd.concat([df, row_df], axis=0)
        return upd_df

# utils
def seq_to_one_hot(seq):
    one_hot = np.zeros((len(seq), 4))
    nucl_indx_map = {'A':0, 'C':1, 'T':2, 'G':3, 'N':4}
    for i, elm in enumerate(seq):
        one_hot[i, nucl_indx_map[elm]] = 1
    return one_hot

def normalize_ig_contributions(ig_matrix):
    ig_matrix_abs = np.abs(ig_matrix)
    max_val = np.max(ig_matrix_abs)
    min_val = np.min(ig_matrix_abs)
    return (ig_matrix_abs - min_val)/(max_val - min_val)

def normalize_ig_contributions_total(ig_matrix,min_val, max_val):
    assert min_val >= 0 
    assert max_val > 0
    ig_matrix_abs = np.abs(ig_matrix)
    return (ig_matrix_abs - min_val)/(max_val - min_val)

def apply_quality_check_convgscores(convg_scores, threshold=1e-3):
    scores_lst = []
    for i in range(len(convg_scores)):
        scores_lst.append(convg_scores[i][0])
    scores_arr = np.array(scores_lst)
    tindices = np.where(scores_arr <= threshold)[0]
    quality_score = 100*len(tindices)/len(scores_arr)
    print(f'% of usable samples with convg. score < {threshold}: ', quality_score)
    return tindices, quality_score