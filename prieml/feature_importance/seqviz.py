import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
from .seqbaseline_generation import normalize_ig_contributions_total, apply_quality_check_convgscores
from ..model import MaskGenerator

class DNASeqViz:
    # adapted from https://github.com/kundajelab/deeplift/blob/master/deeplift/visualization/viz_sequence.py
    def __init__(self):
        self.nucl_num_map = {'A':0, 'C':1, 'T':2, 'G':3}
        
        self.nucl_color_map = {0:'red',
                               1:'yellow',
                               2:'blue',
                               3:'green'}   
        self.html_colors = {'blue':'#aed6f1',
                            'red':'#f5b7b1',
                            'green':'#a3e4d7',
                            'yellow':'#f9e79f',
                            'violet':'#d7bde2'}

        self.plot_funcs_map = {0:self.plot_a, 1:self.plot_c, 2:self.plot_t, 3:self.plot_g}

    def plot_a(self, ax, base, left_edge, height, color):
        a_polygon_coords = [
            np.array([
               [0.0, 0.0],
               [0.5, 1.0],
               [0.5, 0.8],
               [0.2, 0.0],
            ]),
            np.array([
               [1.0, 0.0],
               [0.5, 1.0],
               [0.5, 0.8],
               [0.8, 0.0],
            ]),
            np.array([
               [0.225, 0.45],
               [0.775, 0.45],
               [0.85, 0.3],
               [0.15, 0.3],
            ])
        ]
        for polygon_coords in a_polygon_coords:
            ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                     + np.array([left_edge,base])[None,:]),
                                                    facecolor=color, edgecolor=color))


    def plot_c(self, ax, base, left_edge, height, color):
        ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                                facecolor=color, edgecolor=color))
        ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                                facecolor='white', edgecolor='white'))
        ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                                facecolor='white', edgecolor='white', fill=True))


    def plot_g(self, ax, base, left_edge, height, color):
        ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                                facecolor=color, edgecolor=color))
        ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                                facecolor='white', edgecolor='white'))
        ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                                facecolor='white', edgecolor='white', fill=True))
        ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
                                                facecolor=color, edgecolor=color, fill=True))
        ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
                                                facecolor=color, edgecolor=color, fill=True))


    def plot_t(self, ax, base, left_edge, height, color):
        ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
                      width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
        ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],
                      width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))

#     default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
#     default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}
    
    def _plot_weights_given_ax(self, 
                              ax, 
                              array,
                              height_padding_factor,
                              length_padding,
                              subticks_frequency,
                              highlight):
        
        """
        Args:
            ax: matplotlib axis
            array: numpy matrix, (L, 4) where L is sequence length and 4 is the one-hot-encoding of the nucleotides
            height_padding_factor: float
            length_padding:float
            subticks_frequency: float
            highlight: dict of {color:[(x0, x3)]}, color => list of coordinates to draw rectangles around 
        
        
        """
        
        nucl_color_map = self.nucl_color_map
        html_colors = self.html_colors
        plot_funcs = self.plot_funcs_map
        
        if len(array.shape)==3:
            array = np.squeeze(array)
        assert len(array.shape)==2, array.shape

        max_pos_height = 0.0
        min_neg_height = 0.0
        heights_at_positions = []
        depths_at_positions = []
        for i in range(array.shape[0]):
            #sort from smallest to highest magnitude
            actg_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
#             print('acgt_vals:', acgt_vals)
            positive_height_so_far = 0.0
            negative_height_so_far = 0.0
            for letter in actg_vals:
#                 print('letter:', letter)
                l, sval = letter # (letter index, score)
                plot_func = plot_funcs[l]
                color= matplotlib.colors.to_rgb(html_colors[nucl_color_map[l]])
                if (sval > 0):
                    height_so_far = positive_height_so_far
                    positive_height_so_far += sval                
                else:
                    height_so_far = negative_height_so_far
                    negative_height_so_far += sval
                    
                plot_func(ax=ax, base=height_so_far, left_edge=i, height=sval, color=color)
            max_pos_height = max(max_pos_height, positive_height_so_far)
            min_neg_height = min(min_neg_height, negative_height_so_far)
            heights_at_positions.append(positive_height_so_far)
            depths_at_positions.append(negative_height_so_far)

        #now highlight any desired positions; the key of
        #the highlight dict should be the color
        for color in highlight:
            for start_pos, end_pos in highlight[color]:
                assert start_pos >= 0.0 and end_pos <= array.shape[0]
                min_depth = np.min(depths_at_positions[start_pos:end_pos])
                max_height = np.max(heights_at_positions[start_pos:end_pos])
                ax.add_patch(
                    matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
                        width=end_pos-start_pos,
                        height=max_height-min_depth,
                        edgecolor=color, fill=False))

        ax.set_xlim(-length_padding, array.shape[0]+length_padding)
        ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))
        ax.xaxis.set_tick_params(rotation=90)
        height_padding = max(abs(min_neg_height)*(height_padding_factor),
                             abs(max_pos_height)*(height_padding_factor))
        ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)


    def plot_weights(self,
                     array,
                     ax,
                     height_padding_factor=0.2,
                     length_padding=1.0,
                     subticks_frequency=1.0,
                     highlight={}):
        
        """
        Args:
            array: attribution scores, numpy matrix, (L, 4) where L is sequence length and 4 is the one-hot-encoding of the nucleotides
            figsize: matplotlib figure size, tuple
            height_padding_factor: float
            length_padding:float
            subticks_frequency: float
            highlight: dict of {color:[(x0, x3)]}, color => list of coordinates to draw rectangles around 
        
        
        """

        self._plot_weights_given_ax(ax=ax, 
                                    array=array,
                                    height_padding_factor=height_padding_factor,
                                    length_padding=length_padding,
                                    subticks_frequency=subticks_frequency,
                                    highlight=highlight)
        # plt.show()


# def plot_ig_from_sample(seq_id,
#                         seqs_ids_lst,
#                         df_baseline,
#                         ig_init_contrib_lst,
#                         ig_mut_contrib_lst, 
#                         ig_seqfeat_contrib_lst,
#                         convg_scores_lst,
#                         pred_score,
#                         seqlevel_feat_colnames,
#                         fig_dir=None,
#                         normalize_total=True):

    
# #     fig, ax = plt.subplots(figsize=(20,5), 
# #                            nrows=2, 
# #                            constrained_layout=False)
    

#     fig = plt.figure(figsize=(25,7), constrained_layout=False)

#     gs = GridSpec(3,3, figure=fig, hspace=0.6)
#     ax1 = fig.add_subplot(gs[0, :])
#     ax2 = fig.add_subplot(gs[1, :])
#     ax3 = fig.add_subplot(gs[2, 0:2])
#     ax4 = fig.add_subplot(gs[2, 2])
#     ax = [ax1,ax2,ax3]


#     highlight_colors_map = {'PS':'#102542',#oxford blue,
#                             'PBS':'#A480CF',#light gray
#                             'RT':'#B3A394'#grullo
#                            }
                            
#     seqid_indx_map = {elm:i for i, elm in enumerate(seqs_ids_lst)}
#     df = df_baseline.loc[df_baseline['seq_id'] == seq_id].copy()
#     original_seqid = df['seqid'].values[0]
#     tindx = seqid_indx_map[seq_id]
    
#     mat_concat = np.concatenate([ig_init_contrib_lst[tindx], 
#                     ig_mut_contrib_lst[tindx], 
#                     ig_seqfeat_contrib_lst[tindx]])
#     max_val = np.max(np.abs(mat_concat))
#     min_val = np.min(np.abs(mat_concat))

#     if normalize_total:
#         ig_init_contrib_norm = normalize_ig_contributions_total(ig_init_contrib_lst[tindx],min_val, max_val)
#         ig_init_share = ig_init_contrib_norm.sum()
#     else:
#         ig_init_contrib_norm = normalize_ig_contributions(ig_init_contrib_lst[tindx])
#     init_contrib_dim = ig_init_contrib_lst[tindx].shape[0]
    
    
#     r = df['RT_initial_location'].str.strip('[]').str.split(',')
#     rt_color = highlight_colors_map['RT']
    
#     ps = df['protospacerlocation_only_initial'].str.strip('[]').str.split(',')
#     ps_color = highlight_colors_map['PS']
    
#     pbs = df['PBSlocation'].str.strip('[]').str.split(',')
#     pbs_color = highlight_colors_map['PBS']
#     highlight = {}
#     for elm_color, elm in [(rt_color,r),
#                 (ps_color,ps),
#                 (pbs_color,pbs)]:
#         highlight[elm_color] = [(int(elm.str[0].values[0]), int(elm.str[1].values[0]))]
        
#     t = seq_to_one_hot(df['wide_initial_target'].values[0][:init_contrib_dim])*ig_init_contrib_norm.reshape(-1,1)
#     DNASeqViz().plot_weights(t, ax[0], highlight=highlight)
#     ax[0].set_ylim([0., 1.])
    
#     if normalize_total:
#         ig_mut_contrib_norm = normalize_ig_contributions_total(ig_mut_contrib_lst[tindx],min_val, max_val)
#         ig_mut_share = ig_mut_contrib_norm.sum()
#     else:
#         ig_mut_contrib_norm = normalize_ig_contributions(ig_mut_contrib_lst[tindx])
#     mut_contrib_dim = ig_mut_contrib_lst[tindx].shape[0]
#     r = df['RT_mutated_location'].str.strip('[]').str.split(',')
#     rt_color = highlight_colors_map['RT']
#     highlight = {}
#     for elm_color, elm in [(rt_color,r),
#                            (pbs_color,pbs)]:
#         highlight[elm_color] = [(int(elm.str[0].values[0]), int(elm.str[1].values[0]))]
    
#     t = seq_to_one_hot(df['wide_mutated_target'].values[0][:mut_contrib_dim])*ig_mut_contrib_norm.reshape(-1,1)
#     DNASeqViz().plot_weights(t, ax[1], highlight=highlight)
#     ax[1].set_ylim([0., 1.])
    
# #     if fig_dir:
# #         fig.savefig(os.path.join(fig_dir,f'ig_contribution_initmut_seqs_{original_seqid}.pdf'))
        
#     if normalize_total:
#         ig_seqfeat_contrib_norm = normalize_ig_contributions_total(ig_seqfeat_contrib_lst[tindx],min_val, max_val)
#         ig_seqfeat_share = ig_seqfeat_contrib_norm.sum()
#     else:
#         ig_seqfeat_contrib_norm = normalize_ig_contributions(ig_seqfeat_contrib_lst[tindx])
        
# #     fig = plt.figure(figsize=(11,2), constrained_layout=False)

# #     gs = GridSpec(1, 2, figure=fig)
# #     ax2 = fig.add_subplot(gs[0, 0])
# #     ax3 = fig.add_subplot(gs[0, 1])

# #     fig, ax2 = plt.subplots(figsize=(11,2), 
# #                    ncols=2, 
# #                    constrained_layout=False)
#     cbar_kws={'label': 'IG score', 'orientation': 'vertical'}
#     cmap = 'YlOrRd'
#     g = sns.heatmap(ig_seqfeat_contrib_norm.reshape(1,-1), 
#                     cmap=cmap,
#                     fmt="",
#                     annot=False,
#                     linewidths=.5, cbar_kws=cbar_kws, 
#                     ax=ax3)
    
#     ax3.set_xticklabels(seqlevel_feat_colnames)
#     ax3.xaxis.set_tick_params(rotation=90)
#     ax3.set_yticklabels([''])
    
#     total_share = np.sum([ig_init_share, ig_mut_share, ig_seqfeat_share])
#     dist = np.array([ig_init_share, ig_mut_share, ig_seqfeat_share])/total_share
#     convg_score = convg_scores_lst[tindx][0]
#     edit_eff = pred_score[tindx]*100

#     ax4.bar(['initial seq.','mutated seq.','seqlevel features'], dist)
#     ax4.set_ylabel('% of total IG scores')
#     ax4.text(-0.5, -0.4, f'IG convergence score:{convg_score:.5f}', fontsize=12)
#     ax4.text(-0.5, -0.6, f'seq_id: {original_seqid}', fontsize=12)
#     ax4.text(-0.5, -0.8, f'Edit efficiency: {edit_eff:.2f}%', fontsize=12)
    
#     if fig_dir:
#         fig.savefig(os.path.join(fig_dir,f'ig_contribution_{original_seqid}.pdf'),bbox_inches='tight')
#         plt.close()


def get_topk(arr, topk=10):
    assert len(arr.shape) == 1, 'one dimenesional array is required'
    if topk > arr.shape[0]:
        topk = arr.shape[0] # topk will be equal to the whole array length
    idx = np.argpartition(arr, -topk)[-topk:]
#     print('idx:', idx)
#     print(arr[idx])
    # sorted indices with topk
    idx = idx[np.argsort(arr[idx])][::-1]
    # print('idx:', idx)
    # print(arr[idx])
    return idx
    
def compute_ig_from_sample():
    pass
def compute_overall_ig_from_samples(convg_scores, 
                                    ig_init_contrib_lst, 
                                    ig_mut_contrib_lst, 
                                    ig_seqfeat_contrib_lst,
                                    init_len,
                                    mut_len,
                                    normalize=True,
                                    convg_threshold=5e-2):
    # condition based on the convergence score from computing IG
    tindices, quality_score = apply_quality_check_convgscores(convg_scores, threshold=convg_threshold)
    num_samples = len(tindices)

    # (samples, max_seqlen)
    ig_init = np.abs(np.concatenate([ig_init_contrib_lst[indx].reshape(1,-1) for indx in tindices],axis=0))
    ig_mut = np.abs(np.concatenate([ig_mut_contrib_lst[indx].reshape(1,-1) for indx in tindices],axis=0))
    # (samples, seqlevel_featdim)
    ig_seqfeat = np.abs(np.concatenate([ig_seqfeat_contrib_lst[indx].reshape(1,-1) for indx in tindices],axis=0))
   
    # print('ig_init:', ig_init.shape, ig_init.min(), ig_init.max())
    # print('ig_mut:', ig_mut.shape, ig_mut.min(), ig_mut.max())
    # print('ig_seqfeat:', ig_seqfeat.shape, ig_seqfeat.min(), ig_seqfeat.max())

    init_len_filtered = [init_len[indx] for indx in tindices]
    mut_len_filtered = [mut_len[indx] for indx in tindices]

    assert ig_init_contrib_lst[0].shape[-1] == ig_mut_contrib_lst[0].shape[-1], "By design, the init and mut sequence should have same tensor size!"

    max_seqlen = ig_init_contrib_lst[0].shape[-1]

    # construct a matrix of 1 at all positions until each sequence length
    # use it as a counter for the number of sequences have elements at each position
    # it will be used to reweight the contributions at each position 
    init_len_mask = MaskGenerator().create_content_mask((num_samples, max_seqlen), init_len_filtered)
    init_len_mask = init_len_mask.numpy()
    mut_len_mask = MaskGenerator().create_content_mask((num_samples, max_seqlen), mut_len_filtered)
    mut_len_mask = mut_len_mask.numpy()

    # print('ig_init.sum(axis=0):', ig_init.sum(axis=0))
    ig_init_mean = ig_init.sum(axis=0) / np.clip(init_len_mask.sum(axis=0), a_min=1, a_max=np.inf)
    # print('ig_init_mean:', ig_init_mean)
    ig_mut_mean = ig_mut.sum(axis=0) / np.clip(mut_len_mask.sum(axis=0), a_min=1, a_max=np.inf)
    # print('ig_mut.sum(axis=0):', ig_mut.sum(axis=0))
    # print('ig_mut_mean:', ig_mut_mean)
    # TODO: add num samples contribution as above since not all samples have equal Correction Type!
    ig_seqfeat_mean = ig_seqfeat.mean(axis=0)
    
    # print(ig_init_mean)
    # print(ig_mut_mean)
    # print(ig_seqfeat_mean)
    # print('ig_init_mean:', ig_init_mean.shape, ig_init_mean.min(), ig_init_mean.max())
    # print('ig_mut_mean:', ig_mut_mean.shape, ig_mut_mean.min(), ig_mut_mean.max())
    # print('ig_seqfeat_mean:', ig_seqfeat_mean.shape, ig_seqfeat_mean.min(), ig_seqfeat_mean.max())
    
    if normalize:
        mat_concat = np.concatenate([ig_init_mean, ig_mut_mean, ig_seqfeat_mean])
        max_val = np.max(mat_concat)
        min_val = np.min(mat_concat)

        # print('total min_val:', min_val, 'total max_val:', max_val)

        ig_init_mean_norm = normalize_ig_contributions_total(ig_init_mean, min_val, max_val)
        ig_mut_mean_norm = normalize_ig_contributions_total(ig_mut_mean, min_val, max_val)
        ig_seqfeat_mean_norm = normalize_ig_contributions_total(ig_seqfeat_mean, min_val, max_val)
    else:
        ig_init_mean_norm = ig_init_mean
        ig_mut_mean_norm = ig_mut_mean
        ig_seqfeat_mean_norm = ig_seqfeat_mean 
    
    # print('ig_init_mean_norm:', ig_init_mean_norm.shape, ig_init_mean_norm.min(), ig_init_mean_norm.max())
    # print('ig_mut_mean_norm:', ig_mut_mean_norm.shape, ig_mut_mean_norm.min(), ig_mut_mean_norm.max())
    # print('ig_seqfeat_mean_norm:', ig_seqfeat_mean_norm.shape, ig_seqfeat_mean_norm.min(), ig_seqfeat_mean_norm.max())

    ig_lst = [ig_init_mean_norm, ig_mut_mean_norm, ig_seqfeat_mean_norm]

    return ig_lst, quality_score

def plot_overall_ig_from_samples(ig_lst, seqlevelfeat_names, fig_dir=None, fname=None):
    
    fig = plt.figure(figsize=(20,5), constrained_layout=False)

    gs = GridSpec(3,1, figure=fig, hspace=0.6)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    if len(ig_lst) == 3:
        ax3 = fig.add_subplot(gs[2, :])
        ax = [ax1,ax2,ax3]
        feat_type_names = ['initial_target', 'wide_mutated_target', 'seqlevel_feat']
    else:
        ax = [ax1,ax2]
        feat_type_names = ['initial_mutated_target', 'seqlevel_feat']
#     fig, ax = plt.subplots(figsize=(20,5), 
#                    nrows=3, 
#                    constrained_layout=False)
    
    cbar_kws={'label': 'IG score', 'orientation': 'vertical'}
    cmap = 'YlOrRd'
    for i in range(len(ax)):
        arr = ig_lst[i].reshape(1,-1)
        max_val = np.max(arr)
        min_val = np.min(arr)
        g = sns.heatmap(arr, 
                        cmap=cmap,
                        fmt="",
                        annot=False,
                        linewidths=.5, cbar_kws=cbar_kws, 
                        ax=ax[i],
                       vmin=min_val,
                       vmax=max_val)
        if i == len(ax) - 1:
            ax[i].set_xticklabels(seqlevelfeat_names)
            ax[i].xaxis.set_tick_params(rotation=90)
        else:
            ax[i].xaxis.set_ticks(np.arange(0.0, arr.shape[1], 1.))
            ax[i].set_xticklabels([i for i in range(arr.shape[1])])
            ax[i].xaxis.set_tick_params(rotation=90)

        ax[i].set_yticklabels([feat_type_names[i]])
        ax[i].yaxis.set_tick_params(rotation=0)

    if fig_dir is not None:
        if fname is not None:
            fig.savefig(os.path.join(fig_dir,f'{fname}.pdf'),bbox_inches='tight')
        else:
            fig.savefig(os.path.join(fig_dir,'overall_ig_contributions_from_samples.pdf'),bbox_inches='tight')
