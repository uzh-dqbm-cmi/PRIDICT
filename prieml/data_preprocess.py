import re
import os
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

def get_char(seq):
    """split string int sequence of chars returned in pandas.Series"""
    chars = list(seq)
    return pd.Series(chars)

def is_initial_eq_mutated(df):
    return 100*(df['wide_initial_target'] == df['wide_mutated_target']).sum()/df.shape[0]
    
class PESeqProcessor:
    def __init__(self):
        pass
    def process_perbase_df(self, df, target_cols, seq_colname):
        """cleans a data frame representing sequences and their edit info obtained from crispr experiment
        
        Args:
            df: pandas.DataFrame (dataset dataframe)
            target_cols: list of column names we need to keep
            seq_colname: string, sequence id column
                    
        """
        pbar = tqdm(total=4)

        target_cols += [seq_colname]
        df = df[target_cols].copy()
        # harmonize sequence string representation to capitalized form
        df[seq_colname] = df[seq_colname].str.upper()
        pbar.update(1)

        baseseq_df = df[seq_colname].apply(get_char)
        num_cols = baseseq_df.shape[1]
        baseseq_df.columns = [f'B{i}' for  i in range(0, num_cols)]
        pbar.update(2)

        baseseq_letters_df = baseseq_df.copy()
        baseseq_letters_df.columns = [f'L{i}' for  i in range(0, num_cols)]
        # replace base letters with numbers
        baseseq_df.replace(['A', 'C', 'T', 'G'], [0,1,2,3], inplace=True)
        
        # replace Na in case of unequal length sequences
        baseseq_df = baseseq_df.fillna(4)
        baseseq_letters_df = baseseq_letters_df.fillna('N')
        pbar.update(3)

        base_df = pd.concat([df[target_cols],
                            baseseq_letters_df,
                            baseseq_df], axis=1)
        base_df.reset_index(inplace=True, drop=True)
        pbar.update(4)
        pbar.close()
        return base_df, num_cols
    
    def process_init_mut_seqs(self, df, target_cols, init_seq_colname, mut_seq_colname):
        """
        
        Args:
            df: pandas.DataFrame (dataset dataframe)
            target_cols: list of column names we need to keep
            init_seq_colname: string, column name of initial target sequence
            mut_seq_colname: string, column name of mutated target sequence

        """
        pbar = tqdm(total=4)
        proc_init_df, num_init_cols = self.process_perbase_df(df, target_cols, init_seq_colname)
        pbar.update(1)
        proc_seq_init_df = self.add_seq_annotations(proc_init_df, df, num_init_cols, 'initial')
        pbar.update(2)
        proc_mut_df, num_mut_cols = self.process_perbase_df(df, target_cols, mut_seq_colname)
        pbar.update(3)
        proc_seq_mut_df = self.add_seq_annotations(proc_mut_df, df, num_mut_cols, 'mutated')
        pbar.update(4)
        pbar.close()
        self.validate_df(proc_seq_init_df)
        self.validate_df(proc_seq_mut_df)  

        return proc_seq_init_df, num_init_cols,  proc_seq_mut_df, num_mut_cols

    def add_seq_annotations(self, proc_seq_df, data_df, num_cols, seqtype):
        """
        
        Args:
            proc_seq_df: pandas.DataFrame returned from :func:`self.process_perbase_df`
            data_df: pandas.DataFrame (dataset dataframe)
            num_cols: int, number of columns from identified nucleotides
            seq_type: string, in {initial, mutated}
        """
        if seqtype == 'initial':
            var_arr_map = {'protospacerlocation_only_initial':None,
                           'PBSlocation':None,
                           'RT_initial_location':None}
        elif seqtype == 'mutated':
            var_arr_map = {'PBSlocation':None,
                           'RT_mutated_location':None}
            
        col_name_map = {'protospacerlocation_only_initial': 'Protos',
                        'PBSlocation': 'PBS',
                        'RT_initial_location':'RT',
                        'RT_mutated_location':'RT'}
        proc_df = proc_seq_df.copy() # keep copy without mutating the original one
        num_seqs = data_df.shape[0]
        start_seq = None
        end_seq = None
        for colname in var_arr_map:
            print('processing:', colname)
            range_df = data_df[colname].str.strip('[]').str.split(',')
            start_pos = range_df.str[0].values.astype(int)
            stop_pos =  range_df.str[1].values.astype(int)
            if colname == 'protospacerlocation_only_initial':
                start_seq = start_pos.copy()
            if colname in {'RT_initial_location', 'RT_mutated_location'}:
                end_seq = stop_pos.copy()
                
            arr = np.full((num_seqs, num_cols), 0)
            for i in tqdm(range(num_seqs)):
                r = start_pos[i]
                c = stop_pos[i]
                arr[i, r:c] = 1
            arr_df = pd.DataFrame(arr)
            arr_df.columns = [f'{col_name_map[colname]}{i}' for i in range(0, num_cols)]
            proc_df = pd.concat([proc_df, arr_df], axis=1)
            proc_df[f'start_{col_name_map[colname]}'] = start_pos
            proc_df[f'end_{col_name_map[colname]}'] = stop_pos
        # add start & end of a seq
        if start_seq is not None:
            proc_df['start_seq'] = start_seq
        if end_seq is not None:
            proc_df['end_seq'] = end_seq
        return proc_df
    
    def validate_df(self, df):
        assert df.isna().any().sum() == 0

class Viz_PESeqs:

    html_colors = {'blue':' #aed6f1',
                   'red':' #f5b7b1',
                   'green':' #a3e4d7',
                   'yellow':' #f9e79f',
                   'violet':'#d7bde2',
                   'brown_sugar':'#B47955',
                   'tan':'#DBB68F',
                   'tangerine':'#F19455'}
    
    codes = {'A':'@', 'C':'!', 'T':'#', 'G':'_', 'init':'~', 'mut':'%', 'ewindow':'§', 'prob':'`'}
    
    nucl_colrmap = {'A':'red',
                    'C':'yellow',
                    'T':'blue',
                    'G':'green',
                    'init':'violet',
                    'mut':'brown_sugar',
                    'ewindow':'tan',
                    'prob':'tangerine'}
    
    def __init__(self):
        pass

    # @classmethod
    # def viz_align_initmut_seq_tensor(clss,pe_dtensor, data_df, seqid, window=0, return_type='html'):
    #     """
    #     Args:
    #         pe_dtensor: instance of :class:`PEDatatensor`
    #         data_df: dataset df 
    #         seqid: string, sequence id
    #         return_type: string, default `html`
    #     """
        
    #     codes = clss.codes
        
    #     seqid_colname = 'seq_id'
    #     c_seq_datadf = data_df.loc[data_df[seqid_colname] == seqid]
    #     c_seq_procinit = proc_seq_init_df.loc[proc_seq_init_df[seqid_colname] == seqid]
    #     c_seq_procmut = proc_seq_mut_df.loc[proc_seq_mut_df[seqid_colname] == seqid]
        
    #     end_init_seq = c_seq_procinit['end_seq'].values[0]
    #     end_mut_seq = c_seq_procmut['end_seq'].values[0]
    #     end_annot_seq = np.max([end_init_seq, end_mut_seq])

    #     st_init_seq = c_seq_procinit['start_seq'].values[0]

    #     if 'start_seq' in c_seq_procmut:
    #         st_mut_seq = c_seq_procmut['start_seq'].values[0]
    #     else:
    #         st_mut_seq = st_init_seq
    
    #     st_annot_seq =  np.min([st_init_seq, st_mut_seq])
        
    #     lower_thr = 0
    #     upper_thr = 99
        
    #     upper_init_thr = np.min([st_annot_seq+upper_thr, num_cols, end_annot_seq + window])
    #     end_seq = np.clip(end_annot_seq + window, a_min=lower_thr, a_max=upper_init_thr)
    #     st_seq = np.clip(st_annot_seq - window, a_min=lower_thr, a_max=upper_thr)

    #     # seq_name = c_seq_datadf['Name'].values[0]
        
    #     tb = PrettyTable()
    #     tb.field_names = [f'Seq. ID:\n{seqid}'] + [f'{i}' for i in range(st_seq, end_seq)]
        
    #     # initial sequence information
    #     # Protos
    #     # PBS
    #     # RT
    #     # wide_initial_target
    #     if 'y' in c_seq_datadf:
    #         score = c_seq_datadf['y'].values[0]
    #     else:
    #         score = None

    #     correction_type = c_seq_datadf['Correction_Type'].values[0]
    #     correction_len = c_seq_datadf['Correction_Length'].values[0]
    #     st_editpos = c_seq_datadf['Editing_Position'].values[0]

    #     end_init_seq = end_seq
    #     offset_fromst = 0
    #     ewindow_st = offset_fromst + st_editpos
    #     ewindow_end = end_init_seq
    #     # if correction_type in {'Insertion'}:
    #     #     ewindow_st = offset_fromst + st_editpos
    #     #     ewindow_end = end_init_seq
    #     # else:
    #     #     ewindow_st = offset_fromst + st_editpos 
    #     #     ewindow_end = end_init_seq

    #     # ewindow_st = st_editpos
    #     # ewindow_end = end_init_seq 

    #     ewindow_str_lst = [f'Editing: {correction_type}'] + \
    #                   ['' for elm in range(st_seq, ewindow_st)]+ \
    #                   [f"{codes['ewindow']}*" for elm in range(ewindow_st, ewindow_st + correction_len)] + \
    #                   ['' for elm in range(ewindow_st + correction_len, ewindow_end)]
            
    #     n_rows = data_df.shape[0]
    #     # generate outcome (haplotype) rows
    #     for colname in ('Protos', 'PBS', 'RT'):
    #         vals = c_seq_procinit[[f'{colname}{i}' for i in range(st_seq, end_init_seq)]].values[0]
    #         annot_lst = [f'{colname}']
    #         cl_lst = []
    #         for annot in vals:
    #             if annot:
    #                 cl_lst += [f"{codes['init']}*"]
    #             else:
    #                 cl_lst += [' ']
    #         annot_lst += cl_lst
    #         tb.add_row(annot_lst)

    #     if correction_type == 'Deletion':
    #         tb.add_row(ewindow_str_lst)
            
    #     wide_init_target = c_seq_procinit[[f'L{i}' for i in range(st_seq, end_init_seq)]].values[0]
    #     init_target_str_lst = ['Initial sequence'] + [f'{codes[nucl]}{nucl}' for nucl in wide_init_target]
    #     tb.add_row(init_target_str_lst)
        
    #     # mutated sequence information
    #     # wide_mutated_target
    #     # PBS
    #     # RT
        
    #     end_mut_seq = end_seq
    #     wide_mut_target = c_seq_procmut[[f'L{i}' for i in range(st_seq, end_mut_seq)]].values[0]
    #     if score is not None:
    #         mut_target_str_lst = ['{}Mutated sequence\n Prob. edit={:.3f}'.format(codes['prob'], score)] + \
    #                             [f'{codes[nucl]}{nucl}' for nucl in wide_mut_target]
    #     else:
    #         mut_target_str_lst = ['{}Mutated sequence\n'.format(codes['prob'])] + \
    #                              [f'{codes[nucl]}{nucl}' for nucl in wide_mut_target] 
        
    #     tb.add_row(mut_target_str_lst)
        
    #     if correction_type in {'Insertion', 'Replacement'}:
    #         tb.add_row(ewindow_str_lst)
            
    #     for colname in ('PBS', 'RT'):
    #         vals = c_seq_procmut[[f'{colname}{i}' for i in range(st_seq, end_mut_seq)]].values[0]
    #         annot_lst = [f'{colname}']
    #         cl_lst = []
    #         for annot in vals:
    #             if annot:
    #                 cl_lst += [f"{codes['mut']}*"]
    #             else:
    #                 cl_lst += [f' ']
    #         annot_lst += cl_lst
    #         tb.add_row(annot_lst)

    #     # seqid_str_lst = [f'Sequence ID:\n  {seqid}'] + ['' for elm in range(st_seq, end_seq)]
    #     # tb.add_row(seqid_str_lst)


    #     if return_type == 'html':
    #         return clss._format_html_table(tb.get_html_string())
    #     else: # default string
    #         return tb.get_string()
    @classmethod
    def viz_align_initmut_seq(clss, proc_seq_init_df, proc_seq_mut_df, num_cols, data_df, seqid, window=0, return_type='html'):
        """
        Args:
            proc_seq_init_df:
            proc_seq_mut_df:
            data_df: dataset df 
            seqid: string, sequence id
            return_type: string, default `html`
        """
        
        codes = clss.codes
        
        seqid_colname = 'seq_id'
        c_seq_datadf = data_df.loc[data_df[seqid_colname] == seqid]
        c_seq_procinit = proc_seq_init_df.loc[proc_seq_init_df[seqid_colname] == seqid]
        c_seq_procmut = proc_seq_mut_df.loc[proc_seq_mut_df[seqid_colname] == seqid]
        
        end_init_seq = c_seq_procinit['end_seq'].values[0]
        end_mut_seq = c_seq_procmut['end_seq'].values[0]
        end_annot_seq = np.max([end_init_seq, end_mut_seq])

        st_init_seq = c_seq_procinit['start_seq'].values[0]

        if 'start_seq' in c_seq_procmut:
            st_mut_seq = c_seq_procmut['start_seq'].values[0]
        else:
            st_mut_seq = st_init_seq
    
        st_annot_seq =  np.min([st_init_seq, st_mut_seq])
        
        lower_thr = 0
        upper_thr = 99
        
        upper_init_thr = np.min([st_annot_seq+upper_thr, num_cols, end_annot_seq + window])
        end_seq = np.clip(end_annot_seq + window, a_min=lower_thr, a_max=upper_init_thr)
        st_seq = np.clip(st_annot_seq - window, a_min=lower_thr, a_max=upper_thr)

        # seq_name = c_seq_datadf['Name'].values[0]
        
        tb = PrettyTable()
        tb.field_names = [f'Seq. ID:\n{seqid}'] + [f'{i}' for i in range(st_seq, end_seq)]
        
        # initial sequence information
        # Protos
        # PBS
        # RT
        # wide_initial_target
        if 'y' in c_seq_datadf:
            score = c_seq_datadf['y'].values[0]
        else:
            score = None

        correction_type = c_seq_datadf['Correction_Type'].values[0]
        correction_len = int(c_seq_datadf['Correction_Length'].values[0])
        st_editpos = c_seq_datadf['Editing_Position'].values[0]

        end_init_seq = end_seq
        offset_fromst = 0
        ewindow_st = offset_fromst + st_editpos
        ewindow_end = end_init_seq
        print('start_seq:', st_seq)
        print('ewindow_st:',ewindow_st)
        # if correction_type in {'Insertion'}:
        #     ewindow_st = offset_fromst + st_editpos
        #     ewindow_end = end_init_seq
        # else:
        #     ewindow_st = offset_fromst + st_editpos 
        #     ewindow_end = end_init_seq

        # ewindow_st = st_editpos
        # ewindow_end = end_init_seq 

        ewindow_str_lst = [f'Editing: {correction_type}'] + \
                      ['' for elm in range(st_seq, ewindow_st)]+ \
                      [f"{codes['ewindow']}*" for elm in range(ewindow_st, ewindow_st + correction_len)] + \
                      ['' for elm in range(ewindow_st + correction_len, ewindow_end)]
            
        n_rows = data_df.shape[0]
        # generate outcome (haplotype) rows
        for colname in ('Protos', 'PBS', 'RT'):
            vals = c_seq_procinit[[f'{colname}{i}' for i in range(st_seq, end_init_seq)]].values[0]
            annot_lst = [f'{colname}']
            cl_lst = []
            for annot in vals:
                if annot:
                    cl_lst += [f"{codes['init']}*"]
                else:
                    cl_lst += [' ']
            annot_lst += cl_lst
            tb.add_row(annot_lst)

        if correction_type == 'Deletion':
            tb.add_row(ewindow_str_lst)
            
        wide_init_target = c_seq_procinit[[f'L{i}' for i in range(st_seq, end_init_seq)]].values[0]
        init_target_str_lst = ['Initial sequence'] + [f'{codes[nucl]}{nucl}' for nucl in wide_init_target]
        tb.add_row(init_target_str_lst)
        
        # mutated sequence information
        # wide_mutated_target
        # PBS
        # RT
        
        end_mut_seq = end_seq
        wide_mut_target = c_seq_procmut[[f'L{i}' for i in range(st_seq, end_mut_seq)]].values[0]
        if score is not None:
            mut_target_str_lst = ['{}Mutated sequence\n Prob. edit={:.3f}'.format(codes['prob'], score)] + \
                                [f'{codes[nucl]}{nucl}' for nucl in wide_mut_target]
        else:
            mut_target_str_lst = ['{}Mutated sequence\n'.format(codes['prob'])] + \
                                 [f'{codes[nucl]}{nucl}' for nucl in wide_mut_target] 
        
        tb.add_row(mut_target_str_lst)
        
        if correction_type in {'Insertion', 'Replacement'}:
            tb.add_row(ewindow_str_lst)
            
        for colname in ('PBS', 'RT'):
            vals = c_seq_procmut[[f'{colname}{i}' for i in range(st_seq, end_mut_seq)]].values[0]
            annot_lst = [f'{colname}']
            cl_lst = []
            for annot in vals:
                if annot:
                    cl_lst += [f"{codes['mut']}*"]
                else:
                    cl_lst += [f' ']
            annot_lst += cl_lst
            tb.add_row(annot_lst)

        # seqid_str_lst = [f'Sequence ID:\n  {seqid}'] + ['' for elm in range(st_seq, end_seq)]
        # tb.add_row(seqid_str_lst)


        if return_type == 'html':
            return clss._format_html_table(tb.get_html_string())
        else: # default string
            return tb.get_string()

    @classmethod
    def _format_html_table(clss, html_str):
        html_colors = clss.html_colors
        codes = clss.codes
        nucl_colrmap = clss.nucl_colrmap
        for nucl in codes:
            ctext = codes[nucl]
            color = html_colors[nucl_colrmap[nucl]]
            html_str = re.sub(f'<td>{ctext}', '<td bgcolor="{}">'.format(color), html_str)
        return html_str

class PEVizFile:
    def __init__(self, resc_pth):
        # resc_pth: viz resources folder path
        # it contains 'header.txt',  'jupcellstyle.css', 'begin.txt', and 'end.txt'
        self.resc_pth = resc_pth
    def create(self, tablehtml, dest_pth, fname):
        resc_pth = self.resc_pth
        ls = []
        for ftname in ('header.txt',  'jupcellstyle.css', 'begin.txt'):
            with open(os.path.join(resc_pth, ftname), mode='r') as f:
                ls.extend(f.readlines())
        ls.append(tablehtml)
        with open(os.path.join(resc_pth, 'end.txt'), mode='r') as f:
            ls.extend(f.readlines())
        content = "".join(ls)
        with open(os.path.join(dest_pth, f'{fname}.html'), mode='w') as f:
            f.write(content)

def validate_df(df):
    print('number of NA:', df.isna().any().sum())

