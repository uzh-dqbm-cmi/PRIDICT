"""

PRIDICT pegRNA prediction script including calculation of features.
Additionally, nicking guides (including DeepSpCas9 score from Kim et al. 2019)
and primers for NGS amplicon (based on Primer3 from Untergasser et al. 2012) 
are designed in separate output file.

"""


### set parameters first ###

#usecase = 'website'
usecase = 'commandline'

# set filename of batchfile located in same folder as this python script
batchfile = 'batchfile_template.csv'

# set all PBS lengths which should be predicted; more PBS lengths leads to longer prediction time; default 7-15 bp
PBSlengthrange = range(7,16)


# set all RToverhang lengths which should be predicted; more RToverhang lengths leads to longer prediction time; default 3-19 bp
RToverhanglengthrange = range(3,20)

# set maximum distance of edit to PAM; longer distance leads to longer prediction time; default 25
windowsize_max=25

# Define whether DeepSpCas9 prediction should be performed for nicking guides; default True
# Changing value to False will accelerate prediction by adding dummy prediction values to nicking guides.
# Does not influence predictions performed by PRIDICT model (only affects nicking guides)
deepcas9_trigger=True

### end of parameter delcaration ###


import re
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt
import RNA
import primer3
#import numpy as np
import pandas as pd
import os
import time
import torch.multiprocessing as mp
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'



# # TODO: check if this is necessary
mp.set_start_method("spawn", force=True)



if usecase == 'website':
    import sys
    from . import prieml

    sys.modules['prieml'] = prieml

    from .prieml.predict_outcomedistrib import *
    from .prieml.utilities import get_device
    from .prieml.utilities import create_directory
    from .DeepCas9_TestCode import runprediction
    nicking=True,
    ngsprimer=True

elif usecase == 'commandline':
    import argparse
    import prieml
    from prieml.predict_outcomedistrib import *
    from prieml.utilities import get_device, create_directory
    from DeepCas9_TestCode import runprediction
    

def primesequenceparsing(sequence: str) -> object:
    """
    Function which takes target sequence with desired edit as input and 
    editing characteristics as output. Edit within brackets () and original
    equence separated by backslash from edited sequence: (A/G) == A to G mutation.
    Placeholder for deletions and insertions is '-'.

    Parameters
    ----------
    sequence : str
        Target sequence with desired edit in brackets ().

    Returns
    -------
    five_prime_seq: str
    three_prime_seq: str
    original_seq: str
    edited_seq: str
    original_base: str
    edited_base: str
    editposition: int
        DESCRIPTION.

    """
    
    sequence = sequence.replace('\n','')  # remove any spaces or linebreaks in input
    sequence = sequence.replace(' ','')
    sequence = sequence.upper()
    if sequence.count('(') != 1:
        print(sequence)
        print('More or less than one bracket found in sequence! Please check your input sequence.')
        raise ValueError

    five_prime_seq = sequence.split('(')[0]
    three_prime_seq = sequence.split(')')[1]

    sequence_set = set(sequence)
    if '/' in sequence_set:
        original_base = sequence.split('/')[0].split('(')[1]
        edited_base = sequence.split('/')[1].split(')')[0]

        # edit flanking bases should *not* be included in the brackets
        if (original_base[0] == edited_base[0]) or (original_base[-1] == edited_base[-1]):
            print(sequence)
            print('Flanking bases should not be included in brackets! Please check your input sequence.')
            raise ValueError
    elif '+' in sequence_set:  #insertion
        original_base = '-'
        edited_base = sequence.split('+')[1].split(')')[0]
    elif '-' in sequence_set:  #deletion
        original_base = sequence.split('-')[1].split(')')[0]
        edited_base = '-'

    # ignore "-" in final sequences (deletions or insertions)
    if original_base == '-':
        original_seq = five_prime_seq + three_prime_seq
        if edited_base != '-':
            mutation_type = 'Insertion'
            correction_length = len(edited_base)
        else:
            print(sequence)
            raise ValueError
    else:
        original_seq = five_prime_seq + original_base + three_prime_seq
        if edited_base == '-':
            mutation_type = 'Deletion'
            correction_length = len(original_base)
        elif len(original_base) == 1 and len(edited_base) == 1:
            if isDNA(original_base) and isDNA(edited_base):  # check if only AGCT is in bases
                mutation_type = '1bpReplacement'
                correction_length = len(original_base)
            else:
                print(sequence)
                print('Non DNA bases found in sequence! Please check your input sequence.')
                raise ValueError
        elif len(original_base) > 1 or len(edited_base) > 1:
            if isDNA(original_base) and isDNA(edited_base):  # check if only AGCT is in bases
                mutation_type = 'MultibpReplacement'
                if len(original_base) == len(
                        edited_base):  # only calculate correction length if replacement does not contain insertion/deletion
                    correction_length = len(original_base)
                else:
                    print(sequence)
                    print('Only 1bp replacements or replacements of equal length (before edit/after edit) are supported! Please check your input sequence.')
                    raise ValueError
            else:
                print(sequence)
                print('Non DNA bases found in sequence! Please check your input sequence.')
                raise ValueError

    if edited_base == '-':
        edited_seq = five_prime_seq + three_prime_seq
    else:
        edited_seq = five_prime_seq + edited_base.lower() + three_prime_seq

    if isDNA(edited_seq) and isDNA(original_seq):  # check whether sequences only contain AGCT
        pass
    else:
        raise ValueError

    basebefore_temp = five_prime_seq[
                      -1:]  # base before the edit, could be changed with baseafter_temp if Rv strand is targeted (therefore the "temp" attribute)
    baseafter_temp = three_prime_seq[:1]  # base after the edit

    editposition_left = len(five_prime_seq)
    editposition_right = len(three_prime_seq)
    return original_base, edited_base, original_seq, edited_seq, editposition_left, editposition_right, mutation_type, correction_length, basebefore_temp, baseafter_temp


def editorcharacteristics(editor):
    if editor == 'PE2-NGG':
        PAM = '(?=GG)'
        numberN = 1
        PAM_length = len(PAM) - 4 + numberN # 3 in this case
        variant = 'PE2-NGG'
        protospacerlength = 19
        PAM_side = 'right'
        primescaffoldseq = 'GTTTCAGAGCTATGCTGGAAACAGCATAGCAAGTTGAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC'
        # primescaffoldseq = 'GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC'
    return PAM, numberN, variant, protospacerlength, PAM_side, primescaffoldseq, PAM_length


def isDNA(sequence):
    """ Check whether sequence contains only DNA bases. """
    onlyDNA = True
    diff_set = set(sequence) - set('ACTGatgc')
    if diff_set:
        onlyDNA = False
        print('Non-DNA bases detected. Please use ATGC.')
        print(sequence)
        raise ValueError
    return onlyDNA

def melting_temperature(protospacer, extension, RT, RToverhang, PBS, original_base, edited_base):
    """"Calculate melting temperature for different sequences."""
    protospacermt = mt.Tm_Wallace(Seq(protospacer))
    extensionmt = mt.Tm_Wallace(Seq(extension))
    RTmt = mt.Tm_Wallace(Seq(RT))
    RToverhangmt = mt.Tm_Wallace(Seq(RToverhang))
    PBSmt = mt.Tm_Wallace(Seq(PBS))

    if original_base == '-':
        original_base_mt = 0
        original_base_mt_nan = 1
    else:
        original_base_mt = mt.Tm_Wallace(Seq(original_base))
        original_base_mt_nan = 0

    if edited_base == '-':
        edited_base_mt = 0
        edited_base_mt_nan = 1
    else:
        edited_base_mt = mt.Tm_Wallace(Seq(edited_base))
        edited_base_mt_nan = 0

    return protospacermt, extensionmt, RTmt, RToverhangmt, PBSmt, original_base_mt, edited_base_mt, original_base_mt_nan, edited_base_mt_nan


def RToverhangmatches(RToverhang, edited_seq, RToverhangstartposition, RTlengthoverhang):
    """"Counts whether RToverhang matches up to 15bp downstream of designated position in edited_seq (e.g. due to repetitive motivs) which would prevent editing of certain deletions or insertions (e.g. A(A/-)AAA with 3bp RT overhang)"""
    RToverhangmatchcount = occurrences(
        edited_seq[RToverhangstartposition:RToverhangstartposition + RTlengthoverhang + 15], RToverhang)
    return RToverhangmatchcount


def occurrences(string, sub):
    """"Gives total count of substring in string including overlapping substrings."""
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count += 1
        else:
            return count

def MFE_RNA(protospacer, protospacer_scaffold, extension, extension_scaffold, protospacer_extension_scaffold, RT, PBS):
    """"Calculates MFE (minimum free energy; RNA folding) for different sequences."""
    MFE_protospacer = RNA.fold(protospacer)[1]
    MFE_protospacer_scaffold = RNA.fold(protospacer_scaffold)[1]
    MFE_extension = RNA.fold(extension)[1]
    MFE_extension_scaffold = RNA.fold(extension_scaffold)[1]
    MFE_protospacer_extension_scaffold = RNA.fold(protospacer_extension_scaffold)[1]
    MFE_rt = RNA.fold(RT)[1]
    MFE_pbs = RNA.fold(PBS)[1]

    return MFE_protospacer, MFE_protospacer_scaffold, MFE_extension, MFE_extension_scaffold, MFE_protospacer_extension_scaffold, MFE_rt, MFE_pbs


def deepcas9(deepcas9seqlist):
    """Perform DeepCas9 prediction on 30bp stretches of protospacer + targetseq for each protospacer."""
    if deepcas9_trigger == True:
        deepcas9scorelist = runprediction(deepcas9seqlist, usecase)
        print('deepcas9 calculating...')
        deepcas9scorelist = [round(x, 2) for x in deepcas9scorelist]
        return deepcas9scorelist
    else:  # if deepcas9_trigger is set to false, just calculate dummy data for nicking guides.
        print('deepcas9 calculating...')
        deepcas9scorelist = [100.]*len(deepcas9seqlist)
        return deepcas9scorelist


def nickingguide(original_seq, PAMposition, protospacerlength):
    """Define nickingguide and corresponding 30bp stretch for DeepCas9."""
    nickprotospacer = original_seq[PAMposition - 1 - protospacerlength:PAMposition - 1]
    nickdeepcas9 = original_seq[PAMposition - 1 - protospacerlength - 4 - (20 - protospacerlength):PAMposition - 1 + 6]

    return nickprotospacer, nickdeepcas9

def load_pridict_model(run_ids=[1]):
    """construct and return PRIDICT model along with model files directory """
    
    models_lst = []
    
    # define model options
    device = get_device(True,0)
    wsize=20
    prieml_model = PRIEML_Model(device, wsize=wsize, normalize='max')

    if usecase == 'commandline':
        # trained model directory
        # print(os.path.abspath('../'))
        # print(os.path.abspath('./'))
        top_dir = os.path.dirname(__file__)
    elif usecase == 'website':
        # TODO: double check if this is the path for the webapp!
        top_dir = './predictions/pridict'
    # trained models directory
    for run_num in run_ids:
        model_dir = os.path.join(top_dir,
                            'trained_models',
                            'schwank_rnnattn',
                            'v3',
                            'train_val',
                            f'run_{run_num}')
        model_components = prieml_model.build_retrieve_models(model_dir)
        models_lst.append((prieml_model, model_components))

    return models_lst # list of trained models -- this is to allow for returning the 5 fold trained models

def deeppridict(pegdataframe, models_lst):
    """Perform score prediction on dataframe of features based on RNN model.
    
    Args:
        pegdataframe: pandas DataFrame containing the processed sequence features
        models_lst: list of tuples of (pridict_model, model_run_dir)
    
    """


    # setup the dataframe
    deepdfcols = ['RToverhanglength', 'RTlength',
                  'Correction_Type', 'Correction_Length', 'Editing_Position',
                  'protospacerlocation_only_initial', 'PBSlocation',
                  'RT_initial_location', 'RT_mutated_location', 'wide_initial_target',
                  'wide_mutated_target', 'MFE_protospacer', 'MFE_protospacer_scaffold',
                  'MFE_extension', 'MFE_extension_scaffold',
                  'MFE_protospacer_extension_scaffold', 'MFE_rt', 'MFE_pbs',
                  'RToverhangmatches', 'RTmt', 'RToverhangmt', 'PBSmt', 'protospacermt',
                  'extensionmt', 'original_base_mt', 'edited_base_mt', 'original_base_mt_nan',
                  'edited_base_mt_nan']

    deepdf = pegdataframe[deepdfcols].copy()
    # TODO: Name, seq_id, traintest and y columns can be omitted 
    deepdf.insert(0, 'Name', range(len(deepdf)))
    deepdf.insert(1, 'seq_id', range(len(deepdf)))
    deepdf.insert(2, 'traintest', ['test'] * len(deepdf))

    # deepdf.insert(len(deepdf.columns), 'correction_type_categ', deepdf['Correction_Type'])
    # correction_deletion = [1 if x == 'Deletion' else 0 for x in deepdf['Correction_Type']]
    # correction_insertion = [1 if x == 'Insertion' else 0 for x in deepdf['Correction_Type']]
    # correction_replacement = [1 if x == 'Replacement' else 0 for x in deepdf['Correction_Type']]
    # deepdf.insert(len(deepdf.columns), 'Correction_Deletion', correction_deletion)
    # deepdf.insert(len(deepdf.columns), 'Correction_Insertion', correction_insertion)
    # deepdf.insert(len(deepdf.columns), 'Correction_Replacement', correction_replacement)

    deepdf.insert(len(deepdf.columns), 'y', [0] * len(deepdf))
    deepdf['protospacerlocation_only_initial'] = deepdf['protospacerlocation_only_initial'].apply(lambda x: str(x))
    deepdf['PBSlocation'] = deepdf['PBSlocation'].apply(lambda x: str(x))
    deepdf['RT_initial_location'] = deepdf['RT_initial_location'].apply(lambda x: str(x))
    deepdf['RT_mutated_location'] = deepdf['RT_mutated_location'].apply(lambda x: str(x))

    # set mt for deletions to 0:
    deepdf['edited_base_mt'] = deepdf.apply(lambda x: 0 if x.Correction_Type == 'Deletion' else x.edited_base_mt,
                                            axis=1)

    deepdf['original_base_mt'] = deepdf.apply(lambda x: 0 if x.Correction_Type == 'Insertion' else x.edited_base_mt,
                                              axis=1)
    
    # process data and create datatensors and data loader
    prieml_model, model_components = models_lst[0] 
    dloader = prieml_model.prepare_data(deepdf, maskpos_indices=None, y_ref=['averageedited','averageunedited','averageindel'], batch_size=1500)

   # using one fold
    if len(models_lst) == 1: 
        # not elegant to access _method but it will save on loading models weights each time
        #TODO: maybe rename the _run_prediction to run_prediction :|
        pred_df = prieml_model._run_prediction(model_components, dloader, y_ref=['averageedited','averageunedited','averageindel'])
        # pred_df = prieml_model.predict_from_dloader(dloader, model_dir, y_ref=['averageedited','averageunedited','averageindel'])
    else: # using all 5 runs (i.e. folds)
        pred_lst=[]
        for run_num in range(len(models_lst)):
            prieml_model, model_components = models_lst[run_num]
            # pred_df = prieml_model.predict_from_dloader(dloader, mdir, y_ref=['averageedited','averageunedited','averageindel'])
            pred_df = prieml_model._run_prediction(model_components, dloader, y_ref=['averageedited','averageunedited','averageindel'])
            pred_lst.append(pred_df)
        pred_df = pd.concat(pred_lst, axis=0, ignore_index=True)
        pred_df = prieml_model.compute_avg_predictions(pred_df)


    editingpredictionlist = (pred_df['pred_averageedited']*100).tolist()
    unintendededitingpredictionlist = (pred_df['pred_averageindel']*100).tolist()

    return editingpredictionlist, unintendededitingpredictionlist


def primerdesign(seq):
    '''Design NGS primers flanking the edit by at least +/- 20 bp'''
    try:
        print('Designing PCR primers...')
        seq_set = set(seq)
        if '/' in seq_set:
            original_bases = seq.split('/')[0].split('(')[1]
            if original_bases == '-':
                original_bases = ''
        elif '+' in seq_set:  #insertion
            original_bases = ''
        elif '-' in seq_set:  #deletion
            original_bases = seq.split('-')[1].split(')')[0]


        original_bases_length = len(original_bases)
        seq_before = seq.split('(')[0]
        seq_after = seq.split('(')[1].split(')')[1]
        original_seq = seq_before + original_bases + seq_after
        left_primer_boundary = len(seq_before) - 20
        right_primer_boundary = 20 + original_bases_length + 20

        seqdict = {'SEQUENCE_TEMPLATE': original_seq,
                   'SEQUENCE_TARGET': [left_primer_boundary, right_primer_boundary]}

        globargs = {
            'PRIMER_OPT_SIZE': 20,
            'PRIMER_PICK_INTERNAL_OLIGO': 0,
            'PRIMER_MIN_SIZE': 18,
            'PRIMER_MAX_SIZE': 25,
            'PRIMER_OPT_TM': 60.0,
            'PRIMER_MIN_TM': 56.0,
            'PRIMER_MAX_TM': 64.0,
            'PRIMER_MIN_GC': 20.0,
            'PRIMER_MAX_GC': 80.0,
            'PRIMER_MAX_POLY_X': 100,
            'PRIMER_SALT_MONOVALENT': 50.0,
            'PRIMER_DNA_CONC': 50.0,
            'PRIMER_MAX_NS_ACCEPTED': 0,
            'PRIMER_MAX_SELF_ANY': 12,
            'PRIMER_MAX_SELF_END': 8,
            'PRIMER_PAIR_MAX_COMPL_ANY': 12,
            'PRIMER_PAIR_MAX_COMPL_END': 8,
            'PRIMER_PRODUCT_SIZE_RANGE': [130, 200],
        }

        outcome = primer3.bindings.designPrimers(seqdict, globargs)
        primerdf = pd.DataFrame.from_dict(outcome, orient='index')

        primer_left0 = outcome['PRIMER_LEFT_0_SEQUENCE']
        primer_right0 = outcome['PRIMER_RIGHT_0_SEQUENCE']
        primer_pair_penalty0 = outcome['PRIMER_PAIR_0_PENALTY']
        primer_pair_length0 = outcome['PRIMER_PAIR_0_PRODUCT_SIZE']
        primerdf_short = pd.DataFrame()
        primerdf_short.loc['bestprimers', 'PRIMER_LEFT_0_SEQUENCE'] = primer_left0
        primerdf_short.loc['bestprimers', 'PRIMER_RIGHT_0_SEQUENCE'] = primer_right0
        primerdf_short.loc['bestprimers', 'PRIMER_PAIR_0_PENALTY'] = primer_pair_penalty0
        primerdf_short.loc['bestprimers', 'PRIMER_PAIR_0_PRODUCT_SIZE'] = int(primer_pair_length0)
    except:
        print('No PCR primers generated...')
        primerdf = pd.DataFrame()

        primer_left0 = ''
        primer_right0 = ''
        primer_pair_penalty0 = ''
        primer_pair_length0 = ''
        primerdf_short = pd.DataFrame()
        primerdf_short.loc['bestprimers', 'PRIMER_LEFT_0_SEQUENCE'] = primer_left0
        primerdf_short.loc['bestprimers', 'PRIMER_RIGHT_0_SEQUENCE'] = primer_right0
        primerdf_short.loc['bestprimers', 'PRIMER_PAIR_0_PENALTY'] = primer_pair_penalty0
        primerdf_short.loc['bestprimers', 'PRIMER_PAIR_0_PRODUCT_SIZE'] = ''

    return primerdf_short, primerdf


def parallel_batch_analysis(inp_dir, inp_fname, out_dir, out_fname, num_proc_arg, nicking, ngsprimer, run_ids=[1], combine_dfs=True):
    """Perform pegRNA predictions in batch-mode."""
    batchsequencedf = pd.read_csv(os.path.join(inp_dir, inp_fname))
    if 'editseq' in batchsequencedf:
        if 'sequence_name' in batchsequencedf:
            if len(batchsequencedf.sequence_name.unique()) == len(batchsequencedf.sequence_name):
                print(f'... Designing pegRNAs for {len(batchsequencedf)} sequences ...')
                try:
                    # make sequence_name column to string even if there are only numbers
                    batchsequencedf['sequence_name'] = batchsequencedf['sequence_name'].astype(str)
                    # print(num_proc_arg)
                    # print(inp_fname)
                    run_processing_parallel(batchsequencedf, out_dir, out_fname, num_proc_arg, nicking, ngsprimer, run_ids=run_ids, combine_dfs=combine_dfs)
                except ValueError:
                    # print('***\nLess than 100bp of flanking sequence or no PAM (NGG) found in proximity of edit. Skipping...\n***\n')
                    print('***\n Error :( Check your input format is compatible with PRIDICT! More information in input box on https://pridict.it/ ...\n***\n')

            else:
                print('Please check your input-file! (Names not unique.')
        else:
            print('Please check your input-file! (Missing "sequence_name" column.')

    else:
        print('Please check your input-file! (Missing "editseq" column.')

def pegRNAfinder(dfrow, models_list, queue, pindx, pred_dir, nicking, ngsprimer,
                 editor='PE2-NGG', PBSlength_variants=PBSlengthrange, windowsize=windowsize_max,
                 RTseqoverhang_variants=RToverhanglengthrange):
    """Find pegRNAs and prediction scores for a set desired edit."""
    
    # print(dfrow)
    # print()
    try:
        if type(dfrow['editseq']) == pd.Series: # case of dfrow is a group
            sequence = dfrow['editseq'].values[0]
            name = dfrow['sequence_name'].values[0]
        else: # case of dfrow is a row in a dataframe
            sequence = dfrow['editseq'] # string
            name = dfrow['sequence_name']

        # print('seq:', sequence, type(sequence))
        # print('name:', name, type(name))

        start_time = time.time()
        original_base, edited_base, original_seq, edited_seq, editposition_left, editposition_right, mutation_type, correction_length, basebefore_temp, baseafter_temp = primesequenceparsing(
            sequence)
        if (editposition_left < 99) or (editposition_right< 99):
            print('Less than 100bp flanking sequence! Check your input.')
            raise ValueError
            
        sequence = sequence.upper()
        sequencecalculationtime = time.time() - start_time
        PAM, numberN, variant, protospacerlength, PAM_side, primescaffoldseq, PAM_length = editorcharacteristics(editor)

        mutationtypelist = []
        correctiontypelist = []
        correctionlengthlist = []
        edited_sequence_list = []  # original healthy sequence FW
        revcomp_edited_sequence_list = []  # original healthy sequence RV
        original_sequence_list = []  # mutated sequence FW
        revcomp_original_sequence_list = []  # mutated sequence RV
        mutation_position_to_PAM = []  # position of PAM motif respective to the mutation
        editedallelelist = []  # healthy base of the sequence
        originalallelelist = []  # mutated base of the sequence
        variantList = []  # Cas variant
        target_strandList = []  # defines which strand (FW or RV) should be targeted
        protospacerpamsequence = []
        protospacer_oligo_FW = []  # contains the protospacer oligo to be ordered for pegRNA cloning (FW)
        protospacer_oligo_RV = []  # contains the protospacer oligo to be ordered for pegRNA cloning (RV)
        extension_oligo_FW = []  # contains the extension oligo to be ordered for pegRNA cloning (FW)
        extension_oligo_RV = []  # contains the extension oligo to be ordered for pegRNA cloning (RV)
        editpositionlist = []
        PBSlength_variants_dic = {}
        PBSrevcomp_dic = {}
        for length in PBSlength_variants:
            PBSlength_variants_dic[length] = []
            PBSrevcomp_dic[length] = []
        RTseqoverhang_variants_dic = {}
        for length in RTseqoverhang_variants:
            RTseqoverhang_variants_dic[length] = []

        PBSsequencelist = []
        PBSrevcomplist = []
        RTseqlist = []
        RTseqoverhangrevcomplist = []
        RTseqrevcomplist = []
        deepcas9seqlist = []
        pbslengthlist = []
        rtlengthoverhanglist = []
        rtlengthlist = []
        pegRNA_list = []
        nickingprotospacerlist = []
        nickingdeepcas9list = []
        nickingpositiontoeditlist = []
        nickingtargetstrandlist = []
        mutation_position_to_nicklist = []
        nickingpe3blist = []
        nickingPAMdisruptlist = []
        nicking_oligo_FW = []
        nicking_oligo_RV = []
        protospacermtlist = []
        extensionmtlist = []
        RTmtlist = []
        RToverhangmtlist = []
        PBSmtlist = []
        original_base_mtlist = []
        edited_base_mtlist = []
        original_base_mt_nan_list = []
        edited_base_mt_nan_list = []
        MFE_protospacerlist = []
        MFE_protospacer_scaffoldlist = []
        MFE_extensionlist = []
        MFE_extension_scaffoldlist = []
        MFE_protospacer_extension_scaffoldlist = []
        MFE_rtlist = []
        MFE_pbslist = []
        rtoverhangmatchlist = []
        wide_initial_targetlist = []
        wide_mutated_targetlist = []
        protospacerlocation_only_initiallist = []
        PBSlocationlist = []
        RT_initial_locationlist = []
        RT_mutated_locationlist = []
        deepeditpositionlist = []

        if mutation_type == 'Deletion':
            correction_type = 'Deletion'
        elif mutation_type == 'Insertion':
            correction_type = 'Insertion'
        elif mutation_type == '1bpReplacement':
            correction_type = 'Replacement'
        elif mutation_type == 'MultibpReplacement':  # also set multibp replacements as replacements
            correction_type = 'Replacement'
        else:
            print('Editing type currently not supported for pegRNA prediction.')
            raise ValueError

        target_strandloop_start_time = time.time()
        for target_strand in ['Fw', 'Rv']:
            if target_strand == 'Fw':
                editposition = editposition_left
            else:
                editposition = editposition_right
                if not original_base == '-':
                    original_base = str(Seq(original_base).reverse_complement())
                if not edited_base == '-':
                    edited_base = str(Seq(edited_base).reverse_complement())
                original_seq = str(Seq(original_seq).reverse_complement())
                edited_seq = str(Seq(edited_seq).reverse_complement())

            editingWindow = range(0 - windowsize + 3, 4)  # determines how far the PAM can be away from the edit
            temp_dic = {}  # dictionary which contains many different key/value pairs which are later put together to the final lists

            X = [m.start() for m in re.finditer(PAM, original_seq)]
            X = [x for x in X if 25 <= x < len(
                original_seq) - 4]  # only use protospacer which have sufficient margin for 30bp sequence stretch
            
            # find PAMs in edited sequence for nicking guides
            editedstrand_PAMlist = [m.start() for m in re.finditer(PAM, edited_seq.upper())]
            editedstrand_PAMlist = [x for x in editedstrand_PAMlist if 25 <= x < len(edited_seq) - 4]  # only use protospacer which have sufficient margin for 30bp sequence stretch

            if X:
                xindex = 0
                editedstrandPAMindex = 0
                for editedstrandPAM in editedstrand_PAMlist:
                    editedPAM_int = editedstrand_PAMlist[editedstrandPAMindex] - editposition - numberN
                    editedstrandPAMindex = editedstrandPAMindex + 1
                    editedPAM = editedPAM_int + editposition + numberN
                    editedstart = editedPAM + (len(PAM) - 7) - 3
                    editednickposition = editedstart - editposition
                    editednickprotospacer, editednickdeepcas9 = nickingguide(edited_seq, editedstrandPAM, protospacerlength)
                    nickingtargetstrandlist.append(target_strand)
                    nickingpositiontoeditlist.append(editednickposition)
                    nickingprotospacerlist.append(editednickprotospacer)
                    nickingdeepcas9list.append(editednickdeepcas9)
                    if editednickprotospacer[0] != 'G':
                        editednickprotospacer = 'g'+editednickprotospacer
                    nicking_oligo_FW.append('cacc' + editednickprotospacer)
                    nicking_oligo_RV.append('aaac' + str(Seq(editednickprotospacer).reverse_complement()))
                    pe3bwindowlist = list(range(-5, 17))
                    pe3bwindowlist.remove(-3)
                    #print(pe3bwindowlist)
                    #print(editednickposition)
                    if editednickposition in pe3bwindowlist: # -5 or -4 for NGG or -2 to 14
                        nickingpe3blist.append('PE3b')
                        if editednickposition in [-5,-4]:
                            nickingPAMdisruptlist.append('Nicking_PAM_disrupt')
                        else:
                            nickingPAMdisruptlist.append('No_nicking_PAM_disrupt')
                    else:
                        nickingpe3blist.append('No_PE3b')
                        nickingPAMdisruptlist.append('No_nicking_PAM_disrupt')

                for xvalues in X:
                    X_int = X[xindex] - editposition - numberN
                    xindex = xindex + 1
                    XPAM = X_int + editposition + numberN
                    start = XPAM + (len(PAM) - 7) - 3
                    if X_int in editingWindow:
                        # start coordinates of RT correspond to nick position within protospacer (based on start of input sequence)
                        RTseq = {}
                        RTseqrevcomp = {}
                        RTseqoverhang = {}
                        RTseqoverhangrevcomp = {}
                        RTseqlength = {}

                        for RTlengthoverhang in RTseqoverhang_variants:  # loop which creates dictionaries containing RTseq and RTseqoverhang sequences for all different RT length specified in RTseqoverhang_variants
                            stop = editposition + len(edited_base) + RTlengthoverhang
                            if edited_base == '-':
                                stop -= 1

                            RTseq[RTlengthoverhang] = edited_seq[start:stop]
                            RTseqlength[RTlengthoverhang] = len(
                                RTseq[RTlengthoverhang])  # length of total RTseq (not only overhang)
                            RTseqoverhang[RTlengthoverhang] = edited_seq[stop - RTlengthoverhang:stop]
                            RToverhangstartposition = stop - RTlengthoverhang  # where RToverhang starts

                            RTseqrevcomp[RTlengthoverhang] = str(Seq(RTseq[RTlengthoverhang]).reverse_complement())
                            RTseqoverhangrevcomp[RTlengthoverhang] = str(
                                Seq(RTseqoverhang[RTlengthoverhang]).reverse_complement())

                        protospacerseq = 'G' + original_seq[XPAM + (len(PAM) - 7) - protospacerlength:XPAM + (len(PAM) - 7)] # attach G at position 1 to all protospacer
                        protospacerrev = Seq(protospacerseq).reverse_complement()
                        deepcas9seq = original_seq[XPAM + (len(PAM) - 12) - protospacerlength:XPAM + (len(PAM) - 1)]

                        # design different PBS lengths:
                        PBS = {}
                        PBSrevcomp = {}
                        pegRNA_dic = {}
                        # create pegRNA dictionary based on PBS/RT sequence dictionaries created above
                        for PBSlength in PBSlength_variants:
                            PBS[PBSlength] = original_seq[
                                            XPAM + (len(PAM) - 7) - protospacerlength + (
                                                    protospacerlength - PBSlength) - 3:XPAM + (len(PAM) - 7) - 3]
                            PBSrevcomp[PBSlength] = str(Seq(PBS[PBSlength]).reverse_complement())
                            
                            if 'PBS' + str(
                                    PBSlength) + 'revcomplist_temp' in temp_dic:  # only start appending after first round

                                temp_dic['PBS' + str(PBSlength) + 'revcomplist_temp'] = [
                                    temp_dic['PBS' + str(PBSlength) + 'revcomplist_temp'], PBSrevcomp[PBSlength]]
                                temp_dic['PBS' + str(PBSlength) + 'sequence_temp'] = [
                                    temp_dic['PBS' + str(PBSlength) + 'sequence_temp'], PBS[PBSlength]]
                            else:
                                temp_dic['PBS' + str(PBSlength) + 'revcomplist_temp'] = PBSrevcomp[PBSlength]
                                temp_dic['PBS' + str(PBSlength) + 'sequence_temp'] = PBS[PBSlength]
                            
                            for RTlengthoverhang in RTseqoverhang_variants:
                                pegRNA_dic['pegRNA_' + str(PBSlength) + str('_') + str(
                                    RTlengthoverhang)] = protospacerseq + primescaffoldseq + RTseqrevcomp[
                                    RTlengthoverhang] + PBSrevcomp[PBSlength]
                                        
                                ###### Adding DeepLearning features/sequences:
                                startposition = 10

                                wide_initial_target = original_seq[XPAM + (len(PAM) - 7) - protospacerlength - 10:]
                                wide_initial_target = wide_initial_target[:99]
                                wide_initial_targetlist.append(wide_initial_target)

                                wide_mutated_target = edited_seq[XPAM + (len(PAM) - 7) - protospacerlength - 10:]
                                wide_mutated_target = wide_mutated_target[:99]
                                wide_mutated_targetlist.append(wide_mutated_target)

                                deepeditposition = startposition + protospacerlength - 3 + ((X_int - 3) * -1)
                                deepeditpositionlist.append(deepeditposition)

                                protospacerlocation_only_initial = [startposition, startposition + protospacerlength]
                                protospacerlocation_only_initiallist.append(protospacerlocation_only_initial)

                                PBSlocation = [startposition + protospacerlength - 3 - PBSlength,
                                            startposition + protospacerlength - 3]
                                PBSlocationlist.append(PBSlocation)

                                if correction_type == 'Replacement':
                                    RT_initial_location = [startposition + protospacerlength - 3,
                                                        startposition + protospacerlength - 3 + len(
                                                            RTseq[RTlengthoverhang])]
                                    RT_mutated_location = [startposition + protospacerlength - 3,
                                                        startposition + protospacerlength - 3 + len(
                                                            RTseq[RTlengthoverhang])]
                                elif correction_type == 'Deletion':
                                    RT_initial_location = [startposition + protospacerlength - 3,
                                                        startposition + protospacerlength - 3 + len(
                                                            RTseq[RTlengthoverhang]) + correction_length]
                                    RT_mutated_location = [startposition + protospacerlength - 3,
                                                        startposition + protospacerlength - 3 + len(
                                                            RTseq[RTlengthoverhang])]

                                elif correction_type == 'Insertion':
                                    RT_initial_location = [startposition + protospacerlength - 3,
                                                        startposition + protospacerlength - 3 + len(
                                                            RTseq[RTlengthoverhang]) - correction_length]
                                    RT_mutated_location = [startposition + protospacerlength - 3,
                                                        startposition + protospacerlength - 3 + len(
                                                            RTseq[RTlengthoverhang])]

                                RT_initial_locationlist.append(RT_initial_location)
                                RT_mutated_locationlist.append(RT_mutated_location)
                                ######

                                extension_oligo_FW.append(
                                    primescaffoldseq[-4:] + RTseqrevcomp[RTlengthoverhang] + PBSrevcomp[PBSlength])
                                extension_oligo_RV.append('AAAA' + str(PBS[PBSlength]) + str(RTseq[RTlengthoverhang]))
                                RTseqrevcomplist.append(RTseqrevcomp[RTlengthoverhang])
                                RTseqlist.append(RTseq[RTlengthoverhang])
                                RTseqoverhangrevcomplist.append(RTseqoverhangrevcomp[RTlengthoverhang])
                                pbslengthlist.append(
                                    PBSlength)  # add PBS length to list and to table in the end
                                rtlengthoverhanglist.append(
                                    RTlengthoverhang)  # add RT overhang length to list and to table in the end
                                rtlengthlist.append(RTseqlength[RTlengthoverhang])
                                target_strandList.append(target_strand)
                                protospacerpamsequence.append(protospacerseq)
                                protospacer_oligo_FW.append('CACC' + protospacerseq + primescaffoldseq[:5])
                                protospacer_oligo_RV.append(
                                    str(Seq(primescaffoldseq[:9]).reverse_complement()) + str(protospacerrev))
                                deepcas9seqlist.append(deepcas9seq)
                                PBSsequencelist.append(PBS[PBSlength])
                                PBSrevcomplist.append(PBSrevcomp[PBSlength])
                                editpositionlist.append(editposition - start)
                                pegRNA = protospacerseq + primescaffoldseq + RTseqrevcomp[RTlengthoverhang] + PBSrevcomp[
                                    PBSlength]
                                pegRNA_list.append(pegRNA)
                                edited_sequence_list.append(edited_seq)
                                revcomp_edited_sequence_list.append(str(Seq(edited_seq).reverse_complement()))
                                original_sequence_list.append(original_seq)
                                revcomp_original_sequence_list.append(str(Seq(original_seq).reverse_complement()))
                                variantList.append(variant)
                                editedallelelist.append(edited_base)
                                originalallelelist.append(original_base)
                                mutationtypelist.append(mutation_type)
                                correctiontypelist.append(correction_type)
                                mutation_position_to_PAM.append(X_int)
                                mutation_position_to_nicklist.append(X_int - 3)
                                correctionlengthlist.append(correction_length)
                                protospacermt, extensionmt, RTmt, RToverhangmt, PBSmt, original_base_mt, edited_base_mt, original_base_mt_nan, edited_base_mt_nan = melting_temperature(
                                    protospacerseq, RTseqrevcomp[RTlengthoverhang] + PBSrevcomp[PBSlength],
                                    RTseqrevcomp[RTlengthoverhang], RTseqoverhangrevcomp[RTlengthoverhang],
                                    PBSrevcomp[PBSlength], original_base, edited_base)
                                protospacermtlist.append(protospacermt)
                                extensionmtlist.append(extensionmt)
                                RTmtlist.append(RTmt)
                                RToverhangmtlist.append(RToverhangmt)
                                PBSmtlist.append(PBSmt)
                                original_base_mtlist.append(original_base_mt)
                                edited_base_mtlist.append(edited_base_mt)
                                original_base_mt_nan_list.append(original_base_mt_nan)
                                edited_base_mt_nan_list.append(edited_base_mt_nan)
                                MFE_protospacer, MFE_protospacer_scaffold, MFE_extension, MFE_extension_scaffold, MFE_protospacer_extension_scaffold, MFE_rt, MFE_pbs = MFE_RNA(
                                    protospacerseq, protospacerseq + primescaffoldseq,
                                                    RTseqrevcomp[RTlengthoverhang] + PBSrevcomp[PBSlength],
                                                    primescaffoldseq + RTseqrevcomp[RTlengthoverhang] + PBSrevcomp[
                                                        PBSlength],
                                                    protospacerseq + primescaffoldseq + RTseqrevcomp[RTlengthoverhang] +
                                                    PBSrevcomp[
                                                        PBSlength], RTseqrevcomp[RTlengthoverhang], PBSrevcomp[PBSlength])
                                MFE_protospacerlist.append(MFE_protospacer)
                                MFE_protospacer_scaffoldlist.append(MFE_protospacer_scaffold)
                                MFE_extensionlist.append(MFE_extension)
                                MFE_extension_scaffoldlist.append(MFE_extension_scaffold)
                                MFE_protospacer_extension_scaffoldlist.append(MFE_protospacer_extension_scaffold)
                                MFE_rtlist.append(MFE_rt)
                                MFE_pbslist.append(MFE_pbs)
                                rtoverhangmatch = RToverhangmatches(RTseqoverhang[RTlengthoverhang], edited_seq,
                                                                    RToverhangstartposition, RTlengthoverhang)
                                rtoverhangmatchlist.append(rtoverhangmatch)


        target_strandlooptime = time.time() - target_strandloop_start_time   
        start_time = time.time()
        if nicking:
            nickingdeepcas9scorelist = deepcas9(nickingdeepcas9list)
        nickingdeepcas9time = time.time() - start_time
        start_time = time.time()
        if ngsprimer:
            primerdf_short, primerdf = primerdesign(sequence)  # design PCR primers for targetseq
        ngsprimertime = time.time() - start_time

        ### check whether deepcas9 sequence was made the same for the training set as here!

        if nicking:
            nickingdf = pd.DataFrame(
                {'Nicking-Protospacer': nickingprotospacerlist, 'Nicking-Position-to-edit': nickingpositiontoeditlist,
                'PE3b': nickingpe3blist,
                'Nicking-PAMdisrupt': nickingPAMdisruptlist, 'Target_Strand': nickingtargetstrandlist,
                '30bpseq': nickingdeepcas9list, 'DeepCas9score': nickingdeepcas9scorelist,
                'Nicking-Proto-Oligo-FW': nicking_oligo_FW,
                'Nicking-Proto-Oligo-RV': nicking_oligo_RV})

    
        pegdataframe = pd.DataFrame({'Original_Sequence': original_sequence_list, 'Edited-Sequences': edited_sequence_list,
                                    'Target-Strand': target_strandList, 'Mutation_Type': mutationtypelist,
                                    'Correction_Type': correctiontypelist, 'Correction_Length': correctionlengthlist,
                                    'Editing_Position': editpositionlist,
                                    'PBSlength': pbslengthlist, 'RToverhanglength': rtlengthoverhanglist,
                                    'RTlength': rtlengthlist, 'EditedAllele': editedallelelist,
                                    'OriginalAllele': originalallelelist, 'Protospacer-Sequence': protospacerpamsequence,
                                    'PBSrevcomp13bp': PBSrevcomplist, 'RTseqoverhangrevcomp': RTseqoverhangrevcomplist,
                                    'RTrevcomp': RTseqrevcomplist,
                                    'Protospacer-Oligo-FW': protospacer_oligo_FW,
                                    'Protospacer-Oligo-RV': protospacer_oligo_RV,
                                    'Extension-Oligo-FW': extension_oligo_FW, 'Extension-Oligo-RV': extension_oligo_RV,
                                    'pegRNA': pegRNA_list,
                                    'Editor_Variant': variantList,
                                    'protospacermt': protospacermtlist,
                                    'extensionmt': extensionmtlist, 'RTmt': RTmtlist, 'RToverhangmt': RToverhangmtlist,
                                    'PBSmt': PBSmtlist,
                                    'original_base_mt': original_base_mtlist, 'edited_base_mt': edited_base_mtlist,
                                    'original_base_mt_nan': original_base_mt_nan_list, 'edited_base_mt_nan': edited_base_mt_nan_list,
                                    'RToverhangmatches': rtoverhangmatchlist, 'MFE_protospacer': MFE_protospacerlist,
                                    'MFE_protospacer_scaffold': MFE_protospacer_scaffoldlist,
                                    'MFE_extension': MFE_extensionlist,
                                    'MFE_extension_scaffold': MFE_extension_scaffoldlist,
                                    'MFE_protospacer_extension_scaffold': MFE_protospacer_extension_scaffoldlist,
                                    'MFE_rt': MFE_rtlist, 'MFE_pbs': MFE_pbslist,
                                    'wide_initial_target': wide_initial_targetlist,
                                    'wide_mutated_target': wide_mutated_targetlist,
                                    'protospacerlocation_only_initial': protospacerlocation_only_initiallist,
                                    'PBSlocation': PBSlocationlist, 'RT_initial_location': RT_initial_locationlist,
                                    'RT_mutated_location': RT_mutated_locationlist,
                                    'deepeditposition': deepeditpositionlist})
        

        if len(pegdataframe) < 1:
            print('\n***\nNo PAM (NGG) found in proximity of edit!\n***\n')
            raise ValueError
                
        start_time = time.time()
        predictions_deep, unintendedpredictions_deep = deeppridict(pegdataframe, models_list)
        pridict_time = time.time() - start_time
        
        print()
        print("Calculating features took", round(target_strandlooptime,1), "seconds to run.")
        #print("Sequence preparation took", sequencecalculationtime, "to run")
        print("Deep model took", round(pridict_time,1), "seconds to run.")
        # print("NickingDeepSpCas9 took", nickingdeepcas9time, "to run")
        # print("NGSprimerdesign took", ngsprimertime, "to run")
        

        pegdataframe.insert(len(pegdataframe.columns), 'sequence_name', name)
        pegdataframe.insert(len(pegdataframe.columns), 'PRIDICT_editing_Score_deep', predictions_deep)
        pegdataframe.insert(len(pegdataframe.columns), 'PRIDICT_unintended_Score_deep', unintendedpredictions_deep)
        pegdataframe.sort_values(by=['PRIDICT_editing_Score_deep'], inplace=True, ascending=False)
        pegdataframe.insert(len(pegdataframe.columns), 'rank', range(1, len(pegdataframe) + 1))  # add rank for pegRNAs


        # print('prediction directory:', pred_dir)
        if nicking:
            nickingdf.sort_values(by=['DeepCas9score'], inplace=True, ascending=False)
            nickingdf.to_csv(os.path.join(pred_dir, name + '_nicking_guides.csv'))
        


        # cols = ['Protospacer-Oligo-FW',
        #    'Protospacer-Oligo-RV', 'Extension-Oligo-FW', 'Extension-Oligo-RV',
        #    'Original_Sequence', 'Edited-Sequences', 'Target-Strand',
        #    'Mutation_Type', 'Correction_Type', 'Correction_Length',
        #    'Editing_Position', 'PBSlength', 'RToverhanglength',
        #    'RTlength', 'EditedAllele', 'OriginalAllele',
        #    'Protospacer-Sequence', 'PBSrevcomp13bp',
        #    'RTseqoverhangrevcomp', 'RTrevcomp',
        #    'pegRNA', 'Editor_Variant']

        # pred_cols = ['PRIDICT_editing_Score_deep', 'PRIDICT_unintended_Score_deep', 'rank']
        
        # best pegdf
        # topk=3
        # tmp = pegdataframe.iloc[0:topk].copy()
        # tmp['PCR_Primer3_FW'] =  primerdf_short.at['bestprimers', 'PRIMER_LEFT_0_SEQUENCE']
        # tmp['PCR_Primer3_RV'] = primerdf_short.at['bestprimers', 'PRIMER_RIGHT_0_SEQUENCE']
        # tmp['PCR_Primer3_PRODUCT_SIZE'] = primerdf_short.at['bestprimers', 'PRIMER_PAIR_0_PRODUCT_SIZE']
        # tmp.to_csv(os.path.join(pred_dir, f'{name}_best_pegdf.csv'), index=False)

        # pegdataframe_short = pegdataframe[get_shortdf_colnames()]
        # pegdataframe_short.to_csv(os.path.join(pred_dir, name + '_pegRNA_Pridict_Scores.csv'))
        pegdataframe.to_csv(os.path.join(pred_dir, name + '_pegRNA_Pridict_full.csv'))
        if ngsprimer:
            primerdf_short.to_csv(os.path.join(pred_dir, name + '_best_PCR_primers.csv'))

    except Exception as e:
        print('-- Exception occured --')
        print(e)
    finally:
        queue.put(pindx)

# editseq_test = 'GCCTGGAGGTGTCTGGGTCCCTCCCCCACCCGACTACTTCACTCTCTGTCCTCTCTGCCCAGGAGCCCAGGATGTGCGAGTTCAAGTGCTACCCGA(G/C)GTGCGAGGCCAGCTCGGGGGCACCGTGGAGCTGCCGTGCCACCTGCTGCCACCTGTTCCTGGACTGTACATCTCCCTGGTGACCTGGCAGCGCCCAGATGCACCTGCGAACCACCAGAATGTGGCCGC'

def fix_mkl_issue():
    # https://github.com/pytorch/pytorch/issues/37377
    # https://github.com/IntelPython/mkl-service/issues/14
    import mkl
    a, b, __ = mkl.__version__.split('.')
    if int(a) <=2 and int(b)<=3:
        print('setting MKL_THREADING_LAYER = GNU')
        os.environ['MKL_THREADING_LAYER'] = 'GNU'

def run_processing_parallel(df, pred_dir, fname, num_proc_arg, nicking, ngsprimer, run_ids=[1], combine_dfs=True):

    fix_mkl_issue() # comment this 
    queue = mp.Queue()
    q_processes = []
    if num_proc_arg == 0:
        num_proc = mp.cpu_count()
    elif num_proc_arg <= mp.cpu_count():
        num_proc = num_proc_arg
    else:
        num_proc = mp.cpu_count()


    num_rows = len(df)
    seqnames_lst = []
    models_lst = load_pridict_model(run_ids = run_ids)

    for q_i in range(min(num_proc, num_rows)):
        # print('q_i:', q_i)
        row = df.iloc[q_i] # slice a row
        seqnames_lst.append(row['sequence_name'])
        print('processing sequence:', seqnames_lst[-1])
        q_process = create_q_process(dfrow=row,
                                     models_list=models_lst,
                                     queue=queue,
                                     pindx=q_i,
                                     pred_dir=pred_dir, nicking=nicking, ngsprimer=ngsprimer)
        q_processes.append(q_process)
        spawn_q_process(q_process)

    spawned_processes = min(num_proc, num_rows)

    print("*"*25)
    for q_i in range(num_rows):
        join_q_process(q_processes[q_i])
        released_proc_num = queue.get()
        # print("released_process_num:", released_proc_num)
        q_processes[q_i] = None # free resources ;)
        if(spawned_processes < num_rows):
            q_i_upd = q_i + num_proc
            # print('q_i:', q_i, 'q_i_updated:', q_i_upd)
            row = df.iloc[q_i_upd]
            seqnames_lst.append(row['sequence_name'])
            print('processing sequence:', seqnames_lst[-1])
            q_process = create_q_process(dfrow=row, 
                                         models_list=models_lst,
                                         queue=queue,
                                         pindx=q_i_upd,
                                         pred_dir=pred_dir, nicking=nicking, ngsprimer=ngsprimer)

            q_processes.append(q_process)
            spawn_q_process(q_process)
            spawned_processes = spawned_processes + 1
    
    # assemble all sequence dataframes into one --- optional
    if combine_dfs:
        combined_df = assemble_df(seqnames_lst, 'pegRNA_Pridict_full', pred_dir)
        remove_col(combined_df, 'Unnamed: 0')
        
        combined_df.to_csv(os.path.join(pred_dir, f'{fname}_pegRNA_Pridict_full.csv'), index=False)

        cols = get_shortdf_colnames()
        pegdataframe_short = combined_df[cols]
        pegdataframe_short.to_csv(os.path.join(pred_dir, f'{fname}_pegRNA_Pridict_Scores.csv'), index=False)

        combined_bestpegdf = assemble_df(seqnames_lst, 'best_pegdf', pred_dir)
        combined_bestpegdf.to_csv(os.path.join(pred_dir, f'{fname}_best_pegdf.csv'), index=False)


def remove_col(df, colname):
    if colname in df:
        del df[colname]

def get_shortdf_colnames():
    cols = ['Protospacer-Oligo-FW',
    'Protospacer-Oligo-RV', 'Extension-Oligo-FW', 'Extension-Oligo-RV',
    'Original_Sequence', 'Edited-Sequences', 'Target-Strand',
    'Mutation_Type', 'Correction_Type', 'Correction_Length',
    'Editing_Position', 'PBSlength', 'RToverhanglength',
    'RTlength', 'EditedAllele', 'OriginalAllele',
    'Protospacer-Sequence', 'PBSrevcomp13bp',
    'RTseqoverhangrevcomp', 'RTrevcomp',
    'pegRNA', 'Editor_Variant', 'sequence_name', 'PRIDICT_editing_Score_deep', 'PRIDICT_unintended_Score_deep', 'rank']
    return cols

def assemble_df(seqnames_lst, namesuffix, df_dir):
    df_lst = []
    for seq_name in seqnames_lst:
        fpath = os.path.join(df_dir, f'{seq_name}_{namesuffix}.csv')
        if os.path.isfile(fpath):
            df =  pd.read_csv(fpath)
            df['sequence_name'] = seq_name
            df_lst.append(df)
    combined_df = pd.concat(df_lst, axis=0)
    return combined_df


def spawn_q_process(q_process):
    print(">>> spawning row computation process")
    q_process.start()
    
def join_q_process(q_process):
    q_process.join()
    print("<<< joined row computation process")
    
def create_q_process(dfrow, models_list, queue, pindx, pred_dir, nicking, ngsprimer):
    return mp.Process(target=pegRNAfinder, args=(dfrow, models_list, queue, pindx, pred_dir, nicking, ngsprimer))

# def rank_seqs(gdf, pred_dir):
#     topk=3
#     gdf.sort_values(by=['PRIDICT_editing_Score_deep'], inplace=True, ascending=False)
#     gdf.insert(len(gdf.columns), 'rank', range(1, len(gdf) + 1))  # add rank for pegRNAs
#     seq_name = gdf['sequence_name'].values[0]
#     primerdf_short = pd.read_csv(os.path.join(pred_dir, f'{seq_name}_best_PCR_primers.csv'))
#     # we need these best pegdf for each sequence to assemble later to create a combined df
#     tmp = gdf.iloc[0:topk].copy()
#     tmp['PCR_Primer3_FW'] =  primerdf_short.at[0, 'PRIMER_LEFT_0_SEQUENCE']
#     tmp['PCR_Primer3_RV'] = primerdf_short.at[0, 'PRIMER_RIGHT_0_SEQUENCE']
#     tmp['PCR_Primer3_PRODUCT_SIZE'] = primerdf_short.at[0, 'PRIMER_PAIR_0_PRODUCT_SIZE']
#     tmp.to_csv(os.path.join(pred_dir, f'{seq_name}_best_pegdf.csv'), index=False)
#     return gdf

# def interactive_run():
#     analysistype = input("Batchmode (enter '1') or manual mode (enter '2'):\n")
#     if int(analysistype) == 1:

#         batchanalysis(batchfile)
#     elif int(analysistype) == 2:
#         name = input("Name of edited sequence (Default :'unnamed'):\n")
#         editseq = input("Enter sequence to edit (format: xxxxxxxxx(a/g)xxxxxxxxxx):\n")
#         start_time = time.time()
#         pegdataframe, nickingdf, primerdf_short = pegRNAfinder(editseq, name, 'PE2-NGG')
#         print("My program took", time.time() - start_time, "to run")
#     else:
#         print("Please enter 1 or 2 for different analysis mode.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running PRIDICT to design and predict pegRNAs.")

    subparser = parser.add_subparsers(dest='command')
    manual_m = subparser.add_parser('manual')
    batch_m  = subparser.add_parser('batch')

    manual_m.add_argument("--sequence-name", type=str, help="Name of the sequence (i.e. unique id for the sequence)", required=True)
    manual_m.add_argument("--sequence", type=str, help="Target sequence to edit (format: xxxxxxxxx(a/g)xxxxxxxxxx). Use quotation marks before and after the sequence.", required=True)
    manual_m.add_argument("--output-dir", type=str, default='./predictions', help="Output directory where results are saved on disk.")
    manual_m.add_argument("--use_5folds", action='store_true', help="Use all 5-folds trained models. Default is to use fold-1 model.")
    manual_m.add_argument("--cores", type=int, default=1, help="Number of cores to use for multiprocessing. Default value 0 uses all available cores.")
    manual_m.add_argument("--nicking", action='store_true', help="Additionally, design nicking guides for edit (PE3) with DeepSpCas9 prediction.")
    manual_m.add_argument("--ngsprimer", action='store_true', help="Additionally, design NGS primers for edit based on Primer3 design.")


    batch_m.add_argument("--input-dir", type=str, default='./input', help="Input directory where the input csv file is found on disk")
    batch_m.add_argument("--input-fname", type=str, required=True, help="Input filename - name of csv file that has two columns {editseq, sequence_name}. See batch_template.csv in the ./input folder ")
    batch_m.add_argument("--output-dir", type=str, default='./predictions', help="Output directory where results are dumped on disk")
    batch_m.add_argument("--output-fname", type=str, help="Output filename for the resulting dataframe. If not specified, the name of the input file will be used")

    batch_m.add_argument("--use_5folds", action='store_true', help="Use all 5-folds trained models. Default is to use fold-1 model")
    batch_m.add_argument("--combine_results", action='store_true', help="Compile all results into one dataframe")
    batch_m.add_argument("--nicking", action='store_true', help="Additionally, design nicking guides for edit (PE3) with DeepSpCas9 prediction.")
    batch_m.add_argument("--ngsprimer", action='store_true', help="Additionally, design NGS primers for edit based on Primer3 design.")
    batch_m.add_argument("--cores", type=int, default=0, help="Number of cores to use for multiprocessing. Default value 0 uses all available cores.")
    

    args = parser.parse_args()

    if args.command == 'manual':
        print('Running in manual mode:')
        df = pd.DataFrame({'sequence_name': [args.sequence_name], 'editseq': [args.sequence]})

        if args.output_dir != './predictions':
            out_dir = create_directory(args.output_dir, os.getcwd())
        else:
            out_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
        print('output directory:', out_dir)

        fname = f'{args.sequence_name}'
        if args.use_5folds:
            run_ids = list(range(5))
        else:
            run_ids = [1]
        
        if args.cores:
            num_proc_arg=args.cores
        else:
            num_proc_arg=1
        
        if args.nicking:
            nicking=True
        else:
            nicking=False
            
        if args.ngsprimer:
            ngsprimer=True
        else:
            ngsprimer=False
        
        run_processing_parallel(df, out_dir, fname, num_proc_arg, nicking, ngsprimer, run_ids=run_ids, combine_dfs=False)

    elif args.command == 'batch':
        print('Running in batch mode:')

        if args.input_dir != './input':
            inp_dir = create_directory(args.input_dir, os.getcwd())
        else:
            inp_dir = os.path.join(os.path.dirname(__file__), args.input_dir)
        print('input directory:', inp_dir)

        inp_fname = args.input_fname

        if args.output_dir != './predictions':
            out_dir = create_directory(args.output_dir, os.getcwd())
        else:
            out_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
        print('output directory:', out_dir)

        if args.use_5folds:
            run_ids = list(range(5))
        else:
            run_ids = [1]

        if args.output_fname:
            out_fname = args.output_fname
        else:
            out_fname = args.input_fname.split('.')[0]
        
        if args.cores:
            num_proc_arg=args.cores
        else:
            num_proc_arg=0
            
        if args.nicking:
            nicking=True
        else:
            nicking=False
            
        if args.ngsprimer:
            ngsprimer=True
        else:
            ngsprimer=False
        # print(num_proc_arg)
        # print('run_ids:', run_ids)
        # print('args.combine_results:', args.combine_results)
        parallel_batch_analysis(inp_dir, inp_fname, out_dir, out_fname, num_proc_arg, nicking, ngsprimer, run_ids=run_ids, combine_dfs=args.combine_results)
    elif args.command == None:
        print('Please specify how to run PRIDICT ("manual" or "batch") as argument after the script name.')
    #     out_dir = "./predictions"
    #     out_fname = batchfile[:-3] + "predictions.csv"
    #     fname = out_fname
    #     run_ids = [1]
    #     combine_dfs = False
    #     num_proc_arg = 0
    #     analysistype = input("Batchmode (enter '1') or manual mode (enter '2'):\n")
    #     if int(analysistype) == 1:
    #         inp_dir = "./input"
    #         inp_fname = batchfile
    #         parallel_batch_analysis(inp_dir, inp_fname, out_dir, out_fname, num_proc_arg, run_ids=run_ids, combine_dfs=args.combine_results)
    #     elif int(analysistype) == 2:
    #         name = input("Name of edited sequence (Default :'unnamed'):\n")
    #         editseq = input("Enter sequence to edit (format: xxxxxxxxx(a/g)xxxxxxxxxx):\n")
    #         df = pd.DataFrame({'sequence_name': [name], 'editseq': [editseq]})
    #         run_processing_parallel(df, out_dir, fname, num_proc_arg, run_ids=run_ids, combine_dfs=False)

    #     else:
    #         print("Please enter 1 or 2 for different analysis mode.")
            