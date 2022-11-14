# -*- coding: utf-8 -*-
"""
Created on Mon May 24 00:15:52 2021

@author: nimath
"""

import pandas as pd

# C2df = pd.read_csv('20210522_05-NM-C2_analysisdf.csv')
# C2df = C2df[['Name', 'Gene', 'Phenotype', 'Disease_Block', 'PBSlength',
#        'RToverhanglength', 'RTlength', 'First_RT_nuc', 'Poly_T',
#        'ReferenceAllele', 'AlternateAllele', 'Correction_Type',
#        'Correction_Length', 'DesignNr_per_variant', 'Editing_Position',
#        'Target_Strand', 'Duplicate', 'WT_Target_Correct',
#        'mutated_Target_Correct', 'editedcount', 'uneditedcount', 'indelcount',
#        'totalreads', 'percentageedited', 'percentageunedited',
#        'percentageindel', 'barcodenr', 'multi_barcodes']]
# C2df.to_csv('20210522_05-NM-C2_analysisdf_no_barcode.csv')

Ctr_1_df = pd.read_csv('20220730_04-NM-C1_analysisdf_endo.csv')
Ctr_2_df = pd.read_csv('20220730_05-NM-C2_analysisdf_endo.csv')
Ctr_df = Ctr_1_df.copy()

sum_totalreads = Ctr_1_df['totalreads'] + Ctr_2_df['totalreads']
sum_editedcount = Ctr_1_df['editedcount'] + Ctr_2_df['editedcount']
sum_uneditedcount = Ctr_1_df['uneditedcount'] + Ctr_2_df['uneditedcount']
sum_indelcount = Ctr_1_df['indelcount'] + Ctr_2_df['indelcount']
sum_nickindelcount = Ctr_1_df['nickindelcount'] + Ctr_2_df['nickindelcount']
sum_beforeflapindelcount = Ctr_1_df['beforeflapindelcount'] + Ctr_2_df['beforeflapindelcount']
# Ctr_1_df['multi_barcodes'] = Ctr_1_df['multi_barcodes'].apply(lambda x: set(x.replace("'","").strip("{}").split(', ')))
# Ctr_2_df['multi_barcodes'] = Ctr_2_df['multi_barcodes'].apply(lambda x: set(x.replace("'","").strip("{}").split(', ')))



# sum_barcodes = Ctr_1_df['multi_barcodes'] |= Ctr_2_df['multi_barcodes']

Ctr_1_df['barcodes'] = Ctr_1_df['barcodes'].apply(lambda x: x.replace("'","").strip("[]").split(', '))
Ctr_2_df['barcodes'] = Ctr_2_df['barcodes'].apply(lambda x: x.replace("'","").strip("[]").split(', '))
sum_barcodes = Ctr_1_df['barcodes'] + Ctr_2_df['barcodes']

Ctr_df['totalreads'] = sum_totalreads
Ctr_df['editedcount'] = sum_editedcount
Ctr_df['uneditedcount'] = sum_uneditedcount
Ctr_df['indelcount'] = sum_indelcount
Ctr_df['nickindelcount'] = sum_nickindelcount
Ctr_df['beforeflapindelcount'] = sum_beforeflapindelcount
Ctr_df['barcodes'] = sum_barcodes

for index, row in Ctr_1_df.iterrows():
    # Ctr_df.at[index,'multi_barcodes'] = Ctr_1_df.at[index,'multi_barcodes'].union(Ctr_2_df.at[index,'multi_barcodes'])
    Ctr_df.at[index,'barcodenr'] = len(set(Ctr_df.at[index,'barcodes']))

percentageedited = (Ctr_df["editedcount"]/Ctr_df['totalreads'])*100
Ctr_df['percentageedited'] = percentageedited
percentageunedited = (Ctr_df["uneditedcount"]/Ctr_df['totalreads'])*100
Ctr_df['percentageunedited'] = percentageunedited
percentageindels = (Ctr_df["indelcount"]/Ctr_df['totalreads'])*100
Ctr_df['percentageindel'] = percentageindels


Ctr_df.to_csv('20220730_04-NM-CtrCombined_analysisdf_focused_full.csv')

Ctr_short_df = Ctr_df[['Name', 'Gene', 'Phenotype', 'Disease_Block', 'PBSlength',
       'RToverhanglength', 'RTlength', 'First_RT_nuc', 'Poly_T',
       'ReferenceAllele', 'AlternateAllele', 'Correction_Type',
       'Correction_Length', 'DesignNr_per_variant', 'Editing_Position',
       'Target_Strand', 'Duplicate', 'WT_Target_Correct',
       'mutated_Target_Correct', 'editedcount', 'uneditedcount', 'indelcount',
       'nickindelcount','beforeflapindelcount',
       'totalreads', 'percentageedited', 'percentageunedited',
       'percentageindel']]

Ctr_short_df.to_csv('20220730_04-NM-CtrCombined_analysisdf_focused.csv')

# C2df = C2df[['Name', 'Gene', 'Phenotype', 'Disease_Block', 'PBSlength',
#        'RToverhanglength', 'RTlength', 'First_RT_nuc', 'Poly_T',
#        'ReferenceAllele', 'AlternateAllele', 'Correction_Type',
#        'Correction_Length', 'DesignNr_per_variant', 'Editing_Position',
#        'Target_Strand', 'Duplicate', 'WT_Target_Correct',
#        'mutated_Target_Correct', 'editedcount', 'uneditedcount', 'indelcount',
#        'totalreads', 'percentageedited', 'percentageunedited',
#        'percentageindel', 'barcodenr', 'multi_barcodes']]
# C2df.to_csv('20210522_05-NM-C2_analysisdf_no_barcode.csv')