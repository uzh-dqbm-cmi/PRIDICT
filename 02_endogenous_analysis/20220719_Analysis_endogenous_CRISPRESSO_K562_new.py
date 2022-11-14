# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:47:50 2021

@author: nimath
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


path_arrayed = 'C:\\Users\\nimath\\switchdrive\\Institution\\PhD\\01_Experiments\\05_Prime_Editing\\03_DiseaseScreen\\05_Validations\\02_Arrayed\\'
libraryedits = pd.read_csv(path_arrayed+'20210714_Arrayed_Kimetal_Order_Filtered.csv')
libraryedits['corresponding_endogenous_editname'] = libraryedits['Shortname'].apply(lambda x: x+'libendopeg')
libraryedits = libraryedits.set_index('corresponding_endogenous_editname')

librarypredictioneditsall = pd.read_csv(path_arrayed+'20210907_all_endopegRNAs_with_features.csv')
librarypredictioneditsall = librarypredictioneditsall.set_index('pegRNA',drop=False)

ordereddf =  pd.read_csv(path_arrayed+'20210718_Gblock_Ordering_Twist_FINAL_adapted1and17best.csv')

librarypredictionfinal = pd.DataFrame()
for index, row in ordereddf.iterrows():
    matchguide = librarypredictioneditsall.loc[row.pegRNA,:]
    name = row['Name'].replace('_','')
    matchguide['name'] = name
    librarypredictionfinal = librarypredictionfinal.append(matchguide)

librarypredictionfinal = librarypredictionfinal.set_index('name',drop=False)
librarypredictionfinal['PRIDICT_Score'] = librarypredictionfinal['PRIDICT_Score_deep']

path_rep1 = "C:\\Users\\nimath\\switchdrive\\Institution\\PhD\\01_Experiments\\05_Prime_Editing\\03_DiseaseScreen\\05_Validations\\02_Arrayed\\Sequencing\\NGS\\20211124_K562_Rep1\\"
path_rep2 = "C:\\Users\\nimath\\switchdrive\\Institution\\PhD\\01_Experiments\\05_Prime_Editing\\03_DiseaseScreen\\05_Validations\\02_Arrayed\\Sequencing\\NGS\\20211124_K562_Rep2\\"

path_rep1 = path_rep1 + "FASTQ\\CRISPRessoBatch_on_endogenous_K562_focused_updated_Rep1\\"
path_rep2 = path_rep2 + "FASTQ\\CRISPRessoBatch_on_endogenous_K562_focused_updated_Rep2\\"

samplefolderlist_rep1 = [f for f in os.listdir(path_rep1) if not "." in f]  # make list with subfolders corresponding to samples of Replicate 1
samplefolderlist_rep2 = [f for f in os.listdir(path_rep2) if not "." in f]  # make list with subfolders corresponding to samples of Replicate 2

### Control editing rates:
path_ctr = "C:\\Users\\nimath\\switchdrive\\Institution\\PhD\\01_Experiments\\05_Prime_Editing\\03_DiseaseScreen\\05_Validations\\02_Arrayed\\Sequencing\\NGS\\"
controldf = pd.read_csv(path_ctr+'20220719_Editing_Endogenous_K562_control_deep.csv')
controldf = controldf.set_index('Unnamed: 0')
###


# create empty dataframes to be filled with editing-rates
editingdf_rep1 = pd.DataFrame()
editingdf_rep2 = pd.DataFrame()
indeldf_rep1 = pd.DataFrame()
indeldf_rep2 = pd.DataFrame()

for samplefolder in samplefolderlist_rep1:  # perform loop for first biological replicate
    bioreplicate = samplefolder[-1] # get biological replicate nr. from foldername
    techreplicate = samplefolder[-3] # get technical replicate nr. from foldername

    try:
        crispressodf = pd.read_csv(path_rep1+samplefolder+'\\CRISPResso_quantification_of_editing_frequency.txt', delimiter='\t')
        crispressodf = crispressodf.set_index('Amplicon') # set amplicon as index to access 'HDR' below
        # calculate percentage of HDR reads (correct edit) based on reads aligned to HDR vs. totalreads and store in 'Percentage' column:
        crispressodf['Percentage'] = crispressodf.apply(lambda x: (x.Unmodified/x.Reads_aligned_all_amplicons)*100 ,axis=1)
        indels = (sum(crispressodf.Modified)/crispressodf.Reads_aligned_all_amplicons.mean())*100
        
        if bioreplicate == '1':  # check if it is from biological replicate 1
            if techreplicate == '1': # check if it is from technical replicate 1
                if crispressodf.at['HDR','Reads_aligned_all_amplicons'] >500:
                    editingdf_rep1.loc[samplefolder[14:-5],'TR1'] = crispressodf.at['HDR','Percentage']
                    editingdf_rep1.loc[samplefolder[14:-5],'TR1_editedcount'] = crispressodf.at['HDR','Unmodified']
                    editingdf_rep1.loc[samplefolder[14:-5],'TR1_totalcount'] = crispressodf.at['HDR','Reads_aligned_all_amplicons']
                    indeldf_rep1.loc[samplefolder[14:-5],'TR1'] = indels
                    indeldf_rep1.loc[samplefolder[14:-5],'TR1_indelcount'] = sum(crispressodf.Modified)
                    indeldf_rep1.loc[samplefolder[14:-5],'TR1_totalcount'] = crispressodf.Reads_aligned_all_amplicons.HDR
                else:
                    editingdf_rep1.loc[samplefolder[14:-5],'TR1'] = None
                    editingdf_rep1.loc[samplefolder[14:-5],'TR1_editedcount'] = None
                    editingdf_rep1.loc[samplefolder[14:-5],'TR1_totalcount'] = None
                    indeldf_rep1.loc[samplefolder[14:-5],'TR1'] = None
                    indeldf_rep1.loc[samplefolder[14:-5],'TR1_indelcount'] = None
                    indeldf_rep1.loc[samplefolder[14:-5],'TR1_totalcount'] = None
                    
            elif techreplicate == '2': # check if it is from technical replicate 2
                if crispressodf.at['HDR','Reads_aligned_all_amplicons'] >500:
                    editingdf_rep1.loc[samplefolder[14:-5],'TR2'] = crispressodf.at['HDR','Percentage']
                    editingdf_rep1.loc[samplefolder[14:-5],'TR2_editedcount'] = crispressodf.at['HDR','Unmodified']
                    editingdf_rep1.loc[samplefolder[14:-5],'TR2_totalcount'] = crispressodf.at['HDR','Reads_aligned_all_amplicons']
                    indeldf_rep1.loc[samplefolder[14:-5],'TR2'] = indels
                    indeldf_rep1.loc[samplefolder[14:-5],'TR2_indelcount'] = sum(crispressodf.Modified)
                    indeldf_rep1.loc[samplefolder[14:-5],'TR2_totalcount'] = crispressodf.Reads_aligned_all_amplicons.HDR
                else:
                    editingdf_rep1.loc[samplefolder[14:-5],'TR2'] = None
                    editingdf_rep1.loc[samplefolder[14:-5],'TR2_editedcount'] = None
                    editingdf_rep1.loc[samplefolder[14:-5],'TR2_totalcount'] = None
                    indeldf_rep1.loc[samplefolder[14:-5],'TR2'] = None
                    indeldf_rep1.loc[samplefolder[14:-5],'TR2_indelcount'] = None
                    indeldf_rep1.loc[samplefolder[14:-5],'TR2_totalcount'] = None
        
    except FileNotFoundError:  # if CRISPresso could not generate a quantification file, skip this sample and add "None" as value to dataframe
        if bioreplicate == '2':
            if techreplicate == '1':
                editingdf_rep1.loc[samplefolder[14:-5],'TR1'] = None
                editingdf_rep1.loc[samplefolder[14:-5],'TR1_editedcount'] = None
                editingdf_rep1.loc[samplefolder[14:-5],'TR1_totalcount'] = None
                indeldf_rep1.loc[samplefolder[14:-5],'TR1'] = None
                indeldf_rep1.loc[samplefolder[14:-5],'TR1_indelcount'] = None
                indeldf_rep1.loc[samplefolder[14:-5],'TR1_totalcount'] = None
            elif techreplicate == '2':
                editingdf_rep1.loc[samplefolder[14:-5],'TR2'] = None
                editingdf_rep1.loc[samplefolder[14:-5],'TR2_editedcount'] = None
                editingdf_rep1.loc[samplefolder[14:-5],'TR2_totalcount'] = None
                indeldf_rep1.loc[samplefolder[14:-5],'TR2'] = None
                indeldf_rep1.loc[samplefolder[14:-5],'TR2_indelcount'] = None
                indeldf_rep1.loc[samplefolder[14:-5],'TR2_totalcount'] = None           


                
                
for samplefolder in samplefolderlist_rep2: # perform loop for second biological replicate
    bioreplicate = samplefolder[-1] # get biological replicate nr. from foldername
    techreplicate = samplefolder[-3] # get technical replicate nr. from foldername
    try:
        crispressodf = pd.read_csv(path_rep2+samplefolder+'\\CRISPResso_quantification_of_editing_frequency.txt', delimiter='\t')
        crispressodf = crispressodf.set_index('Amplicon') # set amplicon as index to access 'HDR' below
        # calculate percentage of HDR reads (correct edit) based on reads aligned to HDR vs. totalreads and store in 'Percentage' column:
        crispressodf['Percentage'] = crispressodf.apply(lambda x: (x.Unmodified/x.Reads_aligned_all_amplicons)*100 ,axis=1)
        indels = (sum(crispressodf.Modified)/crispressodf.Reads_aligned_all_amplicons.mean())*100

        if bioreplicate == '2':  # check if it is from biological replicate 1
            if techreplicate == '1': # check if it is from technical replicate 1
                if crispressodf.at['HDR','Reads_aligned_all_amplicons'] >500:
                    editingdf_rep2.loc[samplefolder[14:-5],'TR1'] = crispressodf.at['HDR','Percentage']
                    editingdf_rep2.loc[samplefolder[14:-5],'TR1_editedcount'] = crispressodf.at['HDR','Unmodified']
                    editingdf_rep2.loc[samplefolder[14:-5],'TR1_totalcount'] = crispressodf.at['HDR','Reads_aligned_all_amplicons']
                    indeldf_rep2.loc[samplefolder[14:-5],'TR1'] = indels
                    indeldf_rep2.loc[samplefolder[14:-5],'TR1_indelcount'] = sum(crispressodf.Modified)
                    indeldf_rep2.loc[samplefolder[14:-5],'TR1_totalcount'] = crispressodf.Reads_aligned_all_amplicons.HDR
                else:
                    editingdf_rep2.loc[samplefolder[14:-5],'TR1'] = None
                    editingdf_rep2.loc[samplefolder[14:-5],'TR1_editedcount'] = None
                    editingdf_rep2.loc[samplefolder[14:-5],'TR1_totalcount'] = None
                    indeldf_rep2.loc[samplefolder[14:-5],'TR1'] = None
                    indeldf_rep2.loc[samplefolder[14:-5],'TR1_indelcount'] = None
                    indeldf_rep2.loc[samplefolder[14:-5],'TR1_totalcount'] = None
                    
            elif techreplicate == '2': # check if it is from technical replicate 2
                if crispressodf.at['HDR','Reads_aligned_all_amplicons'] >500:
                    editingdf_rep2.loc[samplefolder[14:-5],'TR2'] = crispressodf.at['HDR','Percentage']
                    editingdf_rep2.loc[samplefolder[14:-5],'TR2_editedcount'] = crispressodf.at['HDR','Unmodified']
                    editingdf_rep2.loc[samplefolder[14:-5],'TR2_totalcount'] = crispressodf.at['HDR','Reads_aligned_all_amplicons']
                    indeldf_rep2.loc[samplefolder[14:-5],'TR2'] = indels
                    indeldf_rep2.loc[samplefolder[14:-5],'TR2_indelcount'] = sum(crispressodf.Modified)
                    indeldf_rep2.loc[samplefolder[14:-5],'TR2_totalcount'] = crispressodf.Reads_aligned_all_amplicons.HDR
                else:
                    editingdf_rep2.loc[samplefolder[14:-5],'TR2'] = None
                    editingdf_rep2.loc[samplefolder[14:-5],'TR2_editedcount'] = None
                    editingdf_rep2.loc[samplefolder[14:-5],'TR2_totalcount'] = None
                    indeldf_rep2.loc[samplefolder[14:-5],'TR2'] = None
                    indeldf_rep2.loc[samplefolder[14:-5],'TR2_indelcount'] = None
                    indeldf_rep2.loc[samplefolder[14:-5],'TR2_totalcount'] = None
        
    except FileNotFoundError:  # if CRISPresso could not generate a quantification file, skip this sample and add "None" as value to dataframe
        if bioreplicate == '2':
            if techreplicate == '1':
                editingdf_rep2.loc[samplefolder[14:-5],'TR1'] = None
                editingdf_rep2.loc[samplefolder[14:-5],'TR1_editedcount'] = None
                editingdf_rep2.loc[samplefolder[14:-5],'TR1_totalcount'] = None
                indeldf_rep2.loc[samplefolder[14:-5],'TR1'] = None
                indeldf_rep2.loc[samplefolder[14:-5],'TR1_indelcount'] = None
                indeldf_rep2.loc[samplefolder[14:-5],'TR1_totalcount'] = None
            elif techreplicate == '2':
                editingdf_rep2.loc[samplefolder[14:-5],'TR2'] = None
                editingdf_rep2.loc[samplefolder[14:-5],'TR2_editedcount'] = None
                editingdf_rep2.loc[samplefolder[14:-5],'TR2_totalcount'] = None
                indeldf_rep2.loc[samplefolder[14:-5],'TR2'] = None
                indeldf_rep2.loc[samplefolder[14:-5],'TR2_indelcount'] = None
                indeldf_rep2.loc[samplefolder[14:-5],'TR2_totalcount'] = None           

for index, row in editingdf_rep1.iterrows():
    try:
        backgroundefficiency = controldf.at[index,'average_between_bio_replicates']
    except KeyError:  # set backgroundefficiency to 0 if no ctrl can be found
        backgroundefficiency = 0 
    editingdf_rep1.at[index,'TR1_corrected'] = ((row.TR1_editedcount-(row.TR1_totalcount*backgroundefficiency)/100)/(row.TR1_totalcount-(row.TR1_totalcount*backgroundefficiency)/100))*100
    editingdf_rep1.at[index,'TR2_corrected'] = ((row.TR2_editedcount-(row.TR2_totalcount*backgroundefficiency)/100)/(row.TR2_totalcount-(row.TR2_totalcount*backgroundefficiency)/100))*100

for index, row in editingdf_rep2.iterrows():
    try:
        backgroundefficiency = controldf.at[index,'average_between_bio_replicates']
    except KeyError:  # set backgroundefficiency to 0 if no ctrl can be found
        backgroundefficiency = 0 
    editingdf_rep2.at[index,'TR1_corrected'] = ((row.TR1_editedcount-(row.TR1_totalcount*backgroundefficiency)/100)/(row.TR1_totalcount-(row.TR1_totalcount*backgroundefficiency)/100))*100
    editingdf_rep2.at[index,'TR2_corrected'] = ((row.TR2_editedcount-(row.TR2_totalcount*backgroundefficiency)/100)/(row.TR2_totalcount-(row.TR2_totalcount*backgroundefficiency)/100))*100

for index, row in indeldf_rep1.iterrows():
    try:
        backgroundefficiency = controldf.at[index,'average_indel_between_bio_replicates']
    except KeyError:  # set backgroundefficiency to 0 if no ctrl can be found
        backgroundefficiency = 0 
    indeldf_rep1.at[index,'TR1_corrected'] = ((row.TR1_indelcount-(row.TR1_totalcount*backgroundefficiency)/100)/(row.TR1_totalcount-(row.TR1_totalcount*backgroundefficiency)/100))*100
    indeldf_rep1.at[index,'TR2_corrected'] = ((row.TR2_indelcount-(row.TR2_totalcount*backgroundefficiency)/100)/(row.TR2_totalcount-(row.TR2_totalcount*backgroundefficiency)/100))*100

for index, row in indeldf_rep2.iterrows():
    try:
        backgroundefficiency = controldf.at[index,'average_indel_between_bio_replicates']
    except KeyError:  # set backgroundefficiency to 0 if no ctrl can be found
        backgroundefficiency = 0 
    indeldf_rep2.at[index,'TR1_corrected'] = ((row.TR1_indelcount-(row.TR1_totalcount*backgroundefficiency)/100)/(row.TR1_totalcount-(row.TR1_totalcount*backgroundefficiency)/100))*100
    indeldf_rep2.at[index,'TR2_corrected'] = ((row.TR2_indelcount-(row.TR2_totalcount*backgroundefficiency)/100)/(row.TR2_totalcount-(row.TR2_totalcount*backgroundefficiency)/100))*100

editingdf_rep1_short= editingdf_rep1[['TR1_corrected','TR2_corrected']]
editingdf_rep2_short= editingdf_rep2[['TR1_corrected','TR2_corrected']]
indeldf_rep1_short= indeldf_rep1[['TR1_corrected','TR2_corrected']]
indeldf_rep2_short= indeldf_rep2[['TR1_corrected','TR2_corrected']]

# calculate mean of the technical replicates and store in average column:
editingdf_rep1['rep1_average'] = editingdf_rep1_short.mean(axis=1, skipna=True)
editingdf_rep2['rep2_average'] = editingdf_rep2_short.mean(axis=1, skipna=True)
indeldf_rep1['rep1_average'] = indeldf_rep1_short.mean(axis=1, skipna=True)
indeldf_rep2['rep2_average'] = indeldf_rep2_short.mean(axis=1, skipna=True)


summaryeditingdf = pd.DataFrame()
summaryindeldf = pd.DataFrame()
summarydf = pd.DataFrame()
summarydf['rep1_average'] = editingdf_rep1['rep1_average']
summarydf['rep2_average'] = editingdf_rep2['rep2_average']
summaryeditingdf['rep1_average'] = editingdf_rep1['rep1_average']
summaryeditingdf['rep2_average'] = editingdf_rep2['rep2_average']


summarydf['rep1_average_indel'] = indeldf_rep1['rep1_average']
summarydf['rep2_average_indel'] = indeldf_rep2['rep2_average']
summaryindeldf['rep1_average_indel'] = indeldf_rep1['rep1_average']
summaryindeldf['rep2_average_indel'] = indeldf_rep2['rep2_average']
summarydf['average_between_bio_replicates'] = summaryeditingdf.mean(axis=1, skipna=True)
summarydf['average_indel_between_bio_replicates'] = summaryindeldf.mean(axis=1, skipna=True)
summarydf['library_editing_rates'] = libraryedits['averageedited']
summarydf['PRIDICT_Score'] = librarypredictionfinal['PRIDICT_Score']
summarydf['locus'] = summarydf.index
summarydf['locus'] = summarydf['locus'].apply(lambda x: x[:2])
uniquelocuslist = summarydf['locus'].unique()

endovslibrarydf = summarydf[['average_between_bio_replicates','average_indel_between_bio_replicates','library_editing_rates', 'PRIDICT_Score']].dropna()


# clamp values to be within 0 and 100
summarydf['average_between_bio_replicates'] = summarydf['average_between_bio_replicates'].apply(lambda x: 0 if x < 0 else x)
summarydf['average_indel_between_bio_replicates'] = summarydf['average_indel_between_bio_replicates'].apply(lambda x: 0 if x < 0 else x)


librarypredictionfinal['edit_efficiency'] = summarydf['average_between_bio_replicates']

summarydf.to_csv('20220719_Editing_Endogenous_K562_deep_new.csv')
