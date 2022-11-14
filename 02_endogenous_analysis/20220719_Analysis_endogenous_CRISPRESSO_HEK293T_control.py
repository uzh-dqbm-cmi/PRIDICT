# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:47:50 2021

@author: nimath
"""

import pandas as pd
import os

path_arrayed = 'C:\\Users\\nimath\\switchdrive\\Institution\\PhD\\01_Experiments\\05_Prime_Editing\\03_DiseaseScreen\\05_Validations\\02_Arrayed\\'
libraryedits = pd.read_csv(path_arrayed+'20210714_Arrayed_Kimetal_Order_Filtered.csv')
libraryedits['corresponding_endogenous_editname'] = libraryedits['Shortname'].apply(lambda x: x+'libendopeg')
libraryedits = libraryedits.set_index('corresponding_endogenous_editname')

librarypredictioneditsall = pd.read_csv(path_arrayed+'20210907_all_endopegRNAs_with_features.csv')
librarypredictioneditsall = librarypredictioneditsall.set_index('pegRNA',drop=False)

ordereddf =  pd.read_csv(path_arrayed+'20210718_Gblock_Ordering_Twist_FINAL_adapted1and17best.csv')


path_rep1 = "C:\\Users\\nimath\\switchdrive\\Institution\\PhD\\01_Experiments\\05_Prime_Editing\\03_DiseaseScreen\\05_Validations\\02_Arrayed\\Sequencing\\NGS\\20211202_Background_Editing_HEK\\"
path_rep1 = path_rep1 + "FASTQ\\CRISPRessoBatch_on_endogenous_HEK293T_control_updated\\"


samplefolderlist_rep1 = [f for f in os.listdir(path_rep1) if not "." in f]  # make list with subfolders corresponding to samples of Replicate 1

# create empty dataframes to be filled with editing-rates
editingdf_rep1 = pd.DataFrame()
indeldf_rep1 = pd.DataFrame()

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
                    indeldf_rep1.loc[samplefolder[14:-5],'TR1'] = indels
                else:
                    editingdf_rep1.loc[samplefolder[14:-5],'TR1'] = None
                    indeldf_rep1.loc[samplefolder[14:-5],'TR1'] = None
            elif techreplicate == '2': # check if it is from technical replicate 2
                if crispressodf.at['HDR','Reads_aligned_all_amplicons'] >500:
                    editingdf_rep1.loc[samplefolder[14:-5],'TR2'] = crispressodf.at['HDR','Percentage']
                    indeldf_rep1.loc[samplefolder[14:-5],'TR2'] = indels
                else:
                    editingdf_rep1.loc[samplefolder[14:-5],'TR2'] = None
                    indeldf_rep1.loc[samplefolder[14:-5],'TR2'] = None

        
    except FileNotFoundError:  # if CRISPresso could not generate a quantification file, skip this sample and add "None" as value to dataframe
        if bioreplicate == '1':
            if techreplicate == '1':
                editingdf_rep1.loc[samplefolder[14:-5],'TR1'] = None
                indeldf_rep1.loc[samplefolder[14:-5],'TR1'] = None
            elif techreplicate == '2':
                editingdf_rep1.loc[samplefolder[14:-5],'TR2'] = None
                indeldf_rep1.loc[samplefolder[14:-5],'TR2'] = None

                         
# calculate mean of the technical replicates and store in average column:
editingdf_rep1['rep1_average'] = editingdf_rep1.mean(axis=1, skipna=True)
indeldf_rep1['rep1_average'] = indeldf_rep1.mean(axis=1, skipna=True)

summaryeditingdf = pd.DataFrame()
summaryindeldf = pd.DataFrame()
summarydf = pd.DataFrame()
summarydf['rep1_average'] = editingdf_rep1['rep1_average']
summaryeditingdf['rep1_average'] = editingdf_rep1['rep1_average']



summarydf['rep1_average_indel'] = indeldf_rep1['rep1_average']
summaryindeldf['rep1_average_indel'] = indeldf_rep1['rep1_average']
summarydf['average_between_bio_replicates'] = summaryeditingdf.mean(axis=1, skipna=True)
summarydf['average_indel_between_bio_replicates'] = summaryindeldf.mean(axis=1, skipna=True)
summarydf['locus'] = summarydf.index
summarydf['locus'] = summarydf['locus'].apply(lambda x: x[:2])
summarydf = summarydf.dropna()
uniquelocuslist = summarydf['locus'].unique()


summarydf.to_csv('20220719_Editing_Endogenous_HEK293T_control_deep.csv')
