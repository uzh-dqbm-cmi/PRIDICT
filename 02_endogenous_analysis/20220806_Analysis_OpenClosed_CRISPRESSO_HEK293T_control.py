# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:47:50 2021

@author: nimath
"""

import pandas as pd
import os

path = 'C:\\Users\\nimath\\switchdrive\\Institution\\PhD\\01_Experiments\\05_Prime_Editing\\03_DiseaseScreen\\05_Validations\\02_Arrayed\\Open_Closed_Regions\\'

openclosedinfodf = pd.read_csv(path+'20220728_OpenClosed_Infodataframe.csv')

replicatelist = ['rep1','rep2','rep3']
replicatepathdict = {'rep1':'20220726_Rep1\\CRISPRessoBatch_on_20220729_CRISPResso_Batchfile_Rep1.batch\\','rep2':'20220805_Rep2\\CRISPRessoBatch_on_20220806_CRISPResso_Batchfile_Rep2.batch\\','rep3':'20220805_Rep3\\CRISPRessoBatch_on_20220806_CRISPResso_Batchfile_Rep3.batch\\',}

summaryeditingdf = pd.DataFrame()
summaryindeldf = pd.DataFrame()
summarydf = pd.DataFrame()

for repl in replicatelist:  # loop through all 3 replicates
    path_rep = "C:\\Users\\nimath\\switchdrive\\Institution\\PhD\\01_Experiments\\05_Prime_Editing\\03_DiseaseScreen\\05_Validations\\02_Arrayed\\Open_Closed_Regions\\NGS\\" + replicatepathdict[repl]
    samplefolderlist_rep = [f for f in os.listdir(path_rep) if (not "." in f) and ('Ctrl' in f)]  # make list with subfolders corresponding to samples of Replicate 1
    
    # create empty dataframes to be filled with editing-rates
    editingdf_rep = pd.DataFrame()
    indeldf_rep = pd.DataFrame()
    
    for samplefolder in samplefolderlist_rep:  # perform loop for first biological replicate
        bioreplicate = '1'
        techreplicate = '1'
        # print(path_rep+samplefolder)
        try:
            crispressodf = pd.read_csv(path_rep+samplefolder+'\\CRISPResso_quantification_of_editing_frequency.txt', delimiter='\t')
            crispressodf = crispressodf.set_index('Amplicon') # set amplicon as index to access 'HDR' below
            # calculate percentage of HDR reads (correct edit) based on reads aligned to HDR vs. totalreads and store in 'Percentage' column:
            crispressodf['Percentage'] = crispressodf.apply(lambda x: (x.Unmodified/x.Reads_aligned_all_amplicons)*100 ,axis=1)
            indels = (sum(crispressodf.Modified)/crispressodf.Reads_aligned_all_amplicons.mean())*100
            
            if bioreplicate == '1':  # check if it is from biological replicate 1
                if techreplicate == '1': # check if it is from technical replicate 1
                    if crispressodf.at['HDR','Reads_aligned_all_amplicons'] >500:
                        editingdf_rep.loc[samplefolder[14:-5],'TR1'] = crispressodf.at['HDR','Percentage']
                        indeldf_rep.loc[samplefolder[14:-5],'TR1'] = indels
                    else:
                        editingdf_rep.loc[samplefolder[14:-5],'TR1'] = None
                        indeldf_rep.loc[samplefolder[14:-5],'TR1'] = None
                elif techreplicate == '2': # check if it is from technical replicate 2
                    if crispressodf.at['HDR','Reads_aligned_all_amplicons'] >500:
                        editingdf_rep.loc[samplefolder[14:-5],'TR2'] = crispressodf.at['HDR','Percentage']
                        indeldf_rep.loc[samplefolder[14:-5],'TR2'] = indels
                    else:
                        editingdf_rep.loc[samplefolder[14:-5],'TR2'] = None
                        indeldf_rep.loc[samplefolder[14:-5],'TR2'] = None
    
            
        except FileNotFoundError:  # if CRISPresso could not generate a quantification file, skip this sample and add "None" as value to dataframe
            if bioreplicate == '1':
                if techreplicate == '1':
                    editingdf_rep.loc[samplefolder[14:-5],'TR1'] = None
                    indeldf_rep.loc[samplefolder[14:-5],'TR1'] = None
                elif techreplicate == '2':
                    editingdf_rep.loc[samplefolder[14:-5],'TR2'] = None
                    indeldf_rep.loc[samplefolder[14:-5],'TR2'] = None
    
                             
    # calculate mean of the technical replicates and store in average column:
    editingdf_rep[repl+'_average'] = editingdf_rep.mean(axis=1, skipna=True)
    indeldf_rep[repl+'_average'] = indeldf_rep.mean(axis=1, skipna=True)
    
    
    summarydf[repl+'_average'] = editingdf_rep[repl+'_average']
    summaryeditingdf[repl+'_average'] = editingdf_rep[repl+'_average']
    summarydf[repl+'_average_indel'] = indeldf_rep[repl+'_average']
    summaryindeldf[repl+'_average_indel'] = indeldf_rep[repl+'_average']




summarydf['average_between_bio_replicates'] = summaryeditingdf.mean(axis=1, skipna=True)
summarydf['average_indel_between_bio_replicates'] = summaryindeldf.mean(axis=1, skipna=True)
summarydf['locus'] = summarydf.index
# summarydf['locus'] = summarydf['locus'].apply(lambda x: x[:2])
summarydf = summarydf.dropna()
uniquelocuslist = summarydf['locus'].unique()


summarydf.to_csv('20220806_Editing_OpenClosed_HEK293T_control_rep.csv')
