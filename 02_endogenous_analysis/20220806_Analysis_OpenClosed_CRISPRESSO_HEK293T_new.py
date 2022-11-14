# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:47:50 2021

@author: nimath
"""

import pandas as pd
import os

path = 'C:\\Users\\nimath\\switchdrive\\Institution\\PhD\\01_Experiments\\05_Prime_Editing\\03_DiseaseScreen\\05_Validations\\02_Arrayed\\Open_Closed_Regions\\'

# add correct indices to openclosedinfodf to add it in the end to summarydf
openclosedinfodf = pd.read_csv(path+'20220728_OpenClosed_Infodataframe.csv')
openclosedinfodf = openclosedinfodf.set_index('Shortname')

### Control editing rates:
path_ctr = "C:\\Users\\nimath\\switchdrive\\Institution\\PhD\\01_Experiments\\05_Prime_Editing\\03_DiseaseScreen\\05_Validations\\02_Arrayed\\Open_Closed_Regions\\NGS\\"
controldf = pd.read_csv(path_ctr+'20220806_Editing_OpenClosed_HEK293T_control_rep.csv')
controldf['realindex'] = controldf['Unnamed: 0'].apply(lambda x: x.split('_')[1]+x.split('_')[0]+'NM')
controldf = controldf.set_index('realindex')
###

summaryeditingdf = pd.DataFrame()
summaryindeldf = pd.DataFrame()
summarydf = pd.DataFrame()

replicatelist = ['rep1','rep2','rep3']
replicatepathdict = {'rep1':'20220726_Rep1\\CRISPRessoBatch_on_20220729_CRISPResso_Batchfile_Rep1.batch\\','rep2':'20220805_Rep2\\CRISPRessoBatch_on_20220806_CRISPResso_Batchfile_Rep2.batch\\','rep3':'20220805_Rep3\\CRISPRessoBatch_on_20220806_CRISPResso_Batchfile_Rep3.batch\\',}

for repl in replicatelist:  # loop through all 3 replicates
    path_rep = "C:\\Users\\nimath\\switchdrive\\Institution\\PhD\\01_Experiments\\05_Prime_Editing\\03_DiseaseScreen\\05_Validations\\02_Arrayed\\Open_Closed_Regions\\NGS\\" + replicatepathdict[repl]
    samplefolderlist_rep = [f for f in os.listdir(path_rep) if (not "." in f) and (not 'Ctrl' in f)]  # make list with subfolders corresponding to samples of Replicate 1
    
    # create empty dataframes to be filled with editing-rates
    editingdf_rep = pd.DataFrame()
    indeldf_rep = pd.DataFrame()
    
    for samplefolder in samplefolderlist_rep:  # perform loop for first biological replicate
        techreplicate = samplefolder[-5:-4] # get technical replicate nr. from foldername
        if techreplicate == 'A':
            techreplicate = '1'
        elif techreplicate == 'B':
            techreplicate = '2'
    
        try:
            crispressodf = pd.read_csv(path_rep+samplefolder+'\\CRISPResso_quantification_of_editing_frequency.txt', delimiter='\t')
            crispressodf = crispressodf.set_index('Amplicon') # set amplicon as index to access 'HDR' below
            
            # calculate percentage of HDR reads (correct edit) based on reads aligned to HDR vs. totalreads and store in 'Percentage' column:
            crispressodf['Percentage'] = crispressodf.apply(lambda x: (x.Unmodified/x.Reads_aligned_all_amplicons)*100 ,axis=1)
            indels = (sum(crispressodf.Modified)/crispressodf.Reads_aligned_all_amplicons.mean())*100
            
            if techreplicate == '1': # check if it is from technical replicate 1
                if crispressodf.at['HDR','Reads_aligned_all_amplicons'] >500:
                    editingdf_rep.loc[samplefolder[14:-5],'TR1'] = crispressodf.at['HDR','Percentage']
                    editingdf_rep.loc[samplefolder[14:-5],'TR1_editedcount'] = crispressodf.at['HDR','Unmodified']
                    editingdf_rep.loc[samplefolder[14:-5],'TR1_totalcount'] = crispressodf.at['HDR','Reads_aligned_all_amplicons']
                    indeldf_rep.loc[samplefolder[14:-5],'TR1'] = indels
                    indeldf_rep.loc[samplefolder[14:-5],'TR1_indelcount'] = sum(crispressodf.Modified)
                    indeldf_rep.loc[samplefolder[14:-5],'TR1_totalcount'] = crispressodf.Reads_aligned_all_amplicons.HDR
                else:
                    editingdf_rep.loc[samplefolder[14:-5],'TR1'] = None
                    editingdf_rep.loc[samplefolder[14:-5],'TR1_editedcount'] = None
                    editingdf_rep.loc[samplefolder[14:-5],'TR1_totalcount'] = None
                    indeldf_rep.loc[samplefolder[14:-5],'TR1'] = None
                    indeldf_rep.loc[samplefolder[14:-5],'TR1_indelcount'] = None
                    indeldf_rep.loc[samplefolder[14:-5],'TR1_totalcount'] = None
                    
            elif techreplicate == '2': # check if it is from technical replicate 2
                if crispressodf.at['HDR','Reads_aligned_all_amplicons'] >500:
                    editingdf_rep.loc[samplefolder[14:-5],'TR2'] = crispressodf.at['HDR','Percentage']
                    editingdf_rep.loc[samplefolder[14:-5],'TR2_editedcount'] = crispressodf.at['HDR','Unmodified']
                    editingdf_rep.loc[samplefolder[14:-5],'TR2_totalcount'] = crispressodf.at['HDR','Reads_aligned_all_amplicons']
                    indeldf_rep.loc[samplefolder[14:-5],'TR2'] = indels
                    indeldf_rep.loc[samplefolder[14:-5],'TR2_indelcount'] = sum(crispressodf.Modified)
                    indeldf_rep.loc[samplefolder[14:-5],'TR2_totalcount'] = crispressodf.Reads_aligned_all_amplicons.HDR
                else:
                    editingdf_rep.loc[samplefolder[14:-5],'TR2'] = None
                    editingdf_rep.loc[samplefolder[14:-5],'TR2_editedcount'] = None
                    editingdf_rep.loc[samplefolder[14:-5],'TR2_totalcount'] = None
                    indeldf_rep.loc[samplefolder[14:-5],'TR2'] = None
                    indeldf_rep.loc[samplefolder[14:-5],'TR2_indelcount'] = None
                    indeldf_rep.loc[samplefolder[14:-5],'TR2_totalcount'] = None
            
        except FileNotFoundError:  # if CRISPresso could not generate a quantification file, skip this sample and add "None" as value to dataframe

            if techreplicate == '1':
                editingdf_rep.loc[samplefolder[14:-5],'TR1'] = None
                editingdf_rep.loc[samplefolder[14:-5],'TR1_editedcount'] = None
                editingdf_rep.loc[samplefolder[14:-5],'TR1_totalcount'] = None
                indeldf_rep.loc[samplefolder[14:-5],'TR1'] = None
                indeldf_rep.loc[samplefolder[14:-5],'TR1_indelcount'] = None
                indeldf_rep.loc[samplefolder[14:-5],'TR1_totalcount'] = None
            elif techreplicate == '2':
                editingdf_rep.loc[samplefolder[14:-5],'TR2'] = None
                editingdf_rep.loc[samplefolder[14:-5],'TR2_editedcount'] = None
                editingdf_rep.loc[samplefolder[14:-5],'TR2_totalcount'] = None
                indeldf_rep.loc[samplefolder[14:-5],'TR2'] = None
                indeldf_rep.loc[samplefolder[14:-5],'TR2_indelcount'] = None
                indeldf_rep.loc[samplefolder[14:-5],'TR2_totalcount'] = None           
    
    
    for index, row in editingdf_rep.iterrows():
        try:
            backgroundefficiency = controldf.at[index,'average_between_bio_replicates']
        except KeyError:  # set backgroundefficiency to 0 if no ctrl can be found
            backgroundefficiency = 0 
        editingdf_rep.at[index,'TR1_corrected'] = ((row.TR1_editedcount-(row.TR1_totalcount*backgroundefficiency)/100)/(row.TR1_totalcount-(row.TR1_totalcount*backgroundefficiency)/100))*100
        editingdf_rep.at[index,'TR2_corrected'] = ((row.TR2_editedcount-(row.TR2_totalcount*backgroundefficiency)/100)/(row.TR2_totalcount-(row.TR2_totalcount*backgroundefficiency)/100))*100
    
    
    for index, row in indeldf_rep.iterrows():
        try:
            backgroundefficiency = controldf.at[index,'average_indel_between_bio_replicates']
        except KeyError:  # set backgroundefficiency to 0 if no ctrl can be found
            backgroundefficiency = 0 
        indeldf_rep.at[index,'TR1_corrected'] = ((row.TR1_indelcount-(row.TR1_totalcount*backgroundefficiency)/100)/(row.TR1_totalcount-(row.TR1_totalcount*backgroundefficiency)/100))*100
        indeldf_rep.at[index,'TR2_corrected'] = ((row.TR2_indelcount-(row.TR2_totalcount*backgroundefficiency)/100)/(row.TR2_totalcount-(row.TR2_totalcount*backgroundefficiency)/100))*100
                                                                                                  
                                                                                                                   
    editingdf_rep_short= editingdf_rep[['TR1_corrected','TR2_corrected']]
    indeldf_rep_short= indeldf_rep[['TR1_corrected','TR2_corrected']]
    
    # calculate mean of the technical replicates and store in average column:
    editingdf_rep[repl+'_average'] = editingdf_rep_short.mean(axis=1, skipna=True)
    indeldf_rep[repl+'_average'] = indeldf_rep_short.mean(axis=1, skipna=True)
    
    summarydf[repl+'_average'] = editingdf_rep[repl+'_average']
    summaryeditingdf[repl+'_average'] = editingdf_rep[repl+'_average']
    
    summarydf[repl+'_average_indel'] = indeldf_rep[repl+'_average']
    summaryindeldf[repl+'_average_indel'] = indeldf_rep[repl+'_average']


summarydf['average_between_bio_replicates'] = summaryeditingdf.mean(axis=1, skipna=True)
summarydf['average_indel_between_bio_replicates'] = summaryindeldf.mean(axis=1, skipna=True)

summarydf['pegRNA'] = summarydf.index
summarydf['context'] = summarydf.index
summarydf['context'] = summarydf['context'].apply(lambda x: x[:-5])
summarydf['pegRNA'] = summarydf['pegRNA'].apply(lambda x: x[-5:])

# clamp values to be within 0 and 100
summarydf['average_between_bio_replicates'] = summarydf['average_between_bio_replicates'].apply(lambda x: 0 if x < 0 else x)
summarydf['average_indel_between_bio_replicates'] = summarydf['average_indel_between_bio_replicates'].apply(lambda x: 0 if x < 0 else x)

uniquelocuslist = summarydf['pegRNA'].unique()

endosummarydf = summarydf[summarydf['context'] == 'endo']
endosummarydf = endosummarydf.set_index('pegRNA')
integsummarydf = summarydf[summarydf['context'] == 'integ']
integsummarydf = integsummarydf.set_index('pegRNA')

cols = ['rep1_average', 'rep1_average_indel','rep2_average', 'rep2_average_indel','rep3_average', 'rep3_average_indel', 'average_between_bio_replicates',
       'average_indel_between_bio_replicates']

for col in cols:
    endosummarydf[col+'_endo'] = endosummarydf[col]
    endosummarydf[col+'_integ'] = integsummarydf[col]

cols = ['context', 'rep1_average_endo',
       'rep1_average_integ', 'rep1_average_indel_endo',
       'rep1_average_indel_integ','rep2_average_endo',
       'rep2_average_integ', 'rep2_average_indel_endo',
       'rep2_average_indel_integ','rep3_average_endo',
       'rep3_average_integ', 'rep3_average_indel_endo',
       'rep3_average_indel_integ', 'average_between_bio_replicates_endo',
       'average_between_bio_replicates_integ',
       'average_indel_between_bio_replicates_endo',
       'average_indel_between_bio_replicates_integ']

summarydf = endosummarydf[cols].copy()

summarydf['ratio_integ_endo'] = summarydf.apply(lambda x: x.average_between_bio_replicates_integ/x.average_between_bio_replicates_endo,axis=1)

cols = ['Storagename', 'Name', 'Chromatin_state',
        'DNase I RPKM HEK293T', 'H3K4me3 RPKM HEK293T', '% CpGme HEK293T',
        'CTCF RPKM HEK293T', 'RNA seq RPKM HEK293T']
for col in cols:
    summarydf[col] = openclosedinfodf[col]

summarydf.to_csv('20220806_Editing_OpenClosed_HEK293T.csv')
