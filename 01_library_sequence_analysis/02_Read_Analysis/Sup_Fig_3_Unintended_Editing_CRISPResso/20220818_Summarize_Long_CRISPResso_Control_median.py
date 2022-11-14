# -*- coding: utf-8 -*-


import pandas as pd
from os import listdir
import os
import numpy as np


def list_files1(directory):
    return [f for f in listdir(directory) if ('CRISPResso_on_' in f) and not ('.html' in f)]

cwd = os.getcwd()

# for Windows, replace the '/' with '//'
cwd = cwd.replace('\\','/')
cwd = cwd.split('/')[-1]
path = './'
filelist = list_files1(path)
variantlist = [x[14:] for x in filelist]

summarydf = pd.DataFrame()

# Create empty lists to be filled for generating dataframe
readsalignedlist = []
readsalignedtotallist = []
readsaligned_total_referencelist = []
readsaligned_total_hdrlist = []
readsaligned_Unmodified_referencelist = []
readsaligned_Unmodified_hdrlist = []
readsaligned_modified_referencelist = []
readsaligned_modified_hdrlist = []
readsaligned_insertions_referencelist = []
readsaligned_insertions_hdrlist = []
readsaligned_deletions_referencelist = []
readsaligned_deletions_hdrlist = []
readsaligned_substitutions_referencelist = []
readsaligned_substitutions_hdrlist = []
finalvariantlist = []
hdrnickmutationlist = []
hdrflapmutationlist = []
hdrnickinsertionlist = []
hdrflapinsertionlist = []
hdrnickdeletionlist = []
hdrflapdeletionlist = []
hdrnicksubstitutionlist = []
hdrflapsubstitutionlist = []
referencenickmutationlist = []
referenceflapmutationlist = []
referencenickinsertionlist = []
referenceflapinsertionlist = []
referencenickdeletionlist = []
referenceflapdeletionlist = []
referencenicksubstitutionlist = []
referenceflapsubstitutionlist = []
nickcolumnlist = []
flapcolumnlist = []

hdrrttinsertionlist = []
hdrrttdeletionlist = []
hdrrttsubstitutionlist = []
hdrrttmutationlist = []

referencerttinsertionlist = []
referencerttdeletionlist = []
referencerttsubstitutionlist = []
referencerttmutationlist = []

hdrpbsinsertionlist = []
hdrpbsdeletionlist = []
hdrpbssubstitutionlist = []
hdrpbsmutationlist = []

referencepbsinsertionlist = []
referencepbsdeletionlist = []
referencepbssubstitutionlist = []
referencepbsmutationlist = []

fiveprimetrim = 10
start = 23 - fiveprimetrim  # start 3 bp before nick
templatedf = pd.read_csv('20210527_templatefiltered.csv')
templatedf['amplicon_seq'] = templatedf['wide_initial_target'].apply(lambda x: x[fiveprimetrim:61])  # crop sequence to 70 bp since input is only 70 bp
templatedf['expected_hdr_amplicon_seq'] = templatedf['wide_mutated_target'].apply(lambda x: x[fiveprimetrim:61])

for index, row in templatedf.iterrows():
    if row.Correction_Type == 'Replacement':
        
        flap_end = start+3+row.RTlength
        templatedf.at[index,'flap_end_reference'] = flap_end
        templatedf.at[index,'flap_end_hdr'] = flap_end

    elif row.Correction_Type == 'Deletion':
        flap_end = start+3+row.RTlength+row.Correction_Length
        templatedf.at[index,'flap_end_reference'] = flap_end
        templatedf.at[index,'flap_end_hdr'] = flap_end - row.Correction_Length
        templatedf.at[index,'expected_hdr_amplicon_seq'] = templatedf.at[index,'expected_hdr_amplicon_seq'][:-row.Correction_Length]
        
    elif row.Correction_Type == 'Insertion':
        flap_end = start+3+row.RTlength-row.Correction_Length
        templatedf.at[index,'flap_end_reference'] = flap_end
        templatedf.at[index,'flap_end_hdr'] = flap_end + row.Correction_Length
        templatedf.at[index,'amplicon_seq'] = templatedf.at[index,'amplicon_seq'][:-row.Correction_Length]
        

ind = 0
indtemp = 0
for variant in variantlist:
    pospercentage = [0,0,0,0]
    flappospercentage = [0,0,0,0,0,0,0]
    rttlisttemp = [None]*templatedf.RTlength.max() # create array to fit all RTlengths 
    
    try:  # check if quantification file is available for analysis, otherwise go to next variant
        editingquantfile = pd.read_table(path+'CRISPResso_on_'+variant+'/CRISPResso_quantification_of_editing_frequency.txt')
        nulceotidepercentagedf = pd.read_table(path+'CRISPResso_on_'+variant+'/'+'Reference'+'.Nucleotide_percentage_table.txt')
        nucleotidecolumns = list(nulceotidepercentagedf.columns)
        flapend = int(templatedf.at[int(variant),'flap_end_reference'])
        if len(nucleotidecolumns) < flapend+5:  # skip variants where reference sequence is too short to visualize 5 bases after flap end
                continue
 
    except FileNotFoundError:
        # skip variants where CRISPResso didn't create the needed file
        continue
    
    for amptype in ['HDR','Reference']:
        nicklist = [pospercentage.copy(),pospercentage.copy(),pospercentage.copy(),pospercentage.copy()]
        flaplist = [flappospercentage.copy(),flappospercentage.copy(),flappospercentage.copy(),flappospercentage.copy()]
        rttlist = [rttlisttemp.copy(),rttlisttemp.copy(),rttlisttemp.copy(),rttlisttemp.copy()]
        rttlist_empty = rttlist.copy()
        pbstemp = [None]*11
        pbslist = [pbstemp.copy(),pbstemp.copy(),pbstemp.copy(),pbstemp.copy()]
        pbslist_empty = pbslist.copy()
        try:
            nulceotidepercentagedf = pd.read_table(path+'CRISPResso_on_'+variant+'/'+amptype+'.Nucleotide_percentage_table.txt')
            nulceotidepercentagedf = nulceotidepercentagedf.set_index('Unnamed: 0', drop=True)
            flapend = int(templatedf.at[int(variant),'flap_end_'+amptype.lower()])
            nucleotidecolumns = list(nulceotidepercentagedf.columns)
            nickcolumns = nucleotidecolumns[start+1:start+5]
            flapcolumns = nucleotidecolumns[flapend-2:flapend+5]
            pbscolumns = nucleotidecolumns[start-10:start+1]
            rttcolumns = nucleotidecolumns[start+3+2:flapend-2]  # start 2bp after RTT start and end 2bp before RTT end since these bases are covered by nick and flapcolumns already
            scaffoldcolumns = nucleotidecolumns[flapend:]
            
            modificationcountdf = pd.read_table(path+'CRISPResso_on_'+variant+'/'+amptype+'.Modification_count_vectors.txt')
            modificationcountdf = modificationcountdf.set_index('Sequence')
            
            # shift Insertion_Left row by 1 column to get Insertion_Right:
            insrightdf = modificationcountdf[modificationcountdf.index == 'Insertions_Left'].shift(axis=1)
            modificationcountdf.loc['Insertions_Right',:] = insrightdf.loc['Insertions_Left']
            
            modlist = ['Insertions_Right','Deletions','Substitutions','All_modifications']
            modificationcountdf.loc['All_modifications',:] = modificationcountdf.loc[modlist[:-1],:].sum(axis=0)  # calculate all_modifications freshly based on insertions_right values (instead of insertions_left)
            
            
            for index, modtype in enumerate(modlist):
                for posind, pos in enumerate(nickcolumns):
    
                    allmods = modificationcountdf.at[modtype,pos]
                    totalreads = modificationcountdf.at['Total',pos]
                    if allmods > totalreads: # fix CRISPResso errors where more mutations are counted than reads are available; set it to max. possible 100%
                        allmods = totalreads
                    if totalreads < 10:  # only analyze if there are at least 10 reads otherwise set to None
                        allmods = None
                    nicklist[index][posind] = allmods

                for posind, pos in enumerate(flapcolumns):
                    allmods = modificationcountdf.at[modtype,pos]
                    totalreads = modificationcountdf.at['Total',pos]
                    if allmods > totalreads: # fix CRISPResso errors where more mutations are counted than reads are available; set it to max. possible 100%
                        allmods = totalreads
                    if totalreads < 10:  # only analyze if there are at least 10 reads otherwise set to None
                        allmods = None
                    flaplist[index][posind] = allmods
                for posind, pos in enumerate(rttcolumns):
                    allmods = modificationcountdf.at[modtype,pos]
                    totalreads = modificationcountdf.at['Total',pos]
                    if allmods > totalreads: # fix CRISPResso errors where more mutations are counted than reads are available; set it to max. possible 100%
                        allmods = totalreads
                    modpercentage = allmods/totalreads*100   
                    if totalreads < 10:  # only analyze if there are at least 10 reads otherwise set to None
                        modpercentage = None
                    rttlist[index][posind] = modpercentage
                
                for posind, pos in enumerate(pbscolumns):
                    allmods = modificationcountdf.at[modtype,pos]
                    totalreads = modificationcountdf.at['Total',pos]
                    if allmods > totalreads: # fix CRISPResso errors where more mutations are counted than reads are available; set it to max. possible 100%
                        allmods = totalreads
                    modpercentage = allmods/totalreads*100
                    if totalreads < 10:  # only analyze if there are at least 10 reads otherwise set to None
                        modpercentage = None
                    pbslist[index][posind] = modpercentage

        
        except FileNotFoundError:  # add None values to list if variant doesn't contain HDR reads (and therefore no HDR file)
            nicklist = [[None,None,None,None],[None,None,None,None],[None,None,None,None],[None,None,None,None]]
            flaplist = [[None,None,None,None,None,None,None],[None,None,None,None,None,None,None],[None,None,None,None,None,None,None],[None,None,None,None,None,None,None]]
            rttlist = rttlist_empty.copy()
            pbslist = pbslist_empty.copy()
        
        
        # if len(nucleotidecolumns) < flapend+5:  # skip variants where reference sequence is too short to visualize 5 bases after flap end
        #         continue
        
        # remove Nones from rttlist
        for x in range(4):
            res = []
            for val in rttlist[x]:
                if val != None :
                    res.append(val)
            rttlist[x] = res      
        if amptype == 'HDR':
            hdrnickinsertionlist.append(nicklist[0])
            hdrnickdeletionlist.append(nicklist[1])
            hdrnicksubstitutionlist.append(nicklist[2])
            hdrnickmutationlist.append(nicklist[3])
            
            hdrflapinsertionlist.append(flaplist[0])
            hdrflapdeletionlist.append(flaplist[1])
            hdrflapsubstitutionlist.append(flaplist[2])
            hdrflapmutationlist.append(flaplist[3])
            
            hdrrttinsertionlist.append(rttlist[0])
            hdrrttdeletionlist.append(rttlist[1])
            hdrrttsubstitutionlist.append(rttlist[2])
            hdrrttmutationlist.append(rttlist[3])
            
            hdrpbsinsertionlist.append(pbslist[0])
            hdrpbsdeletionlist.append(pbslist[1])
            hdrpbssubstitutionlist.append(pbslist[2])
            hdrpbsmutationlist.append(pbslist[3])
            
        elif amptype == 'Reference':
            referencenickinsertionlist.append(nicklist[0])
            referencenickdeletionlist.append(nicklist[1])
            referencenicksubstitutionlist.append(nicklist[2])
            referencenickmutationlist.append(nicklist[3])
            
            referenceflapinsertionlist.append(flaplist[0])
            referenceflapdeletionlist.append(flaplist[1])
            referenceflapsubstitutionlist.append(flaplist[2])
            referenceflapmutationlist.append(flaplist[3])
            
            referencerttinsertionlist.append(rttlist[0])
            referencerttdeletionlist.append(rttlist[1])
            referencerttsubstitutionlist.append(rttlist[2])
            referencerttmutationlist.append(rttlist[3])
            
            referencepbsinsertionlist.append(pbslist[0])
            referencepbsdeletionlist.append(pbslist[1])
            referencepbssubstitutionlist.append(pbslist[2])
            referencepbsmutationlist.append(pbslist[3])
            
            nickcolumnlist.append(nickcolumns)
            flapcolumnlist.append(flapcolumns)
        

    finalvariantlist.append(variant)
    readsalignedtotal = editingquantfile.at[0,'Reads_in_input']  # use all files from input to keep it consistent with PRIDICT evaluation
    readsaligned_total_reference = editingquantfile.at[0,'Reads_aligned']
    readsaligned_total_hdr = editingquantfile.at[1,'Reads_aligned']
    readsaligned_Unmodified_reference = editingquantfile.at[0,'Unmodified']
    readsaligned_Unmodified_hdr = editingquantfile.at[1,'Unmodified']
    readsaligned_modified_reference = editingquantfile.at[0,'Modified']
    readsaligned_modified_hdr = editingquantfile.at[1,'Modified']
    readsaligned_insertions_reference = editingquantfile.at[0,'Only Insertions']
    readsaligned_insertions_hdr = editingquantfile.at[1,'Only Insertions']
    readsaligned_deletions_reference = editingquantfile.at[0,'Only Deletions']
    readsaligned_deletions_hdr = editingquantfile.at[1,'Only Deletions']
    readsaligned_substitutions_reference = editingquantfile.at[0,'Only Substitutions']
    readsaligned_substitutions_hdr = editingquantfile.at[1,'Only Substitutions']
    
    readsalignedtotallist.append(readsalignedtotal)
    readsaligned_total_referencelist.append(readsaligned_total_reference)
    readsaligned_total_hdrlist.append(readsaligned_total_hdr)
    readsaligned_Unmodified_referencelist.append(readsaligned_Unmodified_reference)
    readsaligned_Unmodified_hdrlist.append(readsaligned_Unmodified_hdr)
    readsaligned_modified_referencelist.append(readsaligned_modified_reference)
    readsaligned_modified_hdrlist.append(readsaligned_modified_hdr)
    readsaligned_insertions_referencelist.append(readsaligned_insertions_reference)
    readsaligned_insertions_hdrlist.append(readsaligned_insertions_hdr)
    readsaligned_deletions_referencelist.append(readsaligned_deletions_reference)
    readsaligned_deletions_hdrlist.append(readsaligned_deletions_hdr)
    readsaligned_substitutions_referencelist.append(readsaligned_substitutions_reference)
    readsaligned_substitutions_hdrlist.append(readsaligned_substitutions_hdr)
    
    
    ind+=1
    indtemp+=1
    
    if indtemp == 1000:
        print(ind)
        indtemp = 0

summarydf['variantnr'] = finalvariantlist
summarydf['readsalignedtotal'] = readsalignedtotallist
summarydf['readsaligned_total_reference'] = readsaligned_total_referencelist
summarydf['readsaligned_total_hdr'] = readsaligned_total_hdrlist
summarydf['readsaligned_Unmodified_reference'] = readsaligned_Unmodified_referencelist
summarydf['readsaligned_Unmodified_hdr'] = readsaligned_Unmodified_hdrlist
summarydf['readsaligned_modified_reference'] = readsaligned_modified_referencelist
summarydf['readsaligned_modified_hdr'] = readsaligned_modified_hdrlist
summarydf['readsaligned_insertions_reference'] = readsaligned_insertions_referencelist
summarydf['readsaligned_modified_hdr'] = readsaligned_modified_hdrlist
summarydf['readsaligned_insertions_reference'] = readsaligned_insertions_referencelist
summarydf['readsaligned_insertions_hdr'] = readsaligned_insertions_hdrlist
summarydf['readsaligned_deletions_reference'] = readsaligned_deletions_referencelist
summarydf['readsaligned_deletions_hdr'] = readsaligned_deletions_hdrlist
summarydf['readsaligned_substitutions_reference'] = readsaligned_substitutions_referencelist
summarydf['readsaligned_substitutions_hdr'] = readsaligned_substitutions_hdrlist


summarydf['hdrnickinsertion'] = hdrnickinsertionlist
summarydf['hdrnickdeletion'] = hdrnickdeletionlist
summarydf['hdrnicksubstitution'] = hdrnicksubstitutionlist
summarydf['hdrnickmutation'] = hdrnickmutationlist
summarydf['hdrflapinsertion'] = hdrflapinsertionlist
summarydf['hdrflapdeletion'] = hdrflapdeletionlist
summarydf['hdrflapsubstitution'] = hdrflapsubstitutionlist
summarydf['hdrflapmutation'] = hdrflapmutationlist
summarydf['referencenickinsertion'] = referencenickinsertionlist
summarydf['referencenickdeletion'] = referencenickdeletionlist
summarydf['referencenicksubstitution'] = referencenicksubstitutionlist
summarydf['referencenickmutation'] = referencenickmutationlist
summarydf['referenceflapinsertion'] = referenceflapinsertionlist
summarydf['referenceflapdeletion'] = referenceflapdeletionlist
summarydf['referenceflapsubstitution'] = referenceflapsubstitutionlist
summarydf['referenceflapmutation'] = referenceflapmutationlist
summarydf['nickbases'] = nickcolumnlist
summarydf['flapbases'] = flapcolumnlist


# PBS averages:
hdrpbsdeletionaveragelist = [np.nanmedian(np.array(hdrpbsdeletion,dtype=float)) for hdrpbsdeletion in hdrpbsdeletionlist]
referencepbsdeletionaveragelist = [np.nanmedian(np.array(referencepbsdeletion,dtype=float)) for referencepbsdeletion in referencepbsdeletionlist]

hdrpbsinsertionaveragelist = [np.nanmedian(np.array(hdrpbsinsertion,dtype=float)) for hdrpbsinsertion in hdrpbsinsertionlist]
referencepbsinsertionaveragelist = [np.nanmedian(np.array(referencepbsinsertion,dtype=float)) for referencepbsinsertion in referencepbsinsertionlist]

hdrpbssubstitutionaveragelist = [np.nanmedian(np.array(hdrpbssubstitution,dtype=float)) for hdrpbssubstitution in hdrpbssubstitutionlist]
referencepbssubstitutionaveragelist = [np.nanmedian(np.array(referencepbssubstitution,dtype=float)) for referencepbssubstitution in referencepbssubstitutionlist]

hdrpbsmutationaveragelist = [np.nanmedian(np.array(hdrpbsmutation,dtype=float)) for hdrpbsmutation in hdrpbsmutationlist]
referencepbsmutationaveragelist = [np.nanmedian(np.array(referencepbsmutation,dtype=float)) for referencepbsmutation in referencepbsmutationlist]

summarydf['referencepbsinsertionaverage'] = referencepbsinsertionaveragelist
summarydf['referencepbsdeletionaverage'] = referencepbsdeletionaveragelist
summarydf['referencepbssubstitutionaverage'] = referencepbssubstitutionaveragelist
summarydf['referencepbsmutationaverage'] = referencepbsmutationaveragelist

summarydf['hdrpbsinsertionaverage'] = hdrpbsinsertionaveragelist
summarydf['hdrpbsdeletionaverage'] = hdrpbsdeletionaveragelist
summarydf['hdrpbssubstitutionaverage'] = hdrpbssubstitutionaveragelist
summarydf['hdrpbsmutationaverage'] = hdrpbsmutationaveragelist


# RTT averages:
hdrrttdeletionaveragelist = [np.nanmedian(np.array(hdrrttdeletion,dtype=float)) for hdrrttdeletion in hdrrttdeletionlist]
referencerttdeletionaveragelist = [np.nanmedian(np.array(referencerttdeletion,dtype=float)) for referencerttdeletion in referencerttdeletionlist]

hdrrttinsertionaveragelist = [np.nanmedian(np.array(hdrrttinsertion,dtype=float)) for hdrrttinsertion in hdrrttinsertionlist]
referencerttinsertionaveragelist = [np.nanmedian(np.array(referencerttinsertion,dtype=float)) for referencerttinsertion in referencerttinsertionlist]

hdrrttsubstitutionaveragelist = [np.nanmedian(np.array(hdrrttsubstitution,dtype=float)) for hdrrttsubstitution in hdrrttsubstitutionlist]
referencerttsubstitutionaveragelist = [np.nanmedian(np.array(referencerttsubstitution,dtype=float)) for referencerttsubstitution in referencerttsubstitutionlist]

hdrrttmutationaveragelist = [np.nanmedian(np.array(hdrrttmutation,dtype=float)) for hdrrttmutation in hdrrttmutationlist]
referencerttmutationaveragelist = [np.nanmedian(np.array(referencerttmutation,dtype=float)) for referencerttmutation in referencerttmutationlist]

summarydf['referencerttinsertionaverage'] = referencerttinsertionaveragelist
summarydf['referencerttdeletionaverage'] = referencerttdeletionaveragelist
summarydf['referencerttsubstitutionaverage'] = referencerttsubstitutionaveragelist
summarydf['referencerttmutationaverage'] = referencerttmutationaveragelist

summarydf['hdrrttinsertionaverage'] = hdrrttinsertionaveragelist
summarydf['hdrrttdeletionaverage'] = hdrrttdeletionaveragelist
summarydf['hdrrttsubstitutionaverage'] = hdrrttsubstitutionaveragelist
summarydf['hdrrttmutationaverage'] = hdrrttmutationaveragelist


# Split columns with lists of nickbases and flapbases into individual columns
splitcollist = ['hdrnickinsertion',
       'hdrnickdeletion', 'hdrnicksubstitution', 'hdrnickmutation',
       'hdrflapinsertion', 'hdrflapdeletion', 'hdrflapsubstitution',
       'hdrflapmutation', 'referencenickinsertion', 'referencenickdeletion',
       'referencenicksubstitution', 'referencenickmutation',
       'referenceflapinsertion', 'referenceflapdeletion',
       'referenceflapsubstitution', 'referenceflapmutation']



splitcolsdf = summarydf[splitcollist]
summarydf=summarydf.drop(splitcollist, axis=1)


for col in splitcollist:
    if 'flap' in col:
        splitdf = pd.DataFrame(splitcolsdf[col].to_list(), columns=[col+'_1',col+'_2',col+'_3',col+'_4',col+'_5',col+'_6',col+'_7'])
    else:
        splitdf = pd.DataFrame(splitcolsdf[col].to_list(), columns=[col+'_1',col+'_2',col+'_3',col+'_4'])
    # print(splitdf)
    tempcols = list(splitdf.columns)
    for tempcol in tempcols:
        summarydf[tempcol] = splitdf[tempcol]


summarydf = summarydf[summarydf['readsalignedtotal'] >= 100]  #filter out variants which have less than 100 reads
summarydf = summarydf.fillna(0)  # fill NaN cells with 0, since we combine reference with hdr and can't sum up int with nan

amptype_1 = [ 'readsaligned_total',
 'readsaligned_Unmodified',
 'readsaligned_modified',
 'readsaligned_insertions',
 'readsaligned_deletions',
 'readsaligned_substitutions']

for col in amptype_1:
    sum_column = summarydf[col+"_hdr"] + summarydf[col+"_reference"]
    summarydf[col] = sum_column


# combine hdr/reference reads to one value:
amptype_2 = [
 'nickinsertion_1',
 'nickinsertion_2',
 'nickinsertion_3',
 'nickinsertion_4',
 'nickdeletion_1',
 'nickdeletion_2',
 'nickdeletion_3',
 'nickdeletion_4',
 'nicksubstitution_1',
 'nicksubstitution_2',
 'nicksubstitution_3',
 'nicksubstitution_4',
 'nickmutation_1',
 'nickmutation_2',
 'nickmutation_3',
 'nickmutation_4',
 'flapinsertion_1',
 'flapinsertion_2',
 'flapinsertion_3',
 'flapinsertion_4',
 'flapinsertion_5',
 'flapinsertion_6',
 'flapinsertion_7',
 'flapdeletion_1',
 'flapdeletion_2',
 'flapdeletion_3',
 'flapdeletion_4',
 'flapdeletion_5',
 'flapdeletion_6',
 'flapdeletion_7',
 'flapsubstitution_1',
 'flapsubstitution_2',
 'flapsubstitution_3',
 'flapsubstitution_4',
 'flapsubstitution_5',
 'flapsubstitution_6',
 'flapsubstitution_7',
 'flapmutation_1',
 'flapmutation_2',
 'flapmutation_3',
 'flapmutation_4',
 'flapmutation_5',
 'flapmutation_6',
 'flapmutation_7']

for col in amptype_2:
    sum_column = summarydf["hdr"+col] + summarydf["reference"+col]
    summarydf[col] = sum_column
    summarydf[col] = summarydf.apply(lambda x: (x[col]/x.readsalignedtotal)*100 ,axis=1)

amptype_3 = ['pbsinsertionaverage',
 'pbsdeletionaverage',
 'pbssubstitutionaverage',
 'pbsmutationaverage',
 'rttinsertionaverage',
 'rttdeletionaverage',
 'rttsubstitutionaverage',
 'rttmutationaverage']

# calculate weighted average of pbs/rtt mutations of hdr/reference aligned amplicons
for col in amptype_3:
    sum_column = summarydf["hdr"+col] + summarydf["reference"+col]
    summarydf[col] = summarydf.apply(lambda x: x['hdr'+col]*(x['readsaligned_total_hdr']/x.readsalignedtotal) + x['reference'+col]*(x['readsaligned_total_reference']/x.readsalignedtotal) ,axis=1)



meandf = summarydf.mean(axis=0)
summarydf.loc['Mean',:] = meandf  # add mean row at bottom of dataframe


#create list for visualization of unintended mutations:
meanlist = ['pbsmutationaverage','nickmutation_1',
  'nickmutation_2',
  'nickmutation_3',
  'nickmutation_4','rttmutationaverage', 'flapmutation_1',
  'flapmutation_2',
  'flapmutation_3',
  'flapmutation_4','flapmutation_5',
  'flapmutation_6',
  'flapmutation_7']

for amptype in ['','']:
    if amptype == '':
        meanvalues = meandf[meanlist]
    if amptype == '':
        meanvalues = meandf[meanlist]
print(meanlist, meanvalues)
print(meanlist, meanvalues)

summarydf.to_csv('20220818_CRISPResso_long_summary_'+cwd+'.csv')
