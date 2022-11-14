# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:16:49 2020

Python script for the analysis of NGS sequencing reads in the Prime editing screen project.
Each read is compared to the templatedf which contains original sequences.
After matching the reads to the templatedf based on protospacer, RTtemplate and endoftarget sequence, 
the reads are stored in individual .fastq files for downstream editing analysis.

@author: nimath
"""
import pandas as pd
from Bio import SeqIO
import gzip
from os import listdir
import numpy as np
import os
import time

# Get the current working directory
cwd = os.path.join(os.getcwd(), '')


def lookup(prototemplate, rttemplate, endoftargettemplate, targettemplate):  # generate lookup dictionaries for all three relevant regions (protospacer, RTstart, endoftarget)
    protolookup = {}
    for i, z in enumerate(prototemplate):
            protolookup.setdefault(z, []).append(i)
    
    rtlookup = {}
    for i, z in enumerate(rttemplate):
            rtlookup.setdefault(z[:15], []).append(i)  # only consider first 15 bases of RT
    
    endoftargetlookup = {}
    for i, z in enumerate(endoftargettemplate):
            endoftargetlookup.setdefault(z, []).append(i)
            
    targetlookup = {}
    for i, z in enumerate(targettemplate):
            targetlookup.setdefault(z, []).append(i)
    
    return protolookup, rtlookup, endoftargetlookup, targetlookup


def importfiles(filename, protolookup, rtlookup, endoftargetlookup,targetlookup, barcodelookup):
    '''

    Parameters
    ----------
    filename : string
        Filename of the fasta file, containing the NGS reads to analyze.

    Returns
    -------
    diseasedict : dictionary
        Dictionary containing read identifier as key and protospacer/barcode/targetend/fwread as sub-keys (sequence as value of those) .
    diseasedf : pandas dataframe
        pandas dataframe made from diseasedict.
    diseasedf_filtered : pandas dataframe
        filtered diseasedf without whole sequence, but only sequence parts needed to identify index.
    full_adapter_read_percent : float
        Float which is the amount of sequences (from 0 to 1) which contain all subsequences without NaN.

    '''
    diseasedict = {}
    filename = shortname+'_Protospacer'+filtered+'.fastq.gz'
    with gzip.open(path+filename, "rt") as fasta_file:
        count = 0
        counttemp = 0
        print('Start Protospacer lookup loop...')
        for seq_record in SeqIO.parse(fasta_file, 'fastq'):
            protospacer = seq_record.seq
            protomatch = protolookup.get("G"+str(protospacer))
            identifier = seq_record.id
            if identifier in diseasedict:
                diseasedict[identifier]["protomatch"] = protomatch
            else:
                diseasedict[identifier] = {'protomatch':protomatch}
            count+=1
            counttemp+=1
            if counttemp == 1000000:
                print(count)
                counttemp = 0
    print('Protospacer done')
    
   ### Not needed for final lookup if barcodes have been analyzed in a previous run! ###
    # filename = shortname+'_barcode'+filtered+'.fastq.gz'
    # with gzip.open(path+filename, "rt") as fasta_file:
    #     count = 0
    #     counttemp = 0
    #     print('Start barcode lookup loop...')
    #     for seq_record in SeqIO.parse(fasta_file, 'fastq'):
    #         barcode = seq_record.seq
    #         identifier = seq_record.id
    #         if identifier in diseasedict:
    #             diseasedict[identifier]["barcode"] = str(barcode)
    #         # no else since we are only interested in reads which have protospacer match
    #         # else:
    #         #     diseasedict[identifier] = {'barcode':str(barcode)}
    #         count+=1
    #         counttemp+=1
    #         if counttemp == 1000000:
    #             print(count)
    #             counttemp = 0
    # print('Barcode done')
    
    filename = shortname+'_barcode'+filtered+'.fastq.gz'
    with gzip.open(path+filename, "rt") as fasta_file:
        count = 0
        counttemp = 0
        print('Start barcode lookup loop...')
        for seq_record in SeqIO.parse(fasta_file, 'fastq'):
            barcode = seq_record.seq
            barcodematch = barcodelookup.get(barcode)
            identifier = seq_record.id
            if identifier in diseasedict:
                diseasedict[identifier]["barcodematch"] = barcodematch
            # no else since we are only interested in reads which have protospacer match
            # else:
            #     diseasedict[identifier] = {'barcode':str(barcode)}
            count+=1
            counttemp+=1
            if counttemp == 1000000:
                print(count)
                counttemp = 0
    print('Barcode done')
      
    filename = shortname+'_targetend'+filtered+'.fastq.gz'
    with gzip.open(path+filename, "rt") as fasta_file:
        count = 0
        counttemp = 0
        print('Start targetend lookup loop...')
        for seq_record in SeqIO.parse(fasta_file, 'fastq'):
            targetend = seq_record.seq
            identifier = seq_record.id
            endoftarget = str(targetend[-10:])
            endoftargetmatch = endoftargetlookup.get(endoftarget)
            targetseq = str(seq_record.seq)[-23-20:-23]
            targetmatch = targetlookup.get(targetseq)
            if identifier in diseasedict:
                diseasedict[identifier]["endoftargetmatch"] = endoftargetmatch
                diseasedict[identifier]["targetmatch"] = targetmatch
            # no else since we are only interested in reads which have protospacer match
            # else:
            #     diseasedict[identifier] = {'endoftargetmatch':endoftargetmatch}
            #     diseasedict[identifier] = {'targetmatch':targetmatch}
            count+=1
            counttemp+=1
            if counttemp == 1000000:
                print(count)
                counttemp = 0

    print('Targetend done')
    
    filename = shortname+'_RTshort'+filtered+'.fastq.gz'
    with gzip.open(path+filename, "rt") as fasta_file:
        count = 0
        counttemp = 0
        print('Start RT lookup loop...')
        for seq_record in SeqIO.parse(fasta_file, 'fastq'):
            rt = seq_record.seq
            rttemplate = str(rt[:15])
            rttemplatematch = rtlookup.get(rttemplate)
            identifier = seq_record.id
            if identifier in diseasedict:
                diseasedict[identifier]["rttemplatematch"] = rttemplatematch
            # no else since we are only interested in reads which have protospacer match
            # else:
            #     diseasedict[identifier] = {'rttemplatematch':rttemplatematch}
            count+=1
            counttemp+=1
            if counttemp == 1000000:
                print(count)
                counttemp = 0

    print('RT done')
    
    return diseasedict


def mergeDict(dict1, dict2):
        ''' Merge dictionaries and keep values of common keys in list'''
        dict3 = {**dict1, **dict2}
        for key, value in dict3.items():
            if key in dict1 and key in dict2:
                dict3[key] = [value , dict1[key]]
        return dict3


def barcodelookupfunc(uniquebarcodetemplate):
    ''' Convert series of barcode matches to lookup dictionary'''
    barcodelookup = {}
    for index, valuelist in uniquebarcodetemplate.items():
        tempdict = {}
        for value in valuelist:
            tempdict[value]=index
        barcodelookup = mergeDict(barcodelookup,tempdict)
        
    for key in barcodelookup:
        if type(barcodelookup[key]) == list:
            templist = list(map(int, str(barcodelookup[key]).replace("[","").strip("]").split(', ')))
            barcodelookup[key] = templist
        if type(barcodelookup[key]) == int:
            barcodelookup[key] = [barcodelookup[key]]
    return barcodelookup


templatedf = pd.read_csv('20210527_templatefiltered.csv')
# templatedf['amplicon_short'] = templatedf.apply(lambda row: row.amplicon[-111+row.PBSlength+int(row.RTlength):-23], axis=1)
# templatedf['mutamplicon_short'] = templatedf.apply(lambda row: row.amplicon[-111+row.PBSlength+int(row.RTlength):-23], axis=1)
templatedf['mutated_amplicon_real'] = templatedf.apply(lambda row: row.mutated_amplicon[-len(row.amplicon)-int(row.Correction_Length):] if row.Correction_Type == 'Insertion' else (row.mutated_amplicon[-len(row.amplicon):] if row.Correction_Type == 'Replacement' else row.mutated_amplicon[-len(row.amplicon)+int(row.Correction_Length):]), axis=1)
scaffold = 'GTTTCAGAGCTATGCTGGAAACAGCATAGCAAGTTGAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC'
templatedf['Disease_Block_final'] = templatedf.apply(lambda row: row.Disease_Block[:39]+scaffold+row.Disease_Block[-131:], axis=1)
templatedf['Disease_Block_final_mutated'] = templatedf.apply(lambda row: row.Disease_Block[:39]+scaffold+row.Disease_Block[-131:row.PBSlength+int(row.RTlength)+7]+row.mutated_amplicon_real+row.Disease_Block[-20:], axis=1)
templatedf['WT_Target_Correct'] = templatedf.apply(lambda row: row.Disease_Block_final[-43-40:-43], axis=1)
templatedf['mutated_Target_Correct'] = templatedf.apply(lambda row: row.Disease_Block_final_mutated[-43-40:-43], axis=1)
templatedf['amplicon_for_match'] = templatedf.apply(lambda row: row.amplicon[-23-40:-23], axis=1)

def list_files1(directory):
    return [f for f in listdir(directory) if '_Protospacer' in f]

# adapt path to the location of the fastq files (change to "\\Fastq-Files\\"" when downloading this repo):    
# path = 'C:\\Users\\nimath\\switchdrive\\Institution\\PhD\\01_Experiments\\05_Prime_Editing\\03_DiseaseScreen\\04_Analysis\\NovaSeq\\Testreads\\'
path = cwd
# filelist = list_files1(path)
filelist = ['03-NM-PE3_Protospacer.fastq.gz']
filtered = ''

for filename in filelist[0:1]:  # loop through all SAMPLES in a directory with "R1" in the name (listfiles1 function)
    print(filename)
    shortname = filename[0:-21]
    prototemplate = templatedf['protospacer'].values.flatten()
    endoftargettemplate =templatedf['endoftarget'].values.flatten()
    rttemplate = templatedf['RT_template'].values.flatten()
    targettemplate = templatedf['amplicon_for_match'].values.flatten()
    uniquebarcodetemplate = templatedf['unique_barcodes'][templatedf['unique_barcodes'] != "[]"].apply(lambda x: x.strip("[]").replace("'","").split(', '))
    # uniquebarcodedf = templatedf.loc[:,['unique_barcodes']][templatedf['unique_barcodes'] != "[]"]
    # uniquebarcodedf['unique_barcodes'] = uniquebarcodedf['unique_barcodes'].apply(lambda row: row.strip("[]").replace("'","").split(', '))
    
    barcodelookup = barcodelookupfunc(uniquebarcodetemplate)

    protolookup, rtlookup, endoftargetlookup, targetlookup = lookup(prototemplate,rttemplate,endoftargettemplate, targettemplate)
    del prototemplate, endoftargettemplate, rttemplate, targettemplate, uniquebarcodetemplate
    diseasedict = importfiles(filename,protolookup, rtlookup, endoftargetlookup, targetlookup, barcodelookup)
    diseasedf = pd.DataFrame.from_dict(diseasedict,orient='index')  # make dataframe from dict
    del diseasedict

    ### only need during first round of acquiring barcode information of all reads, comment out afterwards
    # multipledf = diseasedf.dropna()
    # multipledf = multipledf[multipledf['targetmatch'].apply(lambda x: len(x) > 0)]
    # multipledf['initial_match'] = [list(set(a).intersection(set(b), set(c))) for a, b, c in zip(multipledf.protomatch, multipledf.rttemplatematch, multipledf.endoftargetmatch)]
    # multipledf['match'] = [list(set(a).intersection(set(b), set(c), set(d))) for a, b, c, d in zip(multipledf.protomatch, multipledf.rttemplatematch, multipledf.endoftargetmatch, multipledf.targetmatch)]
    # multipledf = multipledf[multipledf['match'].apply(lambda x: len(x) == 1)]
    ###
    
    # diseasedffilter = diseasedf[['protomatch', 'barcodematch', 'endoftargetmatch', 'rttemplatematch']]
    
    
    replace_NaN = pd.isnull(diseasedf['barcodematch']) # make Series with True/False if barcodematch is NaN or not
    diseasedf.loc[replace_NaN,'barcodematch'] = -1 # replace all "NaN" with "-1" which will not match with others
    diseasedf['barcodematch'] = diseasedf['barcodematch'].apply(lambda x: [x] if type(x) == int else x) # put -1 into list for performing matching algorithm below
    print('diseasedf with and without proto/endoftarget/rttemplatematch:',len(diseasedf))
    diseasedf.dropna(subset = ['protomatch', 'endoftargetmatch', 'rttemplatematch'], inplace=True) # do not drop elements without barcode match, since barcode match only includes barcodes from ambiguous sequences
    print('diseasedf after filtering for proto/endoftarget/rttemplatematch:',len(diseasedf))
    diseasedf['match'] = [list(set(a).intersection(set(b), set(c))) for a, b, c in zip(diseasedf.protomatch, diseasedf.rttemplatematch, diseasedf.endoftargetmatch)]
    diseasedf['matchwithbarcode'] = [list(set(a).intersection(set(b), set(c), set(d))) for a, b, c, d in zip(diseasedf.protomatch, diseasedf.rttemplatematch, diseasedf.endoftargetmatch, diseasedf.barcodematch)]
    final_diseasedf = diseasedf[diseasedf['match'].map(lambda d: len(d)) > 0][['match','matchwithbarcode']]  # all reads which have at least one match; those with multiple matches will be assigned by barcodematch
    final_diseasedf['match'] = final_diseasedf.apply(lambda row: row.matchwithbarcode if len(row.match) > 1 else row.match, axis=1)
    print('final_diseasedf after filtering for pegRNAmatch:',len(final_diseasedf))    
    final_diseasedf = final_diseasedf[final_diseasedf['match'].map(lambda d: len(d)) == 1]['match']
    
    # if match > 1 make match with targetmatch
    # multiplematch_diseasedf = diseasedf[diseasedf['match'].map(lambda d: len(d)) > 1]  # all reads which have multiple variants associated (need to be separated by barcode)

print("calculating...")
templatedf["editedcount"] = 0
templatedf["uneditedcount"] = 0
templatedf["indelcount"] = 0
templatedf["nickindelcount"] = 0
templatedf["beforeflapindelcount"] = 0
templatedf["totalreads"] = 0
templatedf['barcodes'] = np.empty((len(templatedf), 0)).tolist()
filename = shortname+'_targetend'+filtered+'.fastq.gz'
templatenumpy = templatedf.to_numpy()

# c = 0
readcounter = 0
readcounttemp = 0
start = time.time()

# define column position for used column in numpy array:
uneditedcountnr = templatedf.columns.get_loc("uneditedcount")
editedcountnr = templatedf.columns.get_loc("editedcount")
indelcountnr = templatedf.columns.get_loc("indelcount")
nickindelcountnr = templatedf.columns.get_loc("nickindelcount")
beforeflapindelcountnr = templatedf.columns.get_loc("beforeflapindelcount")
ampliconnr = templatedf.columns.get_loc("amplicon")
mutated_ampliconnr = templatedf.columns.get_loc("mutated_amplicon")
editingpositionnr = templatedf.columns.get_loc("Editing_Position")
RTlengthpositionnr = templatedf.columns.get_loc("RTlength")
correction_typepositionnr = templatedf.columns.get_loc("Correction_Type")
correction_lengthpositionnr = templatedf.columns.get_loc("Correction_Length")
namenr = templatedf.columns.get_loc("Name")
reporter1fwindex = templatedf.index[templatedf['Name'] == '186PN_short_guide_TGG_FW'][0]
reporter2fwindex = templatedf.index[templatedf['Name'] == '188PN_short_guide_TAC_FW'][0]


with gzip.open(path+filename, "rt") as fasta_file:
    for seq_record in SeqIO.parse(fasta_file, 'fastq'):
        targetend = str(seq_record.seq)
        
        identifier = seq_record.id
        readcounter+=1
        readcounttemp+=1
        
        if readcounttemp == 100000:
            print(readcounter)
            readcounttemp = 0
        if not identifier in final_diseasedf.index:
            # if identifier in multiple_diseasedf.index:
                
            continue
        variantindex = diseasedf.at[identifier,'match'][0]
        if variantindex <119701:  # only analyze endogenous edits in library
            continue
        
        RTlength = templatenumpy[variantindex,RTlengthpositionnr]
        correction_type = templatenumpy[variantindex,correction_typepositionnr]
        correction_length = templatenumpy[variantindex,correction_lengthpositionnr]
        
        if correction_type == 'Replacement':
            sequenceWT = targetend[-2-RTlength-5-25:-25]  # in contrast to other library 1 members, analysis window starts at -25 instead of -24 (there is one more base after protospacer in sequencing read)
            sequenceWT_beforeflap = targetend[+2-RTlength-2-25:-25]  # full sequence until 5bp before flap
            sequenceWT_nick = targetend[-4-25:-25]
            
            sequenceMUT = targetend[-2-RTlength-5-25:-25]
            sequenceMUT_beforeflap = targetend[+2-RTlength-2-25:-25]
            sequenceMUT_nick = targetend[-4-25:-25]
            
            controlWT = templatenumpy[variantindex,ampliconnr][-2-RTlength-5-25:-25]
            controlWT_beforeflap = templatenumpy[variantindex,ampliconnr][+2-RTlength-2-25:-25]
            controlWT_nick = templatenumpy[variantindex,ampliconnr][-4-25:-25]
            
            controlMUT = templatenumpy[variantindex,mutated_ampliconnr][-2-RTlength-5-25:-25]
            controlMUT_beforeflap = templatenumpy[variantindex,mutated_ampliconnr][+2-RTlength-2-25:-25]
            controlMUT_nick = templatenumpy[variantindex,mutated_ampliconnr][-4-25:-25]
           
            if sequenceWT == controlWT:
                templatenumpy[variantindex,uneditedcountnr] += 1 # uneditedcount is column 42
            elif sequenceMUT == controlMUT:
                templatenumpy[variantindex,editedcountnr] += 1
            else:
                templatenumpy[variantindex,indelcountnr] += 1
                if (sequenceWT_nick != controlWT_nick) and (sequenceMUT_nick != controlMUT_nick):  # check if 4bp window around nick has unintended edits
                    templatenumpy[variantindex,nickindelcountnr] += 1
                
                if (sequenceWT_beforeflap != controlWT_beforeflap) and (sequenceMUT_beforeflap != controlMUT_beforeflap):  # check if 4bp window around nick has unintended edits
                    templatenumpy[variantindex,beforeflapindelcountnr] += 1
                    
        elif correction_type == 'Deletion':
            sequenceWT = targetend[-2-RTlength-correction_length-5-25:-25]
            sequenceWT_beforeflap = targetend[+2-RTlength-correction_length-2-25:-25]  # full sequence until 5bp before flap
            sequenceWT_nick = targetend[-4-25:-25]
            
            sequenceMUT = targetend[-2-RTlength-5-25:-25]
            sequenceMUT_beforeflap = targetend[+2-RTlength-2-25:-25]
            sequenceMUT_nick = targetend[-4-25:-25]
            
            controlWT = templatenumpy[variantindex,ampliconnr][-2-RTlength-correction_length-5-25:-25]
            controlWT_beforeflap = templatenumpy[variantindex,ampliconnr][+2-RTlength-correction_length-2-25:-25]
            controlWT_nick = templatenumpy[variantindex,ampliconnr][-4-25:-25]
            
            controlMUT = templatenumpy[variantindex,mutated_ampliconnr][-2-RTlength-5-25:-25]
            controlMUT_beforeflap = templatenumpy[variantindex,mutated_ampliconnr][+2-RTlength-2-25:-25]
            controlMUT_nick = templatenumpy[variantindex,mutated_ampliconnr][-4-25:-25]
           
            if sequenceWT == controlWT:
                templatenumpy[variantindex,uneditedcountnr] += 1 # uneditedcount is column 42
            elif sequenceMUT == controlMUT:
                templatenumpy[variantindex,editedcountnr] += 1
            else:
                templatenumpy[variantindex,indelcountnr] += 1
                if (sequenceWT_nick != controlWT_nick) and (sequenceMUT_nick != controlMUT_nick):  # check if 4bp window around nick has unintended edits
                    templatenumpy[variantindex,nickindelcountnr] += 1
                
                if (sequenceWT_beforeflap != controlWT_beforeflap) and (sequenceMUT_beforeflap != controlMUT_beforeflap):  # check if 4bp window around nick has unintended edits
                    templatenumpy[variantindex,beforeflapindelcountnr] += 1
                    
        elif correction_type == 'Insertion':
            sequenceWT = targetend[-2-RTlength+correction_length-5-25:-25]
            sequenceWT_beforeflap = targetend[+2-RTlength+correction_length-2-25:-25]  # full sequence until 5bp before flap
            sequenceWT_nick = targetend[-4-25:-25]
            
            sequenceMUT = targetend[-2-RTlength-5-25:-25]
            sequenceMUT_beforeflap = targetend[+2-RTlength-2-25:-25]
            sequenceMUT_nick = targetend[-4-25:-25]
            
            controlWT = templatenumpy[variantindex,ampliconnr][-2-RTlength+correction_length-5-25:-25]
            controlWT_beforeflap = templatenumpy[variantindex,ampliconnr][+2-RTlength+correction_length-2-25:-25]
            controlWT_nick = templatenumpy[variantindex,ampliconnr][-4-25:-25]
            
            controlMUT = templatenumpy[variantindex,mutated_ampliconnr][-2-RTlength-5-25:-25]
            controlMUT_beforeflap = templatenumpy[variantindex,mutated_ampliconnr][+2-RTlength-2-25:-25]
            controlMUT_nick = templatenumpy[variantindex,mutated_ampliconnr][-4-25:-25]
           
            if sequenceWT == controlWT:
                templatenumpy[variantindex,uneditedcountnr] += 1 # uneditedcount is column 42
            elif sequenceMUT == controlMUT:
                templatenumpy[variantindex,editedcountnr] += 1
            else:
                templatenumpy[variantindex,indelcountnr] += 1
                if (sequenceWT_nick != controlWT_nick) and (sequenceMUT_nick != controlMUT_nick):  # check if 4bp window around nick has unintended edits
                    templatenumpy[variantindex,nickindelcountnr] += 1
                
                if (sequenceWT_beforeflap != controlWT_beforeflap) and (sequenceMUT_beforeflap != controlMUT_beforeflap):  # check if 4bp window around nick has unintended edits
                    templatenumpy[variantindex,beforeflapindelcountnr] += 1
            
end = time.time()
print('Time for loop:',end-start)

#make again dataframe out of numpy array for easier saving as csv:
templatedf = pd.DataFrame(data = templatenumpy, 
                  index = templatedf.index.tolist(), 
                  columns = templatedf.columns.tolist())
  

# del final_diseasedf # remove final_diseasedf from memory
totalreads = templatedf["uneditedcount"] + templatedf["editedcount"] + templatedf["indelcount"]
templatedf['totalreads'] = totalreads
templatedf['totalreads'].replace(0, np.nan, inplace=True)
percentageedited = (templatedf["editedcount"]/templatedf['totalreads'])*100
templatedf['percentageedited'] = percentageedited
percentageunedited = (templatedf["uneditedcount"]/templatedf['totalreads'])*100
templatedf['percentageunedited'] = percentageunedited
percentageindels = (templatedf["indelcount"]/templatedf['totalreads'])*100
templatedf['percentageindel'] = percentageindels
# templatedf['barcodenr'] = templatedf.apply(lambda row: len(set(row.barcodes)), axis=1)

cols = ['Name', 'Gene', 'Phenotype',
       'Disease_Block', 'PBSlength', 'RToverhanglength', 'RTlength',
       'First_RT_nuc', 'Poly_T', 'ReferenceAllele', 'AlternateAllele',
       'Correction_Type', 'Correction_Length', 'DesignNr_per_variant',
       'Editing_Position', 'Target_Strand', 'Duplicate',
       'WT_Target_Correct', 'mutated_Target_Correct', 'editedcount',
       'uneditedcount', 'indelcount','nickindelcount','beforeflapindelcount', 'totalreads', 'barcodes',
       'percentageedited', 'percentageunedited', 'percentageindel',
       'unique_barcodes']
templatedf = templatedf[cols]

# add unique barcodes of unedited sequences to templatedf which can later be used to distinguish ambiguous sequences
# print('Start multipledf calculation')
# templatedf['multi_barcodes'] = np.empty((len(templatedf), 0)).tolist()
# templatedf['multi_barcodes'] = templatedf['multi_barcodes'].apply(lambda x: set(x))
# for index, row in multipledf.iterrows():
#     variantindex = row.match[0]
#     templatedf.at[variantindex,"multi_barcodes"].add(row.barcode)
    
templatedf.to_csv('20220730_'+shortname+'_analysisdf_endo.csv')