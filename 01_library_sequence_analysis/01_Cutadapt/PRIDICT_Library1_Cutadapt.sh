#!/bin/bash
datum=$(date +"%Y%m%d")
touch $datum"_cutadaptlog.txt"
for  filename in *R1.fastq.gz; do
	shortname="${filename:0:-19}"
	cutadapt -j 0 -g aaggacgaaacaccG -o "Output/"$shortname"_5trim.fastq.gz" $filename >> $datum"_cutadaptlog.txt"
	cutadapt -j 0 -a GTTTCAGAGCTATG -m 19 -M 19 --discard-untrimmed -o "Output/"$shortname"_Protospacer.fastq.gz" "Output/"$shortname"_5trim.fastq.gz" >> $datum"_cutadaptlog.txt"
	cutadapt -j 0 -g CACCGAGTCGGTGC --discard-untrimmed -o "Output/"$shortname"_RTshort.fastq.gz" $filename >> $datum"_cutadaptlog.txt"
done
for  filename in *R2.fastq.gz; do
	shortname="${filename:0:-19}"
	revcompname="${filename:0:-19}""_REVCOMP_001.fastq.gz"
	seqkit seq -r -p $filename | gzip -c > $revcompname
	cutadapt -j 0 -a AGCTTGGCGTAACTAGATCT --discard-untrimmed -o "Output/"$shortname"_targetend.fastq.gz" $revcompname >> $datum"_cutadaptlog.txt"
	cutadapt -j 0 -a ctactctaccacttgtaC --discard-untrimmed -o "Output/"$shortname"_3trim.fastq.gz" $revcompname >> $datum"_cutadaptlog.txt"
	cutadapt -j 0 -g AGCTTGGCGTAACTAGATCT -m 6 -M 9 --discard-untrimmed -o "Output/"$shortname"_barcode.fastq.gz" "Output/"$shortname"_3trim.fastq.gz" >> $datum"_cutadaptlog.txt"
done
