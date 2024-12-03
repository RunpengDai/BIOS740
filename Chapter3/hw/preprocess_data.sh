#!/bin/bash

Hipp=/overflow/tengfei/user/tengfei/projects/UKB_Yalin_project/Hipp/extracted_CSV

cat <(ls $Hipp/ADNIGO2/LeftCSV/) <(ls $Hipp/ADNIGO2/RightCSV/) | sort | uniq -c | awk '$1==2' | cut -d'_' -f-3 | uniq | awk '{print $2}' | head -n1000 > selected_samples.txt

mkdir LeftCSV
mkdir RightCSV

while read -r i; do csv=$(cat <(ls $Hipp/ADNIGO2/LeftCSV/) <(ls $Hipp/ADNIGO2/RightCSV/) | grep $i| sort | uniq -c | awk '$1==2{print $2}' | head -n1); cp $Hipp/ADNIGO2/LeftCSV/$csv LeftCSV/; cp $Hipp/ADNIGO2/RightCSV/$csv RightCSV/;  done < selected_samples.txt

gunzip LeftCSV/* 
gunzip RightCSV/*

sed 's/"//g' ADNIMERGE_01Oct2024.csv | cut -d',' -f4,8 | sed '1d' | sort | uniq | grep -wf selected_samples.txt > adni.csv
