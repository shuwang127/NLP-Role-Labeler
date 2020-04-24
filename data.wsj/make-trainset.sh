#! /bin/bash

# name of the output file 
FILE="train-set" 

cat words/test.wsj.words.train > /tmp/$FILE.words # words 
cat props/test.wsj.props.train > /tmp/$FILE.props # target + args

## Choose syntax
cat synt.cha/test.wsj.synt.cha.train > /tmp/$FILE.synt #full syntax parse 
cat ne/test.wsj.ne.train > /tmp/$FILE.ne #named entity

paste -d ' ' /tmp/$FILE.words /tmp/$FILE.synt /tmp/$FILE.ne /tmp/$FILE.props > /tmp/$FILE.section.txt

echo Generating file $FILE.txt
cat /tmp/$FILE.section* > $FILE.txt

echo Cleaning files
rm -f /tmp/$FILE-*
