#! /bin/bash

# name of the output file 
FILE="dev-set" 

cat words/test.wsj.words.test > /tmp/$FILE.words
cat props/test.wsj.props.test > /tmp/$FILE.props

## Choose syntax
cat synt.cha/test.wsj.synt.cha.test > /tmp/$FILE.synt
cat ne/test.wsj.ne.test > /tmp/$FILE.ne

paste -d ' ' /tmp/$FILE.words /tmp/$FILE.synt /tmp/$FILE.ne /tmp/$FILE.props > /tmp/$FILE.section.txt

echo Generating file $FILE.txt
cat /tmp/$FILE.section* > $FILE.txt

echo Cleaning files
rm -f /tmp/$FILE-*

