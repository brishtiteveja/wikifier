#!/bin/sh
WIKIFIER_HOME=/scratch/conte/a/azehady/nlp/Wikifier2013/Wikifier2013
CONFIG_FILE=COMPARE_OLD_WIKIFIER
cd $WIKIFIER_HOME
java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -annotateData data/testSample/sampleText/test.txt data/Output/"$CONFIG_FILE".txt false configs/"$CONFIG_FILE".xml
