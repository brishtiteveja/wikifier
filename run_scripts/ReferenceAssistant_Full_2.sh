#!/bin/sh
WIKIFIER_HOME=/scratch2/azehady/nlp/Wikifier2013/Wikifier2013
CONFIG_FILE=FULL.xml
cd $WIKIFIER_HOME

#WikipediaSample Data
WikipediaSample_DATA_DIR=$WIKIFIER_HOME/data/WikificationACL2011Data/WikipediaSample

#Wikipedia data training
java -Xmx10G -jar $WIKIFIER_HOME/dist/wikifier-3.0-jar-with-dependencies.jar -buildTrainingDataAndTrain $WikipediaSample_DATA_DIR/ProblemsTrain $WikipediaSample_DATA_DIR/RawTextsTrain $WIKIFIER_HOME/configs/$CONFIG_FILE 

