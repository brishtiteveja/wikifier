#!/bin/sh
WIKIFIER_HOME=/scratch2/azehady/nlp/Wikifier2013/Wikifier2013
CONFIG_FILE=FULL.xml
cd $WIKIFIER_HOME

#SVM Training only
#java ReferenceAssistant -trainSvmModelsOnly $WIKIFIER_HOME/configs/$CONFIG_FILE >  $WIKIFIER_HOME/run_scripts/Output/Train_SVM_FULL.txt

#AQUAINT Data
#AQUAINT_DATA_DIR=$WIKIFIER_HOME/data/WikificationACL2011Data/AQUAINT
#java -Xmx10G -jar $WIKIFIER_HOME/dist/wikifier-3.0-jar-with-dependencies.jar -referenceAssistant $AQUAINT_DATA_DIR/Problems $AQUAINT_DATA_DIR/RawTexts $AQUAINT_DATA_DIR/WikifierACL2011Output $WIKIFIER_HOME/configs/$CONFIG_FILE > $WIKIFIER_HOME/run_scripts/Output/AQUAINT_ReferenceAssistant_Full.txt

#MSNBC Data
#MSNBC_DATA_DIR=$WIKIFIER_HOME/data/WikificationACL2011Data/MSNBC
#java -Xmx10G -jar $WIKIFIER_HOME/dist/wikifier-3.0-jar-with-dependencies.jar -referenceAssistant $MSNBC_DATA_DIR/Problems $MSNBC_DATA_DIR/RawTexts $MSNBC_DATA_DIR/WikifierACL2011Output $WIKIFIER_HOME/configs/$CONFIG_FILE > $WIKIFIER_HOME/run_scripts/Output/MSNBC_ReferenceAssistant_Full.txt

#ACE2004_Coref_Turking_Data
#ACE2004_Coref_Turking_DATA_DIR=$WIKIFIER_HOME/data/WikificationACL2011Data/ACE2004_Coref_Turking/Dev
#java -Xmx10G -jar $WIKIFIER_HOME/dist/wikifier-3.0-jar-with-dependencies.jar -referenceAssistant $ACE2004_Coref_Turking_DATA_DIR/ProblemsNoTranscripts $ACE2004_Coref_Turking_DATA_DIR/RawTextsNoTranscripts $ACE2004_Coref_Turking_DATA_DIR/WikifierACL2011Output $WIKIFIER_HOME/configs/$CONFIG_FILE > $WIKIFIER_HOME/run_scripts/Output/ACE2004_Coref_Turking_ReferenceAssistant_Full.txt

#WikipediaSample Data
WikipediaSample_DATA_DIR=$WIKIFIER_HOME/data/WikificationACL2011Data/WikipediaSample

#Wikipedia data training
java -Xmx10G -jar $WIKIFIER_HOME/dist/wikifier-3.0-jar-with-dependencies.jar -buildTrainingDataAndTrain $WikipediaSample_DATA_DIR/ProblemsTrain $WikipediaSample_DATA_DIR/RawTextsTrain $WIKIFIER_HOME/configs/$CONFIG_FILE 

#Wikipedia data testing
#java -Xmx10G -jar $WIKIFIER_HOME/dist/wikifier-3.0-jar-with-dependencies.jar -referenceAssistant $WikipediaSample_DATA_DIR/ProblemsTest $WikipediaSample_DATA_DIR/RawTextsTest $WikipediaSample_DATA_DIR/WikifierACL2011Output_TestData $WIKIFIER_HOME/configs/$CONFIG_FILE > $WIKIFIER_HOME/run_scripts/Output/WikipediaSample_Test_Data_ReferenceAssistant_Full.txt


#Annotation

#Eclipse arg
#AQUAINT Data
#-referenceAssistant /scratch2/azehady/nlp/Wikifier2013/Wikifier2013/data/WikificationACL2011Data/AQUAINT/Problems /scratch2/azehady/nlp/Wikifier2013/Wikifier2013/data/WikificationACL2011Data/AQUAINT/RawTexts /scratch2/azehady/nlp/Wikifier2013/Wikifier2013/data/WikificationACL2011Data/AQUAINT/WikifierACL2011Output /scratch2/azehady/nlp/Wikifier2013/Wikifier2013/configs/FULL.xml > /scratch2/azehady/nlp/Wikifier2013/Wikifier2013/run_scripts/Output/AQUAINT_ReferenceAssistant_Full.txt

#WikipediaSample Data Test
#-referenceAssistant /scratch2/azehady/nlp/Wikifier2013/Wikifier2013/data/WikificationACL2011Data/WikipediaSample/ProblemsTest /scratch2/azehady/nlp/Wikifier2013/Wikifier2013/data/WikificationACL2011Data/WikipediaSample/RawTextsTest /scratch2/azehady/nlp/Wikifier2013/Wikifier2013/data/WikificationACL2011Data/WikipediaSample/WikifierACL2011Output_TestData /scratch2/azehady/nlp/Wikifier2013/Wikifier2013/configs/FULL.xml > /scratch2/azehady/nlp/Wikifier2013/Wikifier2013/run_scripts/Output/WikipediaSample_Test_ReferenceAssistant_Full.txt

