#!/bin/sh
WIKIFIER_HOME=/scratch2/azehady/nlp/Wikifier2013/Wikifier2013
CONFIG_FILE=COMPARE_OLD_WIKIFIER.xml
cd $WIKIFIER_HOME

#SVM Training only
java ReferenceAssistant -trainSvmModelsOnly $WIKIFIER_HOME/configs/$CONFIG_FILE >  $WIKIFIER_HOME/run_scripts/Output/Train_SVM_COMPARE_OLD_WIKIFIER.txt

#AQUAINT Data
AQUAINT_DATA_DIR=$WIKIFIER_HOME/data/WikificationACL2011Data/AQUAINT
java -Xmx10G -jar $WIKIFIER_HOME/dist/wikifier-3.0-jar-with-dependencies.jar -referenceAssistant $AQUAINT_DATA_DIR/Problems $AQUAINT_DATA_DIR/RawTexts $AQUAINT_DATA_DIR/WikifierACL2011Output $WIKIFIER_HOME/configs/$CONFIG_FILE > $WIKIFIER_HOME/run_scripts/Output/AQUAINT_ReferenceAssistant_CompareOldWikiFier.txt

#MSNBC Data
MSNBC_DATA_DIR=$WIKIFIER_HOME/data/WikificationACL2011Data/MSNBC
java -Xmx10G -jar $WIKIFIER_HOME/dist/wikifier-3.0-jar-with-dependencies.jar -referenceAssistant $MSNBC_DATA_DIR/Problems $MSNBC_DATA_DIR/RawTexts $MSNBC_DATA_DIR/WikifierACL2011Output $WIKIFIER_HOME/configs/$CONFIG_FILE > $WIKIFIER_HOME/run_scripts/Output/MSNBC_ReferenceAssistant_CompareOldWikiFier.txt

#ACE2004_Coref_Turking_Data
ACE2004_Coref_Turking_DATA_DIR=$WIKIFIER_HOME/data/WikificationACL2011Data/ACE2004_Coref_Turking/Dev
java -Xmx10G -jar $WIKIFIER_HOME/dist/wikifier-3.0-jar-with-dependencies.jar -referenceAssistant $ACE2004_Coref_Turking_DATA_DIR/ProblemsNoTranscripts $ACE2004_Coref_Turking_DATA_DIR/RawTextsNoTranscripts $ACE2004_Coref_Turking_DATA_DIR/WikifierACL2011Output $WIKIFIER_HOME/configs/$CONFIG_FILE > $WIKIFIER_HOME/run_scripts/Output/ACE2004_Coref_Turking_ReferenceAssistant_CompareOldWikiFier.txt

#WikipediaSample Data
WikipediaSample_DATA_DIR=$WIKIFIER_HOME/data/WikificationACL2011Data/WikipediaSample

#Wikipedia data training
java -Xmx10G -jar $WIKIFIER_HOME/dist/wikifier-3.0-jar-with-dependencies.jar -buildTrainingDataAndTrain $WikipediaSample_DATA_DIR/ProblemsTrain $WikipediaSample_DATA_DIR/RawTextsTrain $WIKIFIER_HOME/configs/$CONFIG_FILE $WIKIFIER_HOME/run_scripts/Output/WikipediaSample_Train_Data_ReferenceAssistant_BuildTrain_CompareOldWikifier.txt

#Wikipedia data testing
java -Xmx10G -jar $WIKIFIER_HOME/dist/wikifier-3.0-jar-with-dependencies.jar -referenceAssistant $WikipediaSample_DATA_DIR/ProblemsTest $WikipediaSample_DATA_DIR/RawTextsTest $WikipediaSample_DATA_DIR/WikifierACL2011Output_TestData $WIKIFIER_HOME/configs/$CONFIG_FILE > $WIKIFIER_HOME/run_scripts/Output/WikipediaSample_Test_Data_ReferenceAssistant_CompareOldWikiFier.txt


WIKIFIER_HOME=/scratch2/azehady/nlp/Wikifier2013/Wikifier2013
CONFIG_FILE=COMPARE_OLD_WIKIFIER.xml

/scratch2/azehady/nlp/Wikifier2013/Wikifier2013/data/WikificationACL2011Data/WikipediaSample

-referenceAssistant /scratch2/azehady/nlp/Wikifier2013/Wikifier2013/data/WikificationACL2011Data/WikipediaSample/ProblemsTest /scratch2/azehady/nlp/Wikifier2013/Wikifier2013/data/WikificationACL2011Data/WikipediaSample/RawTextsTest /scratch2/azehady/nlp/Wikifier2013/Wikifier2013/data/WikificationACL2011Data/WikipediaSample/WikifierACL2011Output_TestData /scratch2/azehady/nlp/Wikifier2013/Wikifier2013/configs/DEMO.xml > $WIKIFIER_HOME/run_scripts/Output/WikipediaSample_Test_Data_ReferenceAssistant_CompareOldWikiFier.txt

#Annotation