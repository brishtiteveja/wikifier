#!/bin/sh
WIKIFIER_HOME=/scratch/conte/a/azehady/nlp/Wikifier2013/Wikifier2013
CONFIG_FILE=FULL.xml
cd $WIKIFIER_HOME

#java -Xmx10G -jar $WIKIFIER_HOME/dist/wikifier-3.0-jar-with-dependencies.jar -trainSvmModuelsOnly $WIKIFIER_HOME/configs/$CONFIG_FILE

AQUAINT_DATA_DIR=$WIKIFIER_HOME/data/WikificationACL2011Data/AQUAINT
java -Xmx10G -jar $WIKIFIER_HOME/dist/wikifier-3.0-jar-with-dependencies.jar -referenceAssistant $AQUAINT_DATA_DIR/Problems $AQUAINT_DATA_DIR/RawTexts $AQUAINT_DATA_DIR/WikifierACL2011Output $WIKIFIER_HOME/configs/$CONFIG_FILE > $WIKIFIER_HOME/run_scripts/Output/ReferenceAssistant_Full.txt