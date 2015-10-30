#!/bin/sh
WIKIFIER_HOME=/scratch/conte/a/azehady/nlp/Wikifier2013/Wikifier2013
CONFIG_FILE=STAND_ALONE_NO_INFERENCE.xml
cd $WIKIFIER_HOME
java -Xmx10G -jar $WIKIFIER_HOME/dist/wikifier-3.0-jar-with-dependencies.jar -trainSvmModuelsOnly $WIKIFIER_HOME/configs/$CONFIG_FILE
