#!/bin/sh
#
# The script takes a single parameter -- the ../Config filename
#java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -annotateData data/testSample/sampleText/test.txt data/testSample/sampleOutput/ false configs/STAND_ALONE_NO_INFERENCE.xml
#[1]
java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -trainSvmModuelsOnly configs/Baseline.xml
java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -trainSvmModuelsOnly configs/COMPARE_OLD_WIKIFIER.xml
java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -trainSvmModuelsOnly configs/COREF.xml
java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -trainSvmModuelsOnly configs/DEFAULT.xml
java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -trainSvmModuelsOnly configs/DEMO.xml
java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -trainSvmModuelsOnly configs/FULL.xml
java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -trainSvmModuelsOnly configs/FULL_UNAMBIGUOUS.xml
java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -trainSvmModuelsOnly configs/LEXICAL_SEARCH.xml
java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -trainSvmModuelsOnly configs/STAND_ALONE_GUROBI.xml
java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -trainSvmModuelsOnly configs/STAND_ALONE_NO_INFERENCE.xml
java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -trainSvmModuelsOnly configs/TAC.xml
