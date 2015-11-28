package edu.illinois.cs.cogcomp.wikifier.models.tfidf;

import edu.illinois.cs.cogcomp.wikifier.models.WikipediaProtobuffers.LexicalTitleDataInfoProto;
import edu.illinois.cs.cogcomp.wikifier.utils.datastructure.SortedObjects;
import gnu.trove.iterator.TIntDoubleIterator;
import gnu.trove.map.hash.TIntDoubleHashMap;

import java.io.IOException;
import java.io.Serializable;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.io.File;
import java.util.Collection;



// Word2vec import
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.Collection;



public class TFIDF {
    public long docsCount = 0;
    public static FeatureMap map = null;
    public int[] IDF = null;
    public static Word2Vec wvec;
    public static int wordVecDimension = 300;
    public String wordVecFile = "/scratch2/azehady/nlp/Wikifier2013/Wikifier2013/data/Word-Vectors/GoogleNews-vectors-negative300.bin";
    
    public void setWordVecFile(String fileName){
    	this.wordVecFile = fileName; 
    }
    
    public TFIDF(FeatureMap _map) {
        docsCount = 0;
        map = _map;
        IDF = new int[map.dim];
        
        if (wvec != null) {
        	wvec = this.getWordVectors();
        }
    }

    public void updateIDF(Document doc) {
        docsCount++;
        for (int fid : doc.getActiveFid(map))
            IDF[fid]++;
    }

    public void updateIDF(UnigramStatistics stat) {
        docsCount++;
        for (String w : stat.wordCounts.keySet()) {
            if (map.wordToFid.containsKey(w))
                IDF[map.wordToFid.get(w)]++;
        }
    }

    public TF_IDF_Doc getRepresentation(Document d, boolean disableIdfWeighting) {
        UnigramStatistics stat = new UnigramStatistics(true);
        for (String word : d.words)
            stat.addWord(word);
        return getRepresentation(stat, disableIdfWeighting);
    }

    public TF_IDF_Doc getRepresentation(UnigramStatistics stat, boolean disableIdfWeighting) {

        List<Integer> features = new ArrayList<Integer>();
        List<Double> weights = new ArrayList<Double>();

        for (String s : stat.wordCounts.keySet()) {

            if (map.wordToFid.containsKey(s)) {
                int fid = map.wordToFid.get(s);
                if (stat.wordCounts.containsKey(s) && IDF[fid] > 0) {
                    features.add(fid);
                    if (!disableIdfWeighting)
                        weights.add(stat.wordCounts.get(s) * Math.log(((double) docsCount) / ((double) IDF[fid])));
                    else
                        weights.add(stat.wordCounts.get(s).doubleValue());
                }

            }
        }
        
        TF_IDF_Doc doc = new TF_IDF_Doc(features.size(), false);

        for (int i = 0; i < features.size(); i++) {
            doc.activeFids[i] = features.get(i);
            doc.tfIdfWeight[i] = weights.get(i);
        }
        return doc;
    }

    public static int[] getIDF(DocumentCollection docCollection, FeatureMap map) {
        TFIDF tfidf = new TFIDF(map);
        for (int i = 0; i < docCollection.docs.size(); i++)
            tfidf.updateIDF(docCollection.docs.get(i));
        return tfidf.IDF;
    }

    public static List<TF_IDF_Doc> getRepresentation(DocumentCollection docCollection, FeatureMap map, boolean disableIdfWeighting) {
        System.out.println("Building TF IFD Representation");
        TFIDF tfidf = new TFIDF(map);
        for (Document doc:docCollection)
            tfidf.updateIDF(doc);
        List<TF_IDF_Doc> docs = new ArrayList<>();
        for (Document doc:docCollection)
            docs.add(tfidf.getRepresentation(doc, disableIdfWeighting));
        System.out.println("***Done*** Building TF IFD Representation");
        return docs;
    }

    public static double getCosineSim(TF_IDF_Doc d1, TF_IDF_Doc d2) {
        if (d1.featureMap != null && d2.featureMap != null)
            return getCosineSimHashMaps(d1, d2);
        return getCosineSimVectors(d1, d2);
    }
    

    public static double getCosineSimHashMaps(TF_IDF_Doc d1, TF_IDF_Doc d2) {
        double nominator = 0;
        TF_IDF_Doc shorter = d1;
        TF_IDF_Doc longer = d2;
        if (d1.featureMap.size() > d2.featureMap.size()) {
            shorter = d2;
            longer = d1;
        }
        
        for(TIntDoubleIterator it = shorter.featureMap.iterator();it.hasNext();){
            it.advance();
            int fid = it.key();
            if (longer.featureMap.containsKey(fid))
                nominator += shorter.featureMap.get(fid) * longer.featureMap.get(fid);
        }
        
//        for (Iterator<Entry<Integer, Double>> i = shorter.featureMap.entrySet().iterator(); i.hasNext();) {
//            Entry<Integer, Double> e = i.next();
//            int fid = e.getKey();
//            if (longer.featureMap.containsKey(fid))
//                nominator += shorter.featureMap.get(fid) * longer.featureMap.get(fid);
//        }
        if (nominator == 0)
            return 0;
        double norm1 = shorter.getWeightNorm();
        double norm2 = longer.getWeightNorm();
        return nominator / (norm1 * norm2);
    }
    
 //  d1 =  I love Purdue
 // activeFid    =   1     2     3
 // TFIDFWeight  =   0.4  0.6   0.7
 //   h[1] = 0.4
 //   h[2] = 0.6
 //   h[3] = 0.7
    
//    wv[1] = [3 5 7 9 5]
//    wv[2] = [1 2 3 4 5]
//    wv[3] = [9 2 4 6 7]
//    wv_avg1 = [a1 a2 a3 a4 a5]  a1 = (3 + 1 + 9) / 3
    
//  d2 =  We are engineers
// activeFid    =   1     2     3
// TFIDFWeight  =   0.4  0.6   0.7
//   h[1] = 0.4
//   h[2] = 0.6
//   h[3] = 0.7
       
//       wv[1] = [3 5 7 9 5]
//       wv[2] = [1 2 3 4 5]
//       wv[3] = [9 2 4 6 7]
//       wv_avg2 = [a1 a2 a3 a4 a5]  a1 = (3 + 1 + 9) / 3
      
// wv_avg1 . wv_avg2
    
    public static double[] sumTwoWordVec(double[] wv1, double[] wv2) {
    	double[] resV = new double[wv1.length];
    	
    	for (int i = 0; i < wv1.length; i++) {
    		resV[i] = wv1[i] + wv2[i];
    	}
    	
    	return resV;
    }
    
    public static double getCosineSimWithWordVec(TF_IDF_Doc d1, TF_IDF_Doc d2) {
    	if (map == null) {
    		System.out.println("TFIDF class not yet created. Featuremap hasn't been initialized.");
    		return 0;
    	}

        double nominator = 0;
        
        double sum_wv1[] = new double[TFIDF.wordVecDimension];
        for (int i = 0; i < d1.activeFids.length; i++) {
        	int fid = d1.activeFids[i];
        	String w = TFIDF.map.fidToWord.get(fid);
        	double wv[] = TFIDF.wvec.getWordVector(w);
        	
        	sum_wv1 = TFIDF.sumTwoWordVec(sum_wv1, wv);
        }
        for (int i = 0; i < sum_wv1.length; i++) { 
        	sum_wv1[i] /= sum_wv1.length;
        }
        
        
        double sum_wv2[] = new double[TFIDF.wordVecDimension];
        for (int i = 0; i < d2.activeFids.length; i++) {
        	int fid = d2.activeFids[i];
        	String w = TFIDF.map.fidToWord.get(fid);
        	double wv[] = TFIDF.wvec.getWordVector(w);
        	
        	sum_wv2 = sumTwoWordVec(sum_wv1, wv);
        }
        for (int i = 0; i < sum_wv2.length; i++) { 
        	sum_wv2[i] /= sum_wv1.length;
        }
        
        
        for (int i = 0; i < sum_wv1.length; i++)
        	nominator += sum_wv1[i] * sum_wv2[i];

        if (nominator == 0)
            return 0;

        double norm1 = 0;
        for (int i = 0; i < sum_wv1.length; i++)
            norm1 += sum_wv1[i] * sum_wv1[i];
        norm1 = Math.sqrt(norm1);

        double norm2 = 0;
        for (int i = 0; i < sum_wv2.length; i++)
            norm2 += sum_wv2[i] * sum_wv2[i];
        norm2 = Math.sqrt(norm2);

        return nominator / (norm1 * norm2);
    }

    public static double getCosineSimVectors(TF_IDF_Doc d1, TF_IDF_Doc d2) {
        double nominator = 0;
        TF_IDF_Doc shorter = d1;
        TF_IDF_Doc longer = d2;
        if (d1.activeFids.length > d2.activeFids.length) {
            shorter = d2;
            longer = d1;
        }
        
        
//                     a1. a2
//      cos theta =  -----------------
//                   sqrt(a1) sqrt(a2)
        
        TIntDoubleHashMap h = new TIntDoubleHashMap();
        for (int i = 0; i < longer.activeFids.length; i++)
            h.put(longer.activeFids[i], longer.tfIdfWeight[i]);
        
        
//          love = [d11 d12 d13 d14 d15]

//          hate = [d11 d12 d13 d14 d15]
        
//           x1 = [1 3 5 6 7]
//           x2 = [4 7 9 0 0]
        
//        d1 = [love word hate go] = [a1 a2 a3 a4 a5]
//        d2 = [love hate come]    = [b1 b2 b3 b4 b5]
        
//        nom = d1_TI[love]* d2_TI[love] + d1_TI[hate] * d2_TI[hate] + d1_TI[word] * d2_TI[word] + d1_TI[come] * d2_TI[come] + d1_TI[go] * d2_TI[go]
//            = d1_TI[love]* d2_TI[love] + d1_TI[hate] * d2_TI[hate] + d1_TI[word] * 0           + 0           * d2_TI[come] + d1_TI[go] * 0
                
        
        for (int i = 0; i < shorter.activeFids.length; i++)
            if (h.containsKey(shorter.activeFids[i]))
                nominator += shorter.tfIdfWeight[i] * h.get(shorter.activeFids[i]);
        
        if (nominator == 0)
            return 0;
        
        double norm1 = 0;
        for (int i = 0; i < shorter.activeFids.length; i++)
            norm1 += shorter.tfIdfWeight[i] * shorter.tfIdfWeight[i];
        
        norm1 = Math.sqrt(norm1);
        
        double norm2 = 0;
        for (int i = 0; i < longer.activeFids.length; i++)
            norm2 += longer.tfIdfWeight[i] * longer.tfIdfWeight[i];
        norm2 = Math.sqrt(norm2);
        
        return nominator / (norm1 * norm2);
    }

    public static class TF_IDF_Doc implements Serializable {
        /**
		 * 
		 */
        private static final long serialVersionUID = 2338008209620619317L;
        
        public int[] activeFids = null;
        public double[] tfIdfWeight = null;
        // the hashmap reprsentation is only partially supported, and should be used onlt when the
        // same document is used many times
        // for TF-IDF similarity purposes, and we need to pre-compute the hashmaps!
        public TIntDoubleHashMap featureMap = null;
        // featureMap.get(activeFids[i])=tfIdfWeight[i].
        // This will speed things up, but will
        // require way more memory

        public TF_IDF_Doc(LexicalTitleDataInfoProto lexInfo) {
            featureMap = new TIntDoubleHashMap();
            for (int j = 0; j < lexInfo.getContextTokensFidsCount(); j++) {
                int fid = lexInfo.getContextTokensFids(j);
                featureMap.put(fid, lexInfo.getContextTokensFidsWeights(j));
            }

            for (int j = 0; j < lexInfo.getTextTokensFidsCount(); j++) {
                int fid = lexInfo.getTextTokensFids(j);
                double currentWeight = featureMap.containsKey(fid) ? featureMap.get(fid) : 0.0;
                featureMap.put(fid, currentWeight + lexInfo.getTextTokensFidsWeights(j));
            }
        }
        
        public double getWeightNorm(){
            double[] weights = tfIdfWeight==null? featureMap.values() : tfIdfWeight;
            double norm = 0;
            for(double w:weights){
                norm += w*w;
            }
            return Math.sqrt(norm);
        }
        
        /**
         * Creates a reweighted doc from another
         * @param original
         * @param weightedDocCount
         * @param weightedIdfCount
         */
        public TF_IDF_Doc(TF_IDF_Doc original,double weightedDocCount,TIntDoubleHashMap weightedIdfCount){
            this(original.featureMap.size(), true);
            for(TIntDoubleIterator it = original.featureMap.iterator();it.hasNext();){
                it.advance();
                int fid = it.key();
                double weight = it.value();
                featureMap.put(fid, weight * Math.log(weightedDocCount/weightedIdfCount.get(fid)));
            }
        }

        public TF_IDF_Doc(int numberOfActiveFeatures, boolean useHashMapRepresentation) {
            if (useHashMapRepresentation) {
                featureMap = new TIntDoubleHashMap();
            } else {
                activeFids = new int[numberOfActiveFeatures];
                tfIdfWeight = new double[numberOfActiveFeatures];
            }
        }

        // assigns weight 1 to all the features
        public TF_IDF_Doc(int[] activeFeatures) {
            activeFids = activeFeatures;
            tfIdfWeight = new double[activeFeatures.length];
            for (int i = 0; i < activeFeatures.length; i++)
                tfIdfWeight[i] = 1;
        }

        public TF_IDF_Doc filterTopWords(int wordsNum) {
            if (wordsNum >= activeFids.length)
                return this;
            SortedObjects<Integer> top = new SortedObjects<Integer>(wordsNum);
            TF_IDF_Doc res = new TF_IDF_Doc(wordsNum, false);
            for (int i = 0; i < activeFids.length; i++)
                top.add(new Integer(activeFids[i]), tfIdfWeight[i]);
            for (int i = 0; i < wordsNum; i++) {
                res.activeFids[i] = top.topObjects.get(i);
                res.tfIdfWeight[i] = top.topScores.get(i);
            }
            return res;
        }

        public TF_IDF_Doc filterFids(int[] remainingFids) {
            boolean[] mask = new boolean[activeFids.length];
            int count = 0;
            for (int i = 0; i < activeFids.length; i++) {
                for (int j = 0; j < remainingFids.length; j++)
                    if (activeFids[i] == remainingFids[j])
                        mask[i] = true;
                if (mask[i])
                    count++;
            }
            TF_IDF_Doc res = new TF_IDF_Doc(count, false);
            int pos = 0;
            for (int i = 0; i < activeFids.length; i++)
                if (mask[i]) {
                    res.activeFids[pos] = activeFids[i];
                    res.tfIdfWeight[pos] = tfIdfWeight[i];
                    pos++;
                }
            return res;
        }

        public String toString(FeatureMap map) {
            String res = "";
            for (int i = 0; i < activeFids.length; i++)
                if (map.fidToWord.containsKey(activeFids[i]))
                    res += map.fidToWord.get(activeFids[i]) + " ";
            return res;
        }

    }
    
    public Word2Vec getWordVectors() {
    	Word2Vec wvec = null;
    	try {
    		wvec = WordVectorSerializer.loadGoogleModel(new File(this.wordVecFile),true, false);
			System.out.println("Hello");
    	} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	
    	return wvec;
    }
    
    public static void main(String[] args) {
    	ParagraphVectors pf = null;
    	
        // test this class!
        String[] d1 = { "joy", "fun", "love" };
        String[] d2 = { "amazing", "love", "bring", "joy" };
        String[] d3 = { "I", "bring", "amazing", "joy", "in" };
        Document dd1 = new Document(d1, 1, null);
        Document dd2 = new Document(d2, 1, null);
        Document dd3 = new Document(d3, 1, null);
        DocumentCollection docs = new DocumentCollection();
        docs.addDoc(dd1);
        docs.addDoc(dd2);
        docs.addDoc(dd3);
        FeatureMap map = new FeatureMap();
        map.addDocs(docs, 0, true);
        List<TF_IDF_Doc> docs2 = TFIDF.getRepresentation(docs, map, false);
        
        Word2Vec wvec = TFIDF.wvec; 
        
        for (int i = 0; i < docs2.size(); i++) {
        	double[][] wv = new double[docs2.get(i).activeFids.length][];
        	double[] dv = null;
            for (int j = 0; j < docs2.get(i).activeFids.length; j++) {
                int fid = docs2.get(i).activeFids[j];
                System.out.print(map.fidToWord.get(fid) + "(" + docs2.get(i).tfIdfWeight[j] + ") ");
                
                String word = map.fidToWord.get(fid);
                wv[j] = wvec.getWordVector(word);
                System.out.print(word + " = " +"[");
                for (double d: wv[j]) {
                	System.out.print(d + ", ");
                }
                if (dv == null) {
                	dv = new double[wv[j].length];
                	for (int k = 0; k < wv[j].length; k++) {
                		dv[k] = wv[j][k]; 
                	}
                } else {
                	for (int k = 0; k < wv[j].length; k++) {
                		dv[k] += wv[j][k]; 
                	}
                }
                System.out.print("]");
                System.out.println();
            }
            System.out.println("Doc vec for doc " + i + "= [ ");
            for (int k = 0; k < dv.length; k++) {
            	System.out.print(dv[k] / docs2.get(i).activeFids.length + ", ");
        	}
            System.out.print("] ");
            System.out.println();
        }
//        System.out.println("\n\nfiltered version:");
//        for (int i = 0; i < docs2.size(); i++) {
//            TF_IDF_Doc doc = docs2.get(i).filterTopWords(1);
//            for (int j = 0; j < doc.activeFids.length; j++) {
//                int fid = doc.activeFids[j];
//                System.out.print(map.fidToWord.get(fid) + "(" + doc.tfIdfWeight[j] + ") ");
//            }
//            System.out.println("");
//        }
    }
}
