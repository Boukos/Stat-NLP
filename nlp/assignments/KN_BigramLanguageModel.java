package nlp.assignments;

import java.lang.Double;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;

/**
 * A Kneser-Ney smoothed language model using bigram counts
 */
class KN_BigramLanguageModel implements LanguageModel {

  static final String STOP = "</S>";
  static final String START = "<S>";
  
  double discount = 0.5;

  double wordCount = 0.0;
  double vocabSize = 0.0;
  double bigramVocabSize = 0.0;
  double sentenceCount = 0.0;
  
  CounterMap<String, String> bigramCounterMap = new CounterMap<String, String>();
  CounterMap<String, String> reverseBigramCounterMap = new CounterMap<String, String>();
  Counter<String> beforeUNKcounter = new Counter<String>();
  Counter<String> afterUNKcounter = new Counter<String>();
  
  public double unigramCount(String word) {
  	if (word.equals(STOP)) { return sentenceCount; }
  	else if (bigramCounterMap.containsKey(word)) {
  		Counter<String> nextwords = bigramCounterMap.getCounter(word);
  		return nextwords.totalCount();
  	}
  	else { return 0.0; }
  }
  
//  public double getUnigramProbability(List<String> sentence, int index) {
//    String word = sentence.get(index);
//    return unigramCount(word) / wordCount;
//  }
//  
  public double norm(String word, double wordcount) {
  	return (discount / wordcount) * unigramCount(word);
  }
  
  public double p_cont(String word) {
  	return reverseBigramCounterMap.getCounter(word).size() / bigramVocabSize;
  }
  
  public double condBigramProb(String word1, String word2) {
  	if (unigramCount(word1) == 0) { word1 = "UNK"; }
  	if (unigramCount(word2) == 0) { word2 = "UNK"; }
  	double word1count = unigramCount(word1);
  	double bicount = bigramCounterMap.getCount(word1, word2);
    return (Math.max(bicount - discount, 0) / word1count) + (norm(word1, word1count) * p_cont(word2));
  }
  
  public double getCondBigramLogProbability(List<String> sentence, int index) {
  	String word1 = sentence.get(index);
  	String word2 = sentence.get(index+1);
  	double logProb = Math.log(condBigramProb(word1, word2)) / Math.log(2.0);
  	if (((Double)logProb).isNaN() || ((Double)logProb).isInfinite()) { System.out.println("("+word1+", "+word2+"):  "+logProb); }
  	return logProb;
  }

  public double getSentenceProbability(List<String> sentence) {
    List<String> stoppedStartedSentence = new ArrayList<String>(sentence);
    stoppedStartedSentence.add(STOP);
    stoppedStartedSentence.add(0,START);
    double logProbability = 0.0;
    for (int index = 0; index < stoppedStartedSentence.size()-1; index++) {
      logProbability += getCondBigramLogProbability(stoppedStartedSentence, index);
    }
    return logProbability;
  }

  String generateNextWord(String word1) {
    double sample = Math.random();
    double sum = 0.0;
  	Counter<String> nextwords = bigramCounterMap.getCounter(word1);
		for (String word2 : nextwords.keySet()) {
			sum += bigramCounterMap.getCount(word1, word2) / unigramCount(word1);
			if (sum > sample) { return word2; }
		}
  	return "word1: didn't_make_it";
  }
  
//  public double testNextWord(String word) {
//	  double sum = 0.0;
//	  Counter<String> nextwords;
//		
//		if (word.equals("UNK")) {
//			nextwords = afterUNKcounter;
//			System.out.println("unk"+nextwords);
//			for (String word2 : nextwords.keySet()) {
//				sum += languageModel.afterUNKcounter.getCount(word2) / languageModel.afterUNKcounter.totalCount();
//			}
//		}
//		else {
//			double UNKcount = languageModel.beforeUNKcounter.getCount(word);
//			nextwords = languageModel.bigramCounterMap.getCounter(word);
//			nextwords.incrementCount("UNK", UNKcount);
//			System.out.println("notunk"+nextwords);
//	    for (String word2 : nextwords.keySet()) {
//	    	if (word2.equals("UNK")) {
//	    		sum += UNKcount / (languageModel.unigramCount(word) + UNKcount);
//	    	}
//	    	else {
//	        sum += languageModel.bigramCounterMap.getCount(word, word2) / (languageModel.unigramCount(word) + UNKcount);
//	    	}
//	    }
//		}
//		System.out.println("sum:  "+sum);

  public List<String> generateSentence() {
    List<String> sentence = new ArrayList<String>();
    String word = generateNextWord(START);
    while (!word.equals(STOP)) {
      sentence.add(word);
      word = generateNextWord(word);
    }
    return sentence;
  }
  
  public KN_BigramLanguageModel(Collection<List<String>> trainingSet, Set<String> vocab, Collection<List<String>> validSet) {
    for (List<String> sentence : trainingSet) {
      List<String> stoppedStartedSentence = new ArrayList<String>(sentence);
      stoppedStartedSentence.add(STOP);
      stoppedStartedSentence.add(0, START);
      for (int i=0; i < stoppedStartedSentence.size()-1; i++) {
      	String token1 = stoppedStartedSentence.get(i);
      	String token2 = stoppedStartedSentence.get(i+1);
        bigramCounterMap.incrementCount(token1, token2, 1.0);
      }
    }
    for (List<String> sentence : validSet) {
    	List<String> stoppedStartedSentence = new ArrayList<String>(sentence);
      stoppedStartedSentence.add(STOP);
      stoppedStartedSentence.add(0, START);
      for (int i=0; i < stoppedStartedSentence.size()-1; i++) {
      	String token1 = stoppedStartedSentence.get(i);
      	String token2 = stoppedStartedSentence.get(i+1);
      	if (!vocab.contains(token1)) { token1 = "UNK"; }
      	if (!vocab.contains(token2)) { token2 = "UNK"; }
        bigramCounterMap.incrementCount(token1, token2, 1.0);
      }
    }

    wordCount = bigramCounterMap.totalCount();
//    System.out.println("Wordcount:  "+wordCount);
    vocabSize = bigramCounterMap.size();
//    System.out.println("Vocabsize:  "+vocabSize);
    bigramVocabSize = bigramCounterMap.totalSize();
//    System.out.println("BigramVocabsize:  "+bigramVocabSize);
    sentenceCount = trainingSet.size();
//    System.out.println("Sentencecount:  "+sentenceCount);
    
    for (String key : bigramCounterMap.keySet()) {
    	for (String value : bigramCounterMap.getCounter(key).keySet()) {
    		reverseBigramCounterMap.incrementCount(value, key, bigramCounterMap.getCount(key, value));
    	}
    }
//    System.out.println("the cat:  " + bigramCounterMap.getCount("the", "cat"));
//    System.out.println("a cat:  " + bigramCounterMap.getCount("a", "cat"));
//    System.out.println("grade cat:  " + bigramCounterMap.getCount("grade", "cat"));
//    System.out.println(reverseBigramCounterMap.getCounter("cat"));
//    System.out.println( (.5 / 790463.0) * (reverseBigramCounterMap.getCounter("conscript").size() / bigramVocabSize) ) ;
    
  }
}
