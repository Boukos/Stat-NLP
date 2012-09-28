package nlp.assignments;


import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;

/**
 * A Kneser-Ney smoothed language model using bigram counts
 */
class KN_BigramLanguageModel implements LanguageModel {

  static final String STOP = "</S>";
  static final String START = "<S>";
  
//  double singletonBicount = 0.0;
//  double doubletonBicount = 0.0;
  double discount = 0.6;

  double wordCount = 0.0;
  double vocabSize = 0.0;
  double bigramVocabSize = 0.0;
  double sentenceCount = 0.0;
  
  CounterMap<String, String> bigramCounterMap = new CounterMap<String, String>();
  CounterMap<String, String> reverseBigramCounterMap = new CounterMap<String, String>();
  
  public double unigramCount(String word) {
  	if (word.equals(STOP)) { return sentenceCount; }
  	else {
  		Counter<String> nextwords = bigramCounterMap.getCounter(word);
  		if (nextwords.totalCount() == 0) { System.out.println("whoops: "+word); }
  		return nextwords.totalCount();
  	}
  }
  
//  public double getUnigramProbability(List<String> sentence, int index) {
//    String word = sentence.get(index);
//    return unigramCount(word) / wordCount;
//  }
//  
  public double norm(String token) {
  	double tokencount = unigramCount(token);
  	return (discount / tokencount) * bigramCounterMap.getCounter(token).size();
  }
  
  public double p_cont(String word) {
  	return reverseBigramCounterMap.getCounter(word).size() / bigramVocabSize;
  }
  
  public double p_interp(String word1, String word2) {
  	double word1count = unigramCount(word1);
  	double bicount = bigramCounterMap.getCount(word1, word2);
    return (Math.max(bicount - discount, 0) / word1count) + (norm(word1) * p_cont(word2));
  }
  
  public double getP_interp(List<String> sentence, int index) {
  	String word1 = sentence.get(index);
  	String word2 = sentence.get(index+1);
  	double prob = p_interp(word1, word2);
  	if (((Double)prob).isNaN()) { System.out.println(word1+"-"+word2+": NaN"); }
  	return prob;
  }

  public double getSentenceProbability(List<String> sentence) {
    List<String> stoppedStartedSentence = new ArrayList<String>(sentence);
    stoppedStartedSentence.add(STOP);
    stoppedStartedSentence.add(0,START);
    double probability = 1.0;
    for (int index = 0; index < stoppedStartedSentence.size()-1; index++) {
      probability *= getP_interp(stoppedStartedSentence, index);
    }
    return probability;
  }

  String generateNextWord(String word1) {
    double sample = Math.random();
    double sum = 0.0;
  	Counter<String> nextwords = bigramCounterMap.getCounter(word1);
		for (String word2 : nextwords.keySet()) {
			sum += bigramCounterMap.getCount(word1, word2) / unigramCount(word1);
			if (sum > sample) { return word2; }
		}
  	return word1+": nope";
  }

  public List<String> generateSentence() {
    List<String> sentence = new ArrayList<String>();
    String word = generateNextWord(START);
    while (!word.equals(STOP)) {
      sentence.add(word);
      word = generateNextWord(word);
    }
    return sentence;
  }
  
  public KN_BigramLanguageModel(Collection<List<String>> trainingSet,	Collection<List<String>> validSet) {
    for (List<String> sentence : trainingSet) {
      List<String> stoppedStartedSentence = new ArrayList<String>(sentence);
      stoppedStartedSentence.add(STOP);
      stoppedStartedSentence.add(0, START);
      for (int i=0; i < stoppedStartedSentence.size()-1; i++) {
      	String token1 = stoppedStartedSentence.get(i);
      	String token2 = stoppedStartedSentence.get(i+1);
        bigramCounterMap.incrementCount(token1, token2, 1.0);
        reverseBigramCounterMap.incrementCount(token2, token1, 1.0);
      }
    }
    
    for (List<String> sentence : validSet) {
    	List<String> stoppedStartedSentence = new ArrayList<String>(sentence);
      stoppedStartedSentence.add(STOP);
      stoppedStartedSentence.add(0, START);
      for (int i=0; i < stoppedStartedSentence.size()-1; i++) {
      	String token1 = stoppedStartedSentence.get(i);
      	String token2 = stoppedStartedSentence.get(i+1);
        bigramCounterMap.incrementCount(token1, token2, 1.0);
        reverseBigramCounterMap.incrementCount(token2, token1, 1.0);
      }
    }

    wordCount = bigramCounterMap.totalCount() + sentenceCount;
    vocabSize = bigramCounterMap.size();
    bigramVocabSize = bigramCounterMap.totalSize();
    sentenceCount = trainingSet.size() + validSet.size();
    System.out.println("Wordcount:  "+wordCount);
    System.out.println("Vocabsize:  "+vocabSize);
    System.out.println("BigramVocabsize:  "+bigramVocabSize);
    System.out.println("Sentencecount:  "+sentenceCount);
    
//    System.out.println("the cat:  " + bigramCounterMap.getCount("the", "cat"));
//    System.out.println("a cat:  " + bigramCounterMap.getCount("a", "cat"));
//    System.out.println("grade cat:  " + bigramCounterMap.getCount("grade", "cat"));
//    System.out.println(reverseBigramCounterMap.getCounter("cat"));
//    System.out.println( (.5 / 790463.0) * (reverseBigramCounterMap.getCounter("conscript").size() / bigramVocabSize) ) ;
    
//    for (String word1 : bigramCounterMap.keySet()) {
//    	for (String word2 : bigramCounterMap.getCounter(word1).keySet()) {
//    		if (bigramCounterMap.getCount(word1, word2) == 1) { singletonBicount += 1; }
//    		else if (bigramCounterMap.getCount(word1, word2) == 2) { doubletonBicount += 1; }
//    	}
//    }
//    discount = singletonBicount / (singletonBicount + 2 * doubletonBicount);
//    System.out.println("singletonTricount: "+singletonBicount);
//    System.out.println("doubletonTricount: "+doubletonBicount);
//    System.out.println("discount: "+discount);  
  }
}
