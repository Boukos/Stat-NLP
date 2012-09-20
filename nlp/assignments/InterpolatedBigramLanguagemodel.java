package nlp.assignments;


import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;

/**
 * A dummy language model -- uses empirical Bigram counts, plus a single
 * ficticious count for unknown words.
 */
class InterpolatedBigramLanguagemodel implements LanguageModel {

  static final String STOP = "</S>";
  static final String START = "<S>";
  
  double lambda = 0.6;

  double wordCount = 0.0;
  double vocabSize = 0.0;
  double bigramCount = 0.0;
  double bigramVocabSize = 0.0;
  double sentenceCount = 0.0;
  
  CounterMap<String, String> bigramCounterMap = new CounterMap<String, String>();
  
  public double unigramCount(String word) {
  	if (word.equals(STOP)) { return sentenceCount; }
  	else {
  		Counter<String> nextwords = bigramCounterMap.getCounter(word);
  		return nextwords.totalCount();
  	}
  }
  
  public double unigramProb(String token) {
  	double tokencount = unigramCount(token);
  	return tokencount / wordCount;
  }
  
  public double getUnigramProbability(List<String> sentence, int index) {
    String word = sentence.get(index);
    return unigramProb(word);
  }
  
  public double condBigramProb(String word1, String word2) {
  	double bicount = bigramCounterMap.getCount(word1, word2);
  	return bicount / unigramCount(word1);
  }
  
  public double getCondBigramProbability(List<String> sentence, int index) {
  	String word1 = sentence.get(index);
  	String word2 = sentence.get(index+1);
  	return condBigramProb(word1, word2);
  }
  
  public double p_interp(String word1, String word2) {
  	double prob = (lambda * condBigramProb(word1, word2)) + ((1 - lambda) * unigramProb(word2));
//  	if (((Double)prob).isNaN()) { System.out.println(word1+"-"+word2+"-"+word3+": "+prob); }
  	return prob;
  }
  
  public double getP_interp(List<String> sentence, int index) {
  	String word1 = sentence.get(index);
  	String word2 = sentence.get(index+1);
  	return p_interp(word1, word2);
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
    for (String word2 : bigramCounterMap.keySet()) {
      sum += bigramCounterMap.getCount(word1, word2) / unigramCount(word1);
      if (sum > sample) { return word2; }
    }
    return "nope";
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
  
  public InterpolatedBigramLanguagemodel(Collection<List<String>> trainingSet,	Collection<List<String>> validSet) {
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
        bigramCounterMap.incrementCount(token1, token2, 1.0);
      }
    }
    
    wordCount = bigramCounterMap.totalCount() + sentenceCount;
    vocabSize = bigramCounterMap.size();
    sentenceCount = trainingSet.size() + validSet.size();
    bigramCount = bigramCounterMap.totalCount();
    bigramVocabSize = bigramCounterMap.totalSize();
    System.out.println("Wordcount:  "+wordCount);
    System.out.println("Vocabsize:  "+vocabSize);
    System.out.println("Bigramcount:  "+bigramCount);
    System.out.println("BigramVocabsize:  "+bigramVocabSize);
    System.out.println("Sentencecount:  "+sentenceCount);
  }
}
