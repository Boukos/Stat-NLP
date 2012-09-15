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
  
  double discount = 0.5;

  double wordCount = 0.0;
  double vocabSize = 0.0;
  double bigramVocabSize = 0.0;
  double sentenceCount = 0.0;
  
  CounterMap<String, String> bigramCounterMap = new CounterMap<String, String>();
  
  public double unigramCount(String word) {
  	if (word.equals(STOP)) { return sentenceCount; }
  	else if (bigramCounterMap.containsKey(word)) {
  		Counter<String> nextwords = bigramCounterMap.getCounter(word);
  		return nextwords.totalCount();
  	}
  	else { return 0.0; }
  }
  
  public double getUnigramProbability(List<String> sentence, int index) {
    String word = sentence.get(index);
    return unigramCount(word) / wordCount;
  }
  
  public double norm(String word, double wordcount) {
  	return (discount / wordcount) * unigramCount(word);
  }
  
  public double p_cont(String word) {
  	double total = 0.0;
  	for (String key : bigramCounterMap.keySet()) {
  		if (bigramCounterMap.getCounter(key).keySet().contains(word)) { total += 1; }
  	}
  	return total / bigramVocabSize;
  }
  
  public double condBigramProb(String word1, String word2) {
  	double word1count = unigramCount(word1);
  	double bicount = bigramCounterMap.getCount(word1, word2);
    return (Math.max(bicount - discount, 0) / word1count) + (norm(word1, word1count) * p_cont(word2));
  }
  
  public double getCondBigramProbability(List<String> sentence, int index) {
  	String word1 = sentence.get(index);
  	String word2 = sentence.get(index+1);
  	return condBigramProb(word1, word2);
  }

  public double getSentenceProbability(List<String> sentence) {
    List<String> stoppedStartedSentence = new ArrayList<String>(sentence);
    stoppedStartedSentence.add(STOP);
    stoppedStartedSentence.add(0,START);
    double probability = 1.0;
    for (int index = 0; index < stoppedStartedSentence.size()-1; index++) {
      probability *= getCondBigramProbability(stoppedStartedSentence, index);
    }
    return probability;
  }

  String generateNextWord(String word1) {
  	Counter<String> nextwords = bigramCounterMap.getCounter(word1);
    double sample = Math.random();
    double sum = 0.0;
    for (String word2 : nextwords.keySet()) {
      sum += bigramCounterMap.getCount(word1, word2) / unigramCount(word1);
      if (sum > sample) { return word2; }
    }
    return "*UNKNOWN*";
  }

  public List<String> generateSentence() {
    List<String> sentence = new ArrayList<String>();
    String word = generateNextWord(START);
    int loop_count = 0;
    while (!word.equals(STOP) && (loop_count < 15)) {
      sentence.add(word);
      word = generateNextWord(word);
      loop_count += 1;
    }
    return sentence;
  }
  
  public KN_BigramLanguageModel(Collection<List<String>> sentenceCollection) {
    for (List<String> sentence : sentenceCollection) {
      List<String> stoppedStartedSentence = new ArrayList<String>(sentence);
      stoppedStartedSentence.add(STOP);
      stoppedStartedSentence.add(0, START);
      for (int i=0; i < stoppedStartedSentence.size()-1; i++) {
      	String token1 = stoppedStartedSentence.get(i);
      	String token2 = stoppedStartedSentence.get(i+1);
        bigramCounterMap.incrementCount(token1, token2, 1.0);
      }
    }
    wordCount = bigramCounterMap.totalCount();
//    System.out.println("Wordcount:  "+wordCount);
    vocabSize = bigramCounterMap.size();
//    System.out.println("Vocabsize:  "+vocabSize);
    bigramVocabSize = bigramCounterMap.totalSize();
//    System.out.println("BigramVocabsize:  "+bigramVocabSize);
    sentenceCount = sentenceCollection.size();
//    System.out.println("Sentencecount:  "+sentenceCount);
  }
}
