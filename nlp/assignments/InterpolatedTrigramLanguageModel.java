package nlp.assignments;


import java.util.ArrayList;

import java.util.Collection;
import java.util.List;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;
import nlp.util.Pair;


/**
 * A dummy language model -- uses empirical trigram counts, plus a single
 * ficticious count for unknown words.
 */
class InterpolatedTrigramLanguageModel implements LanguageModel {

  static final String STOP = "</S>";
  static final String START = "<S>";
  
  double lambda3 = 0.7;
  double lambda2 = 0.2;
  double lambda1 = 0.1;

  double wordCount = 0.0;
  double vocabSize = 0.0;
  double bigramCount = 0.0;
  double bigramVocabSize = 0.0;
  double trigramCount = 0.0;
  double trigramVocabSize = 0.0;
  double sentenceCount = 0.0;
  
  CounterMap<String, String> bigramCounterMap = new CounterMap<String, String>();
  CounterMap<String,Pair<String,String>> ubTrigramCounterMap = new CounterMap<String,Pair<String,String>>();
  CounterMap<Pair<String,String>,String> buTrigramCounterMap = new CounterMap<Pair<String,String>,String>();
   
  public double trueUnigramCount(String word) {
  	if (word.equals(STOP)) { return sentenceCount; }
  	else {
  		Counter<String> nextwords = bigramCounterMap.getCounter(word);
  		return nextwords.totalCount();
  	}
  }
  
  public double unigramProb(String token) {
  	double tokencount = trueUnigramCount(token);
  	return tokencount / wordCount;
  }
  
  public double condBigramProb(String word1, String word2) {
  	double bicount = bigramCounterMap.getCount(word1, word2);
  	return bicount / trueUnigramCount(word1);
  }
  
  public double getCondBigramProbability(List<String> sentence, int index) {
  	String word1 = sentence.get(index);
  	String word2 = sentence.get(index+1);
  	return condBigramProb(word1, word2);
  }
  
  public double condTrigramProb(String word1, String word2, String word3) {
  	Pair<String,String> bigram = Pair.makePair(word2, word3);
  	double tricount = ubTrigramCounterMap.getCount(word1, bigram);
		double bicount = bigramCounterMap.getCount(word1, word2);
  	if (bicount > 0) { 
//  		System.out.println("Found it! "+word1+"-"+word2+": "+bicount);
  		return tricount / bicount;
  		}
  	else { return 1 / vocabSize; }
  }
  
  public double getCondTrigramProbability(List<String> sentence, int index) {
  	String word1 = sentence.get(index);
  	String word2 = sentence.get(index+1);
  	String word3 = sentence.get(index+2);
  	return condTrigramProb(word1, word2, word3);
  }
  
  public double p_interp(String word1, String word2, String word3) {
  	double prob = (lambda3 * condTrigramProb(word1, word2, word3)) + (lambda2 * condBigramProb(word2, word3)) +
  			(lambda1 * unigramProb(word3));
//  	if (((Double)prob).isNaN()) { System.out.println(word1+"-"+word2+"-"+word3+": "+prob); }
  	return prob;
  }
  
  public double getP_interp(List<String> sentence, int index) {
  	String word1 = sentence.get(index);
  	String word2 = sentence.get(index+1);
  	String word3 = sentence.get(index+2);
  	return p_interp(word1, word2, word3);
  }

  public double getSentenceProbability(List<String> sentence) {
    List<String> stoppedStartedSentence = new ArrayList<String>(sentence);
    stoppedStartedSentence.add(STOP);
    stoppedStartedSentence.add(0,START);
    double probability = 1.0;
    for (int index = 0; index < stoppedStartedSentence.size()-2; index++) {
      probability *= getP_interp(stoppedStartedSentence, index);
    }
    return probability;
  }
  
  String generateFirstWord() {
    double sample = Math.random();
    double sum = 0.0;
  	Counter<String> nextwords = bigramCounterMap.getCounter(START);
		for (String word2 : nextwords.keySet()) {
			sum += bigramCounterMap.getCount(START, word2) / trueUnigramCount(START);
			if (sum > sample) { return word2; }
		}
  	return START+": didn't_make_it";
  }

  String generateNextWord(Pair<String,String> bigram) {
  	String word1 = bigram.getFirst();
  	String word2 = bigram.getSecond();
    double sample = Math.random();
    double sum = 0.0;
  	Counter<String> nextwords = buTrigramCounterMap.getCounter(bigram);
//		System.out.println("bigram: "+bigram);
		for (String word3 : nextwords.keySet()) {
			sum += buTrigramCounterMap.getCount(bigram, word3) / bigramCounterMap.getCount(word1, word2);
			if (sum > sample) { return word3; }
		}
		System.out.println("bigram_fail: "+bigram+" : "+bigramCounterMap.getCount(word1, word2));
		System.out.println(buTrigramCounterMap.getCounter(bigram));
  	return "nope";
  }

  public List<String> generateSentence() {
    List<String> sentence = new ArrayList<String>();
    String word = generateFirstWord();
    sentence.add(word);
    Pair<String,String> bigram = Pair.makePair(START, word);
    String newword = generateNextWord(bigram);
    while (!newword.equals(STOP)) {
      sentence.add(newword);
      bigram = Pair.makePair(bigram.getSecond(), newword);
      newword = generateNextWord(bigram);
    }
    return sentence;
  }
  
  public InterpolatedTrigramLanguageModel(Collection<List<String>> trainingSet, Collection<List<String>> validSet) {
  	for (List<String> sentence : trainingSet) {
      List<String> stoppedStartedSentence = new ArrayList<String>(sentence);
      stoppedStartedSentence.add(STOP);
      stoppedStartedSentence.add(0, START);
      String first_word = stoppedStartedSentence.get(1);
      bigramCounterMap.incrementCount(START, first_word, 1.0);
      for (int i=0; i < stoppedStartedSentence.size()-2; i++) {
      	String token1 = stoppedStartedSentence.get(i);
      	String token2 = stoppedStartedSentence.get(i+1);
      	String token3 = stoppedStartedSentence.get(i+2);
      	Pair<String,String> bigram1 = Pair.makePair(token1, token2);
      	Pair<String,String> bigram2 = Pair.makePair(token2, token3);
        bigramCounterMap.incrementCount(token2, token3, 1.0);
        ubTrigramCounterMap.incrementCount(token1, bigram2, 1.0);
        buTrigramCounterMap.incrementCount(bigram1, token3, 1.0);
      }
    }
    
    for (List<String> sentence : validSet) {
    	List<String> stoppedStartedSentence = new ArrayList<String>(sentence);
      stoppedStartedSentence.add(STOP);
      stoppedStartedSentence.add(0, START);
      String first_word = stoppedStartedSentence.get(1);
      bigramCounterMap.incrementCount(START, first_word, 1.0);
      for (int i=0; i < stoppedStartedSentence.size()-2; i++) {
      	String token1 = stoppedStartedSentence.get(i);
      	String token2 = stoppedStartedSentence.get(i+1);
      	String token3 = stoppedStartedSentence.get(i+2);
      	Pair<String,String> bigram1 = Pair.makePair(token1, token2);
      	Pair<String,String> bigram2 = Pair.makePair(token2, token3);
        bigramCounterMap.incrementCount(token2, token3, 1.0);
        ubTrigramCounterMap.incrementCount(token1, bigram2, 1.0);
        buTrigramCounterMap.incrementCount(bigram1, token3, 1.0);
      }
    }
    wordCount = bigramCounterMap.totalCount() + sentenceCount;
    vocabSize = bigramCounterMap.size() + 1;
    bigramCount = bigramCounterMap.totalCount();
    bigramVocabSize = bigramCounterMap.totalSize();
    trigramCount = ubTrigramCounterMap.totalCount();
    trigramVocabSize = ubTrigramCounterMap.totalSize();
    sentenceCount = trainingSet.size() + validSet.size();
    System.out.println("Wordcount:  "+wordCount);
    System.out.println("Vocabsize:  "+vocabSize);
    System.out.println("Bigramcount:  "+bigramCount);
    System.out.println("BigramVocabsize:  "+bigramVocabSize);
//    System.out.println("Trigramcount:  "+trigramCount);
//    System.out.println("TrigramVocabsize:  "+trigramVocabSize);
//    System.out.println("Sentencecount:  "+sentenceCount);
//    System.out.println("Get START counter:");
//    System.out.println(bigramCounterMap.getCounter(START));
//    System.out.println("Get UNK counter:");
//    System.out.println(bigramCounterMap.getCounter("UNK"));
  }
}
