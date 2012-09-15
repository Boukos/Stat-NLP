package nlp.assignments;


import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.Pair;

/**
 * A dummy language model -- uses empirical Bigram counts, plus a single
 * ficticious count for unknown words.
 */
class UnsmoothedBigramLanguageModel implements LanguageModel {

  static final String STOP = "</S>";
  static final String START = "<S>";
  
  double lambda = 0.6;

  double wordCount = 0.0;
  double vocabSize = 0.0;
  
  Counter<String> unigramCounter = new Counter<String>();
  Counter<Pair<String,String>> bigramCounter = new Counter<Pair<String,String>>();
  
  public double unigramProb(String word) {
    double count = unigramCounter.getCount(word);
    if (count > 0) { return count / wordCount; }
    else {
//    	System.out.println("UNKNOWN WORD: "+word);
//    	System.out.println("Probability:  "+unigramProb("UNK"));
    	return unigramProb("UNK");
    }
  }
  
  public double getUnigramProbability(List<String> sentence, int index) {
    String word = sentence.get(index);
    return unigramProb(word);
  }
  
  public double condBigramProb(String word1, String word2) {
  	double word1count = unigramCounter.getCount(word1);
  	Pair<String,String> bigram = Pair.makePair(word1, word2);
  	double bicount = bigramCounter.getCount(bigram);
  	
  	if (word1count == 0) { return (1 - lambda) * unigramProb(word1); }
  	else { return lambda * (bicount / word1count) + (1 - lambda) * unigramProb(word1); }
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
    double sample = Math.random();
    double sum = 0.0;
    for (String word2 : unigramCounter.keySet()) {
    	Pair<String,String> bigram = Pair.makePair(word1, word2);
      sum += bigramCounter.getCount(bigram) / unigramCounter.getCount(word1);
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
  
  public UnsmoothedBigramLanguageModel(Collection<List<String>> sentenceCollection) {
    for (List<String> sentence : sentenceCollection) {
      List<String> stoppedStartedSentence = new ArrayList<String>(sentence);
      stoppedStartedSentence.add(STOP);
      stoppedStartedSentence.add(0, START);
      for (int i=0; i < stoppedStartedSentence.size()-1; i++) {
      	String token1 = stoppedStartedSentence.get(i);
      	String token2 = stoppedStartedSentence.get(i+1);
      	Pair<String,String> bigram = Pair.makePair(token1, token2);
        unigramCounter.incrementCount(token1, 1.0);
        bigramCounter.incrementCount(bigram, 1.0);
      }
      String last_word = stoppedStartedSentence.get(stoppedStartedSentence.size()-1);
      unigramCounter.incrementCount(last_word, 1.0);
    }
    unigramCounter.incrementCount("UNK", 1.0);
    wordCount = unigramCounter.totalCount();
//    System.out.println("Wordcount:  "+wordCount);
    vocabSize = unigramCounter.size();
//    System.out.println("Vocabsize:  "+vocabSize);
  }
}
