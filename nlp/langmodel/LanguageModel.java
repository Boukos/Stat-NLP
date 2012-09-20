package nlp.langmodel;

import java.util.List;


/**
 * Language models assign probabilities to sentences and generate sentences.
 */
public interface LanguageModel {
	double getSentenceProbability(List<String> sentence);
  List<String> generateSentence();
//  double p_interp(String word1, String word2, String word3);
//  double p_interp(String word1, String word2);
//  double unigramProb(String word1);
}
