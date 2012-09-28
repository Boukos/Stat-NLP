package nlp.assignments;


import java.util.List;

import nlp.classify.*;
import nlp.util.Counter;
import nlp.util.CounterMap;

class CharacterUnigramClassifier<I,F,L> implements ProbabilisticClassifier<I,L> {
	
  public static class Factory<I,F,L> implements ProbabilisticClassifierFactory<I,L> {
  	
  	CounterMap<L,F> labelFeatureMap = new CounterMap<L,F>();
  	Counter<L> labelCounter = new Counter<L>();
    FeatureExtractor<I,F> featureExtractor;

    public ProbabilisticClassifier<I,L> trainClassifier(List<LabeledInstance<I,L>> trainingData) {
      for (LabeledInstance<I,L> datum : trainingData) {
        L label = datum.getLabel();
        labelCounter.incrementCount(label, 1.0);
        Counter<F> features = featureExtractor.extractFeatures(datum.getInput());
        labelFeatureMap.incrementByCounter(label, features);
      }
      return new CharacterUnigramClassifier<I,F,L>(labelFeatureMap, featureExtractor, labelCounter);
    }
    public Factory(FeatureExtractor<I,F> featureExtractor) {
      this.featureExtractor = featureExtractor;
    }
  }
  
	CounterMap<L,F> labelFeatureMap;
	Counter<L> labelCounter;
  private FeatureExtractor<I,F> featureExtractor;
  
  public double labelCharacterCount(L label) {
  	Counter<F> features = labelFeatureMap.getCounter(label);
  	return features.totalCount();
  }
  
  // ML class probabilities, no smoothing
  public double labelProb(L label) {
  	return labelCounter.getCount(label) / labelCounter.totalCount();
  }
  
  // ML P(X_i = x | Y = y_j), Laplace smoothing
  public double featureGivenLabelProb(L label, F feature) {
  	return (labelFeatureMap.getCount(label, feature) + 1) / (labelCharacterCount(label) + 5);
  }
  
  public Counter<L> getProbabilities(I input) {
    Counter<L> counter = new Counter<L>();
    Counter<F> inputFeatures = featureExtractor.extractFeatures(input);
    double denominator = 0.0;
    for (L denom_label : labelFeatureMap.keySet()) {
    	double denom_prior = labelProb(denom_label);
    	double denom_prob = 1.0;
    	for (F feature : inputFeatures.keySet()) {
    		double prob = Math.pow(featureGivenLabelProb(denom_label, feature), inputFeatures.getCount(feature));
    		denom_prob *= prob;
    	}
    	denominator += denom_prior * denom_prob;
    }
    for (L label : labelFeatureMap.keySet()) {
    	double label_prior = labelProb(label);
      double product_term = 1.0;
      for (F feature : inputFeatures.keySet()) {
      	double prob = Math.pow(featureGivenLabelProb(label, feature), inputFeatures.getCount(feature));
      	product_term *= prob;
      }
      double label_given_features_prob = label_prior * product_term / denominator;
      counter.incrementCount(label, label_given_features_prob);
    }
    return counter;
  }

  public L getLabel(I input) {
    return getProbabilities(input).argMax();
  }

//  public CharacterUnigramClassifier(CounterMap<L,F> labelFeatureMap, FeatureExtractor<I,F> featureExtractor) {
//    this.labelFeatureMap = labelFeatureMap;
//    this.featureExtractor = featureExtractor;
//  }
  
  public CharacterUnigramClassifier(CounterMap<L,F> labelFeatureMap, FeatureExtractor<I,F> featureExtractor, Counter<L> labelCounter) {
    this.labelFeatureMap = labelFeatureMap;
    this.featureExtractor = featureExtractor;
    this.labelCounter = labelCounter;
  }
	
}