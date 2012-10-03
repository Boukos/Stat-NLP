package nlp.assignments;

import com.aliasi.classify.ConfusionMatrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import nlp.classify.FeatureExtractor;
import nlp.classify.LabeledInstance;
import nlp.classify.ProbabilisticClassifier;
import nlp.classify.ProbabilisticClassifierFactory;
import nlp.util.CommandLineUtils;
import nlp.util.Counter;
import nlp.util.CounterMap;

/**
 * This is the main harness for assignment 2.  To run this harness, use
 * <p/>
 * java edu.berkeley.nlp.assignments.ProperNameTester -path ASSIGNMENT_DATA_PATH -model MODEL_DESCRIPTOR_STRING
 * <p/>
 * First verify that the data can be read on your system using the baseline model.  Second, find the point in the main
 * method (near the bottom) where a MostFrequentLabelClassifier is constructed.  You will be writing new
 * implementations of the ProbabilisticClassifer interface and constructing them there.
 */
public class ProperNameTester {

  public static class ProperNameFeatureExtractor implements FeatureExtractor<String,String> {

    /**
     * This method takes the list of characters representing the proper name
     * description, and produces a list of features which represent that
     * description.  The basic implementation is that only character-unigram
     * features are extracted.  An easy extension would be to also throw character
     * bigrams into the feature list, but better features are also possible.
     */
  	
    public Counter<String> extractFeatures(String name) {
      char[] characters = name.toCharArray();
      Counter<String> features = new Counter<String>();
      
      // add character unigram/bigram features
      for (int i = 0; i < characters.length; i++) {
        char character = characters[i];
        features.incrementCount("UNI-" + character, 1.0);
        if (i < characters.length - 1) {
        	char character2 = characters[i+1];
      		features.incrementCount("BI-" + character + character2, 1.0);
        	if (i < characters.length - 2) {
          	char character3 = characters[i+2];
          	features.incrementCount("TRI-" + character + character2 + character3, 1.0);
          	if (i < characters.length - 3) {
          		char character4 = characters[i+3];
          		features.incrementCount("QUAD-" + character + character2 + character3 + character4, 1.0);
//        		  if (i < characters.length - 4) {
//        		  	char character5 = characters[i+4];
//        		  	features.incrementCount("QUINT-" + character + character2 + character3 + character4 + character5, 1.0);
//          		}
          	}
        	}
        }
      }
      
      // approximate syllable and wordlength averages, which are apparently useless
//      int wordcount = 1;
//      int sum = 0;
//      int sylcount = 0;
//      String[] protowords = name.split(" ");
//      String word_regex = "[^\\p{L}]";
//      Pattern word_pattern = Pattern.compile(word_regex);
//      for (String word : protowords) {
//        if (!word_pattern.matcher(word).find()) { 
//        	wordcount++;
//        	sum += word.length();
//        }
//      }
//      String syll_regex = "(?i)\\w*?[aeiouy]+(?=([a-z&&[^aeiou]]+[aeiouy]+))\\w*?";
//      Pattern syll_pattern = Pattern.compile(syll_regex);
//      Matcher syll_matcher = syll_pattern.matcher(name);
//      while (syll_matcher.find()) {
//      	sylcount++;
//      }
//      double wordavg = sum / wordcount;
//      double syllavg = (sylcount + wordcount) / wordcount;
//      features.incrementCount("AvgSylCount", syllavg);
//      features.incrementCount("AvgWordLength", wordavg);
      
      // extract name-like patterns
      String name_regex = "(?u)^(\\p{Lu}[\\p{L}&&[^\\p{Lu}]]+ |(\\p{Lu}\\.)+ )((\\p{Lu}\\.)+ )*\\p{Lu}[\\p{L}&&[^\\p{Lu}]]+$";
      Pattern name_pattern = Pattern.compile(name_regex);
      Matcher name_matcher = name_pattern.matcher(name);
      String name_filter = "^The | ((?i)Inc|Co|Corp|Corporation|Company|Ltd|Limited|Trust|Plc)$";
      Pattern filter_pattern = Pattern.compile(name_filter);
      if (name_matcher.find()) {
      	String namelike = name_matcher.group(0);
      	if (!filter_pattern.matcher(namelike).find()) {
//      		System.out.println(namelike);
          features.incrementCount("Namelike", 1.0);
      	}
      }
      
      // extract "Inc."-style tags
      String corp_regex = "(?i).*\\b(?: Inc|Co|Corp|Corporation|Company|Ltd|Limited|Trust|Plc|S[\\. ]?A)\\.?$";
      Pattern corp_pattern = Pattern.compile(corp_regex);
      Matcher corp_matcher = corp_pattern.matcher(name);
      if (corp_matcher.matches()) { features.incrementCount("Inc", 1.0); }
      
      // extract chemical-sounding endings
      String chem_regex = ".*\\w(?: a[cs]e|tone|al|yl|aid|gens?|zyme)$";
      Pattern chem_pattern = Pattern.compile(chem_regex);
      Matcher chem_matcher = chem_pattern.matcher(name);
      if (chem_matcher.matches()) { features.incrementCount("ChemEnding", 1.0); }
      
      // extract in-name-dict
//      List<String> words = new ArrayList<String>();
//	    String[] protowords = name.split(" ");
//	    String word_regex = "[^\\p{L}]";
//	    Pattern word_pattern = Pattern.compile(word_regex);
//	    for (String word : protowords) {
//	      if (!word_pattern.matcher(word).find()) { words.add(word); }
//	    }
//	    for (String word : words) {
//      	if (name_dict.contains(word)) {
//      		features.incrementCount("containsName", 1.0);
//      		break;
//      	}
//      }
      	
      return features;
    }
  }

  private static List<String> loadNames() throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader("/Users/bumford/Desktop/Given-Names.txt"));
    List<String> given_names = new ArrayList<String>();
    while (reader.ready()) {
      String given_name = reader.readLine().trim();
      given_names.add(given_name);
    }
    return given_names;
  }

  private static List<LabeledInstance<String, String>> loadData(String fileName) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader(fileName));
    List<LabeledInstance<String, String>> labeledInstances = new ArrayList<LabeledInstance<String, String>>();
    while (reader.ready()) {
      String line = reader.readLine();
      String[] parts = line.split("\t");
      String label = parts[0];
      String name = parts[1];
      LabeledInstance<String, String> labeledInstance = new LabeledInstance<String, String>(label, name);
      labeledInstances.add(labeledInstance);
    }
    return labeledInstances;
  }
  
  public static void printCM(ConfusionMatrix cm, String[] cats) {
  	System.out.println("\nConfusion Matrix:");
  	System.out.print("\t");
    for (String cat : cats) {
    	System.out.printf("%8s", cat);
    }
    System.out.print("\n");
    int[][] matrix = cm.matrix();
    for (int i=0; i < cats.length; i++) {
    	System.out.printf("%8s ", cats[i]);
    	for (int j=0; j < cats.length; j++) {
        System.out.printf("%-8d ", matrix[i][j]);
    	}
    	System.out.print("\n");
    }
  }
  
  public static String repeatString(String s, int n) { 
    StringBuilder sb = new StringBuilder(s.length() * n); 
    for (int i = 0; i < n; i++) 
       sb.append(s); 
    return sb.toString(); 
  }
  
  public static void printConfHist(CounterMap<Double,String> cb) {
  	Double[] bins = cb.keySet().toArray(new Double[cb.keySet().size()]);
  	Arrays.sort(bins);
  	System.out.println("\nConfidence Accuracy Histogram:");
  	for (Double bin : bins) {
  		double binAccuracy = cb.getCount(bin, "Correct") / cb.getCount(bin, "Total");
  		System.out.printf("%-3.0f %s (%.2f)\n", bin*10, repeatString("*", (int) Math.ceil(10*binAccuracy)), binAccuracy);
  	}
  	System.out.println("\n");
  }

  private static void testClassifier(ProbabilisticClassifier<String, String> classifier, List<LabeledInstance<String, String>> testData, boolean verbose) {
  	CounterMap<String,String> confusionmap = new CounterMap<String,String>();
  	List<String> categories = new ArrayList<String>();
  	CounterMap<Double,String> confidenceBin = new CounterMap<Double,String>();
    for (LabeledInstance<String, String> testDatum : testData) {
      String name = testDatum.getInput();
      String gold_label = testDatum.getLabel();
      String response_label = classifier.getLabel(name);
      double confidence = classifier.getProbabilities(name).getCount(response_label);
      if (response_label.equals(gold_label)) {
        confidenceBin.incrementCount(Math.ceil(confidence * 10), "Correct", 1.0);
      } else {
        if (verbose) {
          // display an error
          System.err.println("Error: "+name+" guess="+response_label+" gold="+gold_label+" confidence="+confidence);
        }
      }
      confidenceBin.incrementCount(Math.ceil(confidence * 10), "Total", 1.0);
      confusionmap.incrementCount(gold_label, response_label, 1.0);
    }
    
    // build confusion matrix    
    for (String category : confusionmap.keySet()) {
    	categories.add(category);
    }
    String[] cats = categories.toArray(new String[categories.size()]);
    ConfusionMatrix cm = new ConfusionMatrix(cats);
    for (int i=0; i < cats.length; i++) {
    	for (int j=0; j < cats.length; j++) {
      	cm.incrementByN(i,j, (int)confusionmap.getCount(cats[i], cats[j]));
    	}
    }
    System.out.println("Accuracy: " + cm.totalAccuracy());
    printCM(cm, cats);
    printConfHist(confidenceBin);
  }

  public static void main(String[] args) throws IOException {
    // Parse command line flags and arguments
    Map<String, String> argMap = CommandLineUtils.simpleCommandLineParser(args);

    // Set up default parameters and settings
    String basePath = ".";
    String model = "baseline";
    boolean verbose = false;
    boolean useValidation = true;

    // Update defaults using command line specifications

    // The path to the assignment data
    if (argMap.containsKey("-path")) {
      basePath = argMap.get("-path");
    }
    System.out.println("Using base path: " + basePath);

    // A string descriptor of the model to use
    if (argMap.containsKey("-model")) {
      model = argMap.get("-model");
    }
    System.out.println("Using model: " + model);

    // A string descriptor of the model to use
    if (argMap.containsKey("-test")) {
      String testString = argMap.get("-test");
      if (testString.equalsIgnoreCase("test"))
        useValidation = false;
    }
    System.out.println("Testing on: " + (useValidation ? "validation" : "test"));

    // Whether or not to print the individual speech errors.
    if (argMap.containsKey("-verbose")) {
      verbose = true;
    }

    // Load training, validation, and test data
    List<LabeledInstance<String, String>> trainingData = loadData(basePath + "/pnp-train.txt");
    List<LabeledInstance<String, String>> validationData = loadData(basePath + "/pnp-validate.txt");
    List<LabeledInstance<String, String>> testData = loadData(basePath + "/pnp-test.txt");

    // Learn a classifier
    ProbabilisticClassifier<String, String> classifier = null;
    if (model.equalsIgnoreCase("baseline")) {
      classifier = new MostFrequentLabelClassifier.Factory<String, String>().trainClassifier(trainingData);
    } else if (model.equalsIgnoreCase("n-gram")) {
      ProbabilisticClassifierFactory<String,String> factory = new CharacterUnigramClassifier.Factory<String,String,String>(new ProperNameFeatureExtractor());
      classifier = factory.trainClassifier(trainingData);
    } else if (model.equalsIgnoreCase("maxent")) {
      ProbabilisticClassifierFactory<String,String> factory = new MaximumEntropyClassifier.Factory<String,String,String>(1.0, 40, new ProperNameFeatureExtractor());
      classifier = factory.trainClassifier(trainingData);
    } else if (model.equalsIgnoreCase("perceptron")) {
    	ProbabilisticClassifierFactory<String,String> factory = new PerceptronClassifier.Factory<String,String,String>(new ProperNameFeatureExtractor());
    	classifier = factory.trainClassifier(trainingData);
    } else {
      throw new RuntimeException("Unknown model descriptor: " + model);
    }

    // Test classifier
    testClassifier(classifier, (useValidation ? validationData : testData), verbose);
  }
}
