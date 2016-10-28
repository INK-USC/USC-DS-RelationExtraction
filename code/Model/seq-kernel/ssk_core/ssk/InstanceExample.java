package ssk;

import java.io.*;
import java.util.*;

/**
 * Class InstanceExample represents a training/testing example as an array
 * of feature sets (here, a feature set contains a word and its POS tag).
 * 
 * @author Razvan Bunescu
 */
public class InstanceExample implements java.io.Serializable {

  int label_;
  String[][] sequence_;


  public InstanceExample(String text, FeatureDictionary fd)
  {
    Vector<String[]> sequence = new Vector<String[]>();

    StringTokenizer st = new StringTokenizer(text);
    while (st.hasMoreTokens()) {
      String word_tag = st.nextToken();

      // Features are separated by '/'.
      int separator = word_tag.lastIndexOf('/');
      assert separator != -1;

      String word = word_tag.substring(0, separator);
      String tag = word_tag.substring(separator + 1);

      // Use object with the same value from the dictionary.
      // If no such object, add new feature to the dictionary.
      String[] features = new String[FeatureDictionary.FEAT_TYPES];
      features[FeatureDictionary.FEAT_WORD] = 
	fd.getAddFeature(FeatureDictionary.FEAT_WORD, word);
      features[FeatureDictionary.FEAT_POS] = 
	fd.getAddFeature(FeatureDictionary.FEAT_POS, tag);

      sequence.add(features);
    }

    sequence_ = sequence.toArray(new String[0][]);
  }


  public void setLabel(int label)
  {
    label_ = label;
  }

  public int getLabel()
  {
    return label_;
  }


  public String toString()
  {
    String result = "";
    for (int i = 0; i < sequence_.length; i++) {
      result += 
	sequence_[i][FeatureDictionary.FEAT_WORD] +
	"/" +
	sequence_[i][FeatureDictionary.FEAT_POS] +
	" ";
    }

    return result;
  }


  public void write(DataOutputStream fp) throws IOException
  {
    fp.writeBytes(toString());
    fp.writeBytes("\n");
  }
  
}
