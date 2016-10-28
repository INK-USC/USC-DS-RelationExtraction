package ssk;

import java.util.*;

/**
 * Class FeatureDictionary. A dictionary (i.e set of unique features) is 
 * created for each feature type.
 *
 * Current feature types:
 * - words;
 * - POS tags;
 *
 * Other possible feature types:
 * - phrase tags;
 * - entity types;
 * - WordNet synsets;
 *
 * @author Razvan Bunescu
 */
public class FeatureDictionary {

  static public int FEAT_TYPES = 2;

  static public int FEAT_WORD = 0;
  static public int FEAT_POS = 1;

  public HashMap<String, String>[] m_features;

  public FeatureDictionary()
  {
    m_features = (HashMap<String, String>[]) new HashMap[FEAT_TYPES];

    for (int i = 0; i < m_features.length; i++)
      m_features[i] = new HashMap<String, String>();
  }


  public String getAddFeature(int nType, String strFeature)
  {
    String strUnique = m_features[nType].get(strFeature);
    if (strUnique == null) {
      m_features[nType].put(strFeature, strFeature);
      strUnique = strFeature;
    }

    return strUnique;
  }

}
