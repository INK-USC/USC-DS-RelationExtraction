package ssk;

import libsvm.*;
import java.util.*;


import java.util.Vector;
import java.io.File;  
import java.io.InputStreamReader;  
import java.io.BufferedReader;  
import java.io.BufferedWriter;  
import java.io.FileInputStream;  
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;

/**
 * Generalized subsequence kernel implementation.
 * 
 * @author Razvan Bunescu
 */
public class SubsequenceKernel extends CustomKernel
{
  static final int DEFAULT_MAXLEN = 4;
  static final double DEFAULT_LAMBDA = 0.75;
  static final boolean DEFAULT_CACHE = true; 
  static final boolean DEFAULT_NORM = true; 
  
  // maximum length of common subsequences
  int m_maxlen;
  // gap penalty
  double m_lambda;
  // true if self kernels are cached
  boolean m_bCache;
  // true if kernels are normalized
  boolean m_bNorm;
  
  protected HashMap m_mapStoK;
  
  public SubsequenceKernel(int maxlen, double lambda,
			   boolean bCache, boolean bNorm)
  {
    m_maxlen = maxlen;
    m_lambda = lambda;
    
    m_bCache = bCache;
    m_bNorm = bNorm;
    
    m_mapStoK = new HashMap();
  }
  
  
  public SubsequenceKernel() 
  {
    // Default values.
    m_maxlen = DEFAULT_MAXLEN;
    m_lambda = DEFAULT_LAMBDA;
    
    m_bCache = DEFAULT_CACHE;
    m_bNorm = DEFAULT_NORM;
    
    m_mapStoK = new HashMap();
  }
  
  
  /**
   * Computes the (normalized) subsequence kernel between two sequences.
   *
   * @param ie1 sequence instance 1
   * @param ie2 sequence instance 2
   * @return kernel value.
   */
  public double kernel(InstanceExample ie1, InstanceExample ie2)
  {
    if (m_bNorm) {
      double k1 = selfKernel(ie1.sequence_);
      double k2 = selfKernel(ie2.sequence_);
      
      double k = singleKernel(ie1.sequence_, ie2.sequence_);
      if (k == 0)
	return 0;
      
      assert k1 != 0;
      assert k2 != 0;
      
      // normalize
      return k / Math.sqrt (k1 * k2);
    }
    
    // don't normalize
    return singleKernel(ie1.sequence_, ie2.sequence_);
  }
  
  
  /**
   * Kernel method, with prototype specified by CustomKernel.
   *
   * @param x1 first instances.
   * @param x2 second instance.
   * @return kernel value.
   */
  public double kernel(svm_node[] x1, svm_node[] x2)
  {
    InstanceExample ie1 = ((intex_node) x1[0]).m_value;
    InstanceExample ie2 = ((intex_node) x2[0]).m_value;

    return kernel(ie1, ie2);
  }
  
  
  public svm_node new_svm_node()
  {
    return new intex_node();
  }


  public double selfKernel(String[][] s)
  {
    if (m_bCache) {
      // get cached value
      Double dblk = (Double) m_mapStoK.get(s);
      if (dblk == null) {
	double k = singleKernel(s, s);
	m_mapStoK.put(s, new Double(k));
	return k;
      }
      return dblk.doubleValue();
    }

    return singleKernel(s, s);
  }
  

  public double singleKernel(String[][] s1, String[][] s2)
  {
    double[] sk = stringKernel(s1, s2, m_maxlen, m_lambda);
    double result = 0.0;
    for (int i = 0; i < sk.length; i++)
      result += sk[i];

    return result;
  }


  /**
   * Computes the number of common subsequences between two sequences.
   *
   * @param s first sequence of features.
   * @param t second sequence of features.
   * @param n maximum subsequence length.
   * @param lambda gap penalty.
   * @return kernel value K[], one position for every length up to n.
   *
   * The algorithm corresponds to the recursive computation from Figure 1
   * in the paper "Subsequence Kernels for Relation Extraction" (NIPS 2005),
   * where:
   * - K stands for K;
   * - Kp stands for K';
   * - Kpp stands for K'';
   * - common stands for c;
  */
  protected double[] stringKernel(String[][] s, String[][] t, 
				  int n, double lambda)
  {
    int sl = s.length;
    int tl = t.length;
    
    double[][][] Kp = new double[n + 1][sl][tl];
    
    for (int j = 0; j < sl; j++)
      for (int k = 0; k < tl; k++)
	Kp[0][j][k] = 1;
    
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < sl - 1; j++) {
	double Kpp = 0.0;
	for (int k = 0; k < tl - 1; k++) {
	  Kpp = lambda * (Kpp + lambda * common(s[j], t[k])  * Kp[i][j][k]);
	  Kp[i + 1][j + 1][k + 1] = lambda * Kp[i + 1][j][k + 1] + Kpp;
	}
      }
    }

    double[] K = new double[n];
    for (int l = 0; l < K.length; l++) {
      K[l] = 0.0;
      for (int j = 0; j < sl; j++) {
	for (int k = 0; k < tl; k++)
	  K[l] += lambda * lambda * common(s[j], t[k]) * Kp[l][j][k];
      }
    }

    return K;
  }


  /**
   * Computes the number of common features between two sets of featurses.
   *
   * @param s first set of features.
   * @param t second set of features.
   * @return number of common features.
   *
   * The use of FeatureDictionary ensures that identical features correspond
   * to the same object reference. Hence, the operator '==' can be used to
   * speed-up the computation.
  */
  protected int common(String[] s, String[] t)
  {
    assert s.length == t.length;
    int nCount = 0;
    for (int i = 0; i < s.length; i++)
      if (s[i] != null && s[i] == t[i])
	nCount++;

    return nCount;
  }


  public static void main (String[] args)
  {
    ArrayList<String> listb = new ArrayList<>();
    ArrayList<String> listi = new ArrayList<>();

    File filenameb = new File("base.txt");
    try
    {
        InputStreamReader reader = new InputStreamReader(new FileInputStream(filenameb));
    
        BufferedReader br = new BufferedReader(reader);
        String line = null;
    
        while ((line = br.readLine()) != null)
        {
            listb.add(line);
        }
    }
    catch(IOException e)
    {
    }
      
    File filenamei = new File("infer.txt");
    try
    {
        InputStreamReader reader = new InputStreamReader(new FileInputStream(filenamei));
          
        BufferedReader br = new BufferedReader(reader);
        String line = null;
          
        while ((line = br.readLine()) != null)
        {
            listi.add(line);
        }
    }
    catch(IOException e)
    {
    }
      
      System.out.println(listi.size());
      
      System.out.println(listb.size());

      
    File writename = new File("out.txt");
    try
    {
        writename.createNewFile();
        BufferedWriter out = new BufferedWriter(new FileWriter(writename));
        
        out.write(listi.size() + " " + listb.size() + "\n");
        
        String texti, textj;
        for (int i = 0; i != listi.size(); i++)
        {
            System.out.println(i);
            texti = (String)listi.get(i);
            for (int j = 0; j != listb.size(); j++)
            {
                textj = (String)listb.get(j);
                //System.out.println(text);
                
                FeatureDictionary fd = new FeatureDictionary();
                InstanceExample ie1 = new InstanceExample(texti, fd);
                InstanceExample ie2 = new InstanceExample(textj, fd);
                
                SubsequenceKernel rk = new SubsequenceKernel(2, 1.0, false, false);
                out.write(rk.kernel(ie1, ie2) + " ");
                out.flush();
            }
            
            out.write("\n");
            //System.out.println("\n");
        }
        out.close();
    }
    catch(IOException e)
    {
    }

    /*FeatureDictionary fd = new FeatureDictionary();

    String text1 = "Zori/NNP runs/VBZ after/IN every/DT rabbit/NN ./.";
    InstanceExample ie1 = new InstanceExample(text1, fd);
    String text2 = "Zori/NNP walks/VBZ the/DT dog/NN every/DT morning/NN ./.";
    InstanceExample ie2 = new InstanceExample(text2, fd);

    // Maximum subsequence length is 1, gap penalty is 1 (i.e. no penalty),
    // no cache, no normalization.
    // Should return the number of matching words or tags => 10.
    // Obs: Matchings at different positions are counted as different.
    // Examples: Zori, after, every, ., NNP, VBZ, DT, ...
    SubsequenceKernel rk = new SubsequenceKernel(1, 1.0, false, false);
    System.out.println(rk.kernel(ie1, ie2));

    // Maximum subsequence length is 2, gap penalty is 1 (i.e. no penalty),
    // no cache, no normalization.
    // Should return the number of matching subsequences of words and POS tags
    // of length up to 2 => 47.
    // Examples: all of the above, plus 'Zori ... every', 'Zori ... DT',
    //           'Zori ... DT', Zori ... NN', 'VBZ ... NN', ...
    rk = new SubsequenceKernel(2, 1.0, false, false);
    System.out.println(rk.kernel(ie1, ie2));


    // Just for sanity check, associate unique POS tag to each word,
    // so the kernel returns only number of common subsequences of words.
    text1 = "Zori/T1 runs/T2 after/T3 every/T4 rabbit/T5 ./T6";
    ie1 = new InstanceExample(text1, fd);
    text2 = "Zori/U1 walks/U2 the/U3 dog/U4 every/U5 morning/U6 ./U7";
    ie2 = new InstanceExample(text2, fd);

    // Returns number of common words => 3.
    // Examples: 'Zori', 'every', '.'.
    rk = new SubsequenceKernel(1, 1.0, false, false);
    System.out.println(rk.kernel(ie1, ie2));

    // Returns number of common subsequences of words of length up to 2 => 6.
    // Examples: all three above, plus 'Zori ... every', 'Zori ... .', 'after
    // ... .'
    rk = new SubsequenceKernel(2, 1.0, false, false);
    System.out.println(rk.kernel(ie1, ie2));*/
  }

}

