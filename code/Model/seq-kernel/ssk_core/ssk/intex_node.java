package ssk;

import libsvm.*;

import java.io.*;
import java.util.*;

/**
 * Class intex_node extends svm_node to allow for a custom representation of
 * training/testing instances.
 * 
 * @author Razvan Bunescu
 */
public class intex_node extends svm_node {

  static public FeatureDictionary m_fd;
  static {
    m_fd = new FeatureDictionary();
  }

  public InstanceExample m_value;


  public void read(String line, double[][] coef, int m, int index)
  {
    StringTokenizer st = new StringTokenizer(line," \t\n\r\f");
    // Read SV coefficients.
    for (int k = 0; k < m; k++)
      coef[k][index] = svm.atof(st.nextToken());

    // Put remaining tokens in 'text'.
    String text = "";
    while (st.hasMoreTokens())
      text += st.nextToken() + " ";

    // Create instance example from 'text'.
    m_value = new InstanceExample(text, m_fd);
  }


  public void write(DataOutputStream fp) throws IOException
  {
    m_value.write(fp);
  }
  
}
