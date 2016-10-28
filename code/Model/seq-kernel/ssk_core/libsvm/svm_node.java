package libsvm;
import java.io.*;

public class svm_node implements java.io.Serializable
{
  public int index;
  public double value;

  public void read(String line, double[][] coef, int m, int index)
  {
  }

  public void write(DataOutputStream fp) throws IOException
  {
    fp.writeBytes(index + ":" + value + " ");
  }
  
}
