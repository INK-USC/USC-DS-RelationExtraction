package libsvm;

public abstract class CustomKernel
{
  abstract public double kernel(svm_node[] x, svm_node[] y);
  abstract public svm_node new_svm_node();
}
