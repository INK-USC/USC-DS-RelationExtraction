[Razvan Bunescu - August, 2007]

The 'ssk_core' directory contains the following two Java packages:
------------------------------------------------------------------

[ssk]
-----

   This is an implementation of the generalized subsequence kernel described
in the paper "Subsequence Kernels for Relation Extracion" (Bunescu & Mooney,
NIPS 2005). The code was written to be compatible with the libsvm package 
(see section on [libsvm] below).
   The source is self explanatory. For testing, I have written a main function 
inside the class 'SubsequenceKernel'.

To compile (make sure libsvm is already compiled, see [libsvm] below):
   // Descend into the 'ssk' folder.
   > cd ssk
   // Build the 'ssk.jar' package.
   > make
   // Go back into the 'ssk_core' folder.
   > cd ..

To test:
   // Run test examples from 'SubsequenceKernel'.
   > java ssk.SubsequenceKernel


[libsvm]
--------

   This is Chih-Jen Lin's Java package for SVM learning, modified to accept 
custom kernels. I have added a new abstract class called 'CustomKernel', and 
also modified 'svm', svm_node', and 'svm_parameter'.

To compile:
   // Descend into the 'libsvm' folder.
   > cd libsvm
   // Build the 'libsvm.jar' package.
   > make
   // Go back into the 'ssk_core' folder.
   > cd ..

------------------------------------------------------------------
