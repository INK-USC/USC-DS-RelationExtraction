This distribution contains the source code for the experiments presented in
the following research publication:

    Raphael Hoffmann, Congle Zhang, Xiao Ling, Luke Zettlemoyer and 
    Daniel S. Weld (2011). "Knowledge-Based Weak Supervision for Information 
    Extraction of Overlapping Relations", in Proceedings of the Annual Meeting 
    of the Association for Computational Linguistics (ACL), 2011.

It includes algorithms for learning and inference, taking as input data files
in the format used in the following publication:

    Sebastian Riedel, Limin Yao and Andrew McCallum (2010). "Modeling Relations 
    and Their Mentions without Labeled Text", in Proceedings of the European
    Conference on Machine Learning and Knowledge Discovery in Databases
    (ECML PKDD), 2010.



To run the experiments in the ACL-11 paper, you can proceed as follows:

1. Convert train and test data into input format accepted by multiR:
   
   java -cp ".:../lib/protobuf-java-2.3.0.jar" 
      edu.uw.cs.multir.main.Main preprocess 
      -trainFile train.pb.gz 
      -testFile test.pb.gz 
      -outDir .
      
2. Train

   java -cp "." edu.uw.cs.multir.main.Main train 
      -dir .
      
3. Generate file with results on test set
       
   java -cp ".:../lib/protobuf-java-2.3.0.jar" 
      edu.uw.cs.multir.main.Main results 
      -dir .
      
4. Generate sentence-level precision/recall curve

   java -cp "." 
      edu.uw.cs.multir.main.Main senPR 
      -labelsFile ../annotations/sentential.txt 
      -resultsFile ./results
      
5. Generate sentence-level precision/recall by relation
      
   java -cp "." 
      edu.uw.cs.multir.main.Main senPR 
      -labelsFile ../annotations/sentential-byrelation.txt 
      -resultsFile ./results      