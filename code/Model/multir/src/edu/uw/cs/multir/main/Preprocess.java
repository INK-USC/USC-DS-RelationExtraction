package edu.uw.cs.multir.main;

import java.io.File;
import java.io.IOException;

import edu.uw.cs.multir.learning.algorithm.Model;
import edu.uw.cs.multir.preprocess.ConvertProtobufToMILDocument;
import edu.uw.cs.multir.preprocess.Mappings;

public class Preprocess {

	public static void preprocess(String trainFile, String testFile, String outDir) 
	throws IOException {
			
		String mappingFile = outDir + File.separatorChar + "mapping";
		String modelFile = outDir + File.separatorChar + "model";
		
		{
			String output1 = outDir + File.separatorChar + "train";
			ConvertProtobufToMILDocument.convert(trainFile, output1, mappingFile, true, true);
		}
		
		{
			String output2 = outDir + File.separatorChar + "test";
			ConvertProtobufToMILDocument.convert(testFile, output2, mappingFile, false, false);
		}
		
		{
			Model m = new Model();
			Mappings mappings = new Mappings();
			mappings.read(mappingFile);
			m.numRelations = mappings.numRelations();
			m.numFeaturesPerRelation = new int[m.numRelations];
			for (int i=0; i < m.numRelations; i++)
				m.numFeaturesPerRelation[i] = mappings.numFeatures();
			m.write(modelFile);
		}		
	}
}
