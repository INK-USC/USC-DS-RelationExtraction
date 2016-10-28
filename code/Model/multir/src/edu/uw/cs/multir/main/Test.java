package edu.uw.cs.multir.main;

import java.io.File;
import java.io.IOException;

import edu.uw.cs.multir.learning.algorithm.FullInference;
import edu.uw.cs.multir.learning.algorithm.Model;
import edu.uw.cs.multir.learning.algorithm.Parameters;
import edu.uw.cs.multir.learning.algorithm.Scorer;
import edu.uw.cs.multir.learning.data.Dataset;
import edu.uw.cs.multir.learning.data.MILDocument;
import edu.uw.cs.multir.learning.data.MemoryDataset;

public class Test {

	public static void test(String dir) throws IOException {

		Model model = new Model();
		model.read(dir + File.separatorChar + "model");

		Parameters params = new Parameters();
		params.model = model;
		params.deserialize(dir + File.separatorChar + "params");
		
		Dataset test = new MemoryDataset(dir + File.separatorChar + "test");
		
		long startTest = System.currentTimeMillis();
		MILDocument doc = new MILDocument();
		Scorer scorer = new Scorer();
		test.reset();
		while (test.next(doc)) {
			FullInference.infer(doc, scorer, params);
		}
		long endTest = System.currentTimeMillis();
		System.out.println("testing time " + (endTest-startTest)/1000.0 + " seconds");
	}
}
