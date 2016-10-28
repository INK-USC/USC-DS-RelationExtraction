package edu.uw.cs.multir.main;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;

import edu.uw.cs.multir.learning.algorithm.FullInference;
import edu.uw.cs.multir.learning.algorithm.Model;
import edu.uw.cs.multir.learning.algorithm.Parameters;
import edu.uw.cs.multir.learning.algorithm.Parse;
import edu.uw.cs.multir.learning.algorithm.Scorer;
import edu.uw.cs.multir.learning.data.Dataset;
import edu.uw.cs.multir.learning.data.MILDocument;
import edu.uw.cs.multir.learning.data.MemoryDataset;
import edu.uw.cs.multir.preprocess.Mappings;

public class ResultWriter {

	// allow only a single output label per entity pair, instead of multiple
	static boolean singleBestOnly = false;
	
	public static void write(String dir) throws IOException {

		Model model = new Model();
		model.read(dir + File.separatorChar + "model");

		Parameters params = new Parameters();
		params.model = model;
		params.deserialize(dir + File.separatorChar + "params");
		
		Dataset test = new MemoryDataset(dir + File.separatorChar + "test");
		
		PrintStream ps = new PrintStream(dir + File.separatorChar + "results");
		ResultWriter.eval(dir + File.separatorChar + "mapping", test, params, ps);
		ps.close();
	}
	
	public static void eval(String mappingFile, Dataset test, Parameters params,
			PrintStream ps) throws IOException {
		
		// need mapping from relIDs to rels
		Mappings mapping = new Mappings();
		mapping.read(mappingFile);
		Map<Integer,String> relID2rel = new HashMap<Integer,String>();
		for (Map.Entry<String,Integer> e : mapping.getRel2RelID().entrySet())
			relID2rel.put(e.getValue(), e.getKey());
		
		System.out.println("eval");
		Scorer scorer = new Scorer();

		StringBuilder sb1 = new StringBuilder();
		for (int i=0; i < mapping.numRelations(); i++)
			sb1.append(relID2rel.get(i) + " ");
		ps.append(sb1.toString() + "\n");
		
		MILDocument doc = new MILDocument();
		test.reset();		
		while (test.next(doc)) {
			Parse parse = FullInference.infer(doc, scorer, params);
			int[] Yp = parse.Y;
			
			if (Yp.length > 1 && singleBestOnly) {
				int max = 0;
				for (int i=1; i < Yp.length; i++)
					if (parse.scores[Yp[i]] > parse.scores[Yp[max]]) max = i;
				Yp = new int[] { Yp[max] };
				
				// set sentence-level predictions
				for (int m = 0; m < doc.numMentions; m++) {
					if (parse.Z[m] != 0 && parse.Z[m] != max) {
						if (parse.allScores[m][0] > parse.allScores[m][max]) parse.Z[m] = 0;
						else parse.Z[m] = max;
					}
				}
			}

			for (int m = 0; m < doc.numMentions; m++) {
				StringBuilder sb2 = new StringBuilder();
				ps.append(doc.arg1 + "\t" + doc.arg2 + "\t" + m + "\t" + 
						relID2rel.get(parse.Z[m]) + "\t" + parse.score + "\t" + sb2.toString() + "\n");
			}
		}
	}
}
