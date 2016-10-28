package edu.uw.cs.multir.main;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import edu.uw.cs.multir.learning.algorithm.FullInference;
import edu.uw.cs.multir.learning.algorithm.Model;
import edu.uw.cs.multir.learning.algorithm.Parameters;
import edu.uw.cs.multir.learning.algorithm.Parse;
import edu.uw.cs.multir.learning.algorithm.Scorer;
import edu.uw.cs.multir.learning.data.Dataset;
import edu.uw.cs.multir.learning.data.MILDocument;
import edu.uw.cs.multir.learning.data.MemoryDataset;

public class AggregatePrecisionRecallCurve {
	
	public static void run(String dir) 
		throws IOException {
		
		Model model = new Model();
		model.read(dir + File.separatorChar + "model");

		Parameters params = new Parameters();
		params.model = model;
		params.deserialize(dir + File.separatorChar + "params");
		
		Dataset test = new MemoryDataset(dir + File.separatorChar + "test");

		eval(test, params, System.out);
	}
	
	public static void eval(Dataset test, Parameters params,
			PrintStream ps) throws IOException {
		System.out.println("eval");
		Scorer scorer = new Scorer();
		
		// this could also be a file
		List<Prediction> predictions = new ArrayList<Prediction>();
		MILDocument doc = new MILDocument();
		int numRelationInst = 0;
		test.reset();
		while (test.next(doc)) {
			//numRelationInst += doc.Y.length;
			Parse parse = FullInference.infer(doc, scorer, params);
			int[] Yt = doc.Y;
			int[] Yp = parse.Y;
			
			// NA is empty array
			if (Yt.length == 0 && Yp.length == 0) continue;
				// true negative, we ignore that

			boolean[] binaryYt = new boolean[100];
			boolean[] binaryYp = new boolean[100];
			for (int i=0; i < Yt.length; i++)
				binaryYt[Yt[i]] = true;
			for (int i=0; i < Yp.length; i++)
				binaryYp[Yp[i]] = true;
			
			for (int i=1; i < binaryYt.length; i++) {				
				if (binaryYt[i] || binaryYp[i]) {
					predictions.add
						(new Prediction(i, binaryYt[i], binaryYp[i], parse.scores[i], doc, parse));
				}
			}
			
			for (int i=1; i < binaryYt.length; i++)
				if (binaryYt[i]) numRelationInst++;
		}
		
		Collections.sort(predictions, new Comparator<Prediction>() {
			public int compare(Prediction p1, Prediction p2) {
				if (p1.score > p2.score) return -1;
				else return +1;
			} });
		
		PrecisionRecallTester prt = new PrecisionRecallTester();
		prt.reset();
		double prevRec = -1, prevPre = -1;
		for (int i=0; i < predictions.size(); i++) {
			Prediction p = predictions.get(i);
			prt.handle(p.rel, p.predRel, p.trueRel, p.score);
			prt.numRelations = numRelationInst;
			double recall = prt.recall();
			double precision = prt.precision();
			if (recall != prevRec || precision != prevPre) {
				ps.println(recall + "\t" + precision);
				prevRec = recall;
				prevPre = precision;
			}
		}
	}
	
	static class Prediction {
		int rel;
		boolean trueRel;
		boolean predRel;
		double score;
		MILDocument doc;
		Parse parse;
		Prediction(int rel, boolean trueRel, boolean predRel, double score, 
				MILDocument doc, Parse parse) {
			this.rel = rel;
			this.trueRel = trueRel;
			this.predRel = predRel;
			this.score = score;
			this.doc = doc;
			this.parse = parse;
		}
	}

	
	static class PrecisionRecallTester {
		public double numCorrect, numPredictions, numRelations;
		
		public void handle(String[] tokens, boolean[] predictedLabels, 
				boolean[] trueLabels, double score) {
			boolean[] p = predictedLabels;
			boolean[] t = trueLabels;
			
			for (int i=1; i < p.length; i++) {
				if (p[i] && !t[i]) numPredictions++;
				else if (!p[i] && t[i]) numRelations++;
				else if (p[i] && t[i]) {
					numCorrect++;
					numPredictions++;
					numRelations++;
				}
			}
		}
		
		public void handle(int rel, boolean p, boolean t, double score) {
			if (p && !t) numPredictions++;
			else if (!p && t) numRelations++;
			else if (p && t) {
				numCorrect++;
				numPredictions++;
				numRelations++;
			}
		}
		
		public void reset() { numCorrect = numPredictions = numRelations = 0; }
		public double precision() {
			if (numPredictions == 0) return 1;
			return numCorrect / numPredictions; 
		}
		public double recall() { 
			return numCorrect / numRelations; 
		}
	}
}
