package edu.uw.cs.multir.main;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SententialPrecisionRecallCurve {

	static boolean outputTopSentences = false;

	// treat "indirect" labels as "y" labels
	static boolean indirect = true;
	
	public static void run(String labelsFile, String resultsFile) throws IOException {
		
		// put results into map
		// guid1, guid2, mtnID -> ex
		List<Example> predictions = new ArrayList<Example>();
		{
			BufferedReader r = new BufferedReader(new InputStreamReader
					(new FileInputStream(resultsFile), "utf-8"));
			String l = null;
			while ((l = r.readLine())!= null) {
				String[] c = l.split("\t");
				if (c.length < 2) continue; // column header

				Example e = new Example();
				e.arg1 = c[0];
				e.arg2 = c[1];
				e.mentionID = Integer.parseInt(c[2]);
				e.predRelation = c[3];
				e.predScore = Double.parseDouble(c[4]);
				predictions.add(e);
			}
			r.close();
		}
		
		Map<String,List<Label>> labels = new HashMap<String,List<Label>>();
		{
			BufferedReader r = new BufferedReader(new InputStreamReader
					(new FileInputStream(labelsFile), "utf-8"));
			String l = null;
			while ((l = r.readLine())!= null) {
				String[] c = l.split("\t");
				String key = c[0] + "\t" + c[1] + "\t" + c[2]; // arg1, arg2, mentionID
				Label label = new Label();
				label.relation = c[3]; 
				
				// ignore relation /location/administrative_division/country
				// since it is just the inverse of 
				// /location/country/administrative_divisions which is also
				// in the dataset
				if (label.relation.equals("/location/administrative_division/country")) continue;
				
				label.tf = c[4].equals("y") || c[4].equals("indirect");
				label.name1 = c[6];
				label.name2 = c[7];
				label.sentence = c[8];
				
				List<Label> ll = labels.get(key);
				if (ll == null) {
					ll = new ArrayList<Label>();
					labels.put(key, ll);
				}
				ll.add(label);
			}
			r.close();
		}

		// sort predictions by decreasing score
		Collections.sort(predictions, new Comparator<Example>() {
			public int compare(Example e1, Example e2) {
				if (e1.predScore > e2.predScore) return -1; else return 1;
			}
		});

		// max recall
		int MAX_TP = 0;
		for (List<Label> ll : labels.values()) {
			for (Label l : ll)
				if (!l.relation.equals("NA") && l.tf) MAX_TP++;
		}
		
		
		List<double[]> curve = new ArrayList<double[]>();
		
		int TP = 0, FP = 0, FN = 0;
		for (Example e : predictions) {
			String key = e.arg1 + "\t" + e.arg2 + "\t" + e.mentionID;
			List<Label> ll = labels.get(key);
			if (ll != null) {
				for (Label l : ll) {
					if (l.relation.equals(e.predRelation)) {
						if (l.tf) TP++;
						else FP++;
					} else {
						if (l.tf) FN++; // && e.predRelation.equals("NA")) FN++;
						//else TN++;
					}
				}
				double precision = TP / (double)(TP + FP);
				double recall = TP / (double)(MAX_TP);
				curve.add(new double[] { precision, recall } );
			}
		}
		
		{
			for (double[] d : curve) {
				System.out.println(d[1] + "\t" + d[0]);
			}
		}
		
		// print the most confident predictions
		if (outputTopSentences)
		{
			for (Example e : predictions) {
				String key = e.arg1 + "\t" + e.arg2 + "\t" + e.mentionID;
				List<Label> ll = labels.get(key);
				if (ll != null) {
					StringBuilder sb = new StringBuilder();
					for (Label l : ll) {
						sb.append(l.tf + ":" + l.relation + ", ");
					}
					Label l1 = ll.get(0);
					System.out.println(l1.name1 + "\t" + l1.name2 + "\t" + 
							e.predRelation + "\t" + sb.toString() + "\t" + 
							l1.sentence + "\t" + e.predScore + "\n");
				}
			}
		}		
	}

	static class Example {
		String arg1;
		String arg2;
		int mentionID;
		String predRelation;
		double predScore;
		boolean correct = false;
	}
	
	static class Label {
		String relation;
		boolean tf;
		String name1;
		String name2;
		String sentence;
	}
}