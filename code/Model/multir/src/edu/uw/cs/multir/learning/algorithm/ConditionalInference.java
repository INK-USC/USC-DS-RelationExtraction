package edu.uw.cs.multir.learning.algorithm;

import java.util.Arrays;
import java.util.Comparator;

import edu.uw.cs.multir.learning.data.MILDocument;

public class ConditionalInference {

	public static Parse infer(MILDocument doc,
			Scorer parseScorer, Parameters params) {
		int numMentions = doc.numMentions;
		
		Parse parse = new Parse();
		parse.doc = doc;
		parseScorer.setParameters(params);
		
		Viterbi viterbi = new Viterbi(params.model, parseScorer);
		
		Viterbi.Parse[] vp = new Viterbi.Parse[numMentions];
		for (int m = 0; m < numMentions; m++) {
			vp[m] = viterbi.parse(doc, m);
		}
		
		// each mention can be linked to one of the doc relations or NA
		int numRelevantRelations = doc.Y.length + 1;
		
		// solve bipartite graph matching problem
		Edge[] es = new Edge[numMentions * numRelevantRelations];
		for (int m = 0; m < numMentions; m++) {
			// edge from m to NA
			es[numRelevantRelations*m + 0] =
				new Edge(m, 0, vp[m].scores[0]);
			// edge from m to any other relation
			for (int y = 1; y < numRelevantRelations; y++)
				es[numRelevantRelations*m + y] = 
					new Edge(m, y, vp[m].scores[doc.Y[y-1]]);
		}

		// NOTE: strictly speaking, no sorting is necessary
		// in the following steps; however, we do sorting
		// for easier code maintainability
		
		// array to hold solution (mapping from z's to y's)
		int[] z = new int[numMentions];
		for (int i=0; i < numMentions; i++) z[i] = -1;

		// there is a special case where there are more target
		// relations than there are mentions; in this case we
		// only add the highest scoring edges
		if (numMentions < doc.Y.length) {
			// sort edges by decreasing score
			Arrays.sort(es, new Comparator<Edge>() {
				public int compare(Edge e1, Edge e2) {
					double d = e2.score - e1.score;
					if (d < 0) return -1; else return 1;
				}});

			boolean[] ysCovered = new boolean[numRelevantRelations];			
			for (int ei = 0; ei < es.length; ei++) {
				Edge e = es[ei];
				if (e.y == 0) continue;
				if (z[e.m] < 0 && !ysCovered[e.y]) {
					z[e.m] = doc.Y[e.y-1];
					ysCovered[e.y] = true;
				}
			}
		} else {
			// more mentions than target relations: enforce all Ys
			
			// sort by y, then decreasing score
			Arrays.sort(es, new Comparator<Edge>() {
				public int compare(Edge e1, Edge e2) {
					int c = e1.y - e2.y;
					if (c != 0) return c;
					double d = e2.score - e1.score;
					if (d < 0) return -1; else return 1;
				}});
			
			// note that after this step the "es" array has to
			// be indexed differently
			
			// iterate over y's
			for (int y=1; y < numRelevantRelations; y++) {
	
				// find highest weight edge to y, from a
				// mention m which does not yet have an
				// outgoing edge
				
				for (int j=0; j < numMentions; j++) {
					Edge e = es[numMentions*y + j];
					if (z[e.m] < 0) {
						// we can add this edge
						//System.out.println("adding " + doc.Y[y-1]);
						z[e.m] = (y==0)? 0 : doc.Y[y-1];
						break;
					}
				}
			}
			
			// there might be unmapped m's
			// sort by m, then decreasing score
			Arrays.sort(es, new Comparator<Edge>() {
				public int compare(Edge e1, Edge e2) {
					int c = e1.m - e2.m;
					if (c != 0) return c;
					double d = e2.score - e1.score;
					if (d < 0) return -1; else return 1;
				}});

			
			for (int m=0; m < numMentions; m++) {
				if (z[m] < 0) {
					// unmapped mention, need to take highest score
					Edge e = es[numRelevantRelations*m];
					z[m] = e.y == 0? 0 : doc.Y[e.y-1];
				}
			}
		}
		
		// we can now write the results
		parse.Y = doc.Y;
		parse.Z = z;
		parse.score = 0;
		for (int i=0; i < numMentions; i++) {
			parse.score += vp[i].scores[z[i]];
		}
		
		return parse;
	}

	static class Edge {
		int m;
		int y;
		double score;		
		Edge(int m, int y, double score) {
			this.m = m;
			this.y = y;
			this.score = score;
		}
	}
}
