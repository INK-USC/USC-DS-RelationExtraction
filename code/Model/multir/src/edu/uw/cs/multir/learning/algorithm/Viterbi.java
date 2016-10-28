package edu.uw.cs.multir.learning.algorithm;

import edu.uw.cs.multir.learning.data.MILDocument;

public class Viterbi {

	private Scorer parseScorer;
	private Model model;
	
	public Viterbi(Model model, Scorer parseScorer) {
		this.model = model;
		this.parseScorer = parseScorer;
	}
	
	public Parse parse(MILDocument doc, int mention) {
		int numRelations = model.numRelations;

		// relation X argsReversed
		double[] scores = new double[numRelations];
				
		// lookup signature
		for (int s = 0; s < numRelations; s++)
			scores[s] = parseScorer.scoreMentionRelation(doc, mention, s);

		int bestRel = 0;
		for (int r = 0; r < model.numRelations; r++) {
			if (scores[r] > scores[bestRel]) {
				bestRel = r; }
		}

		Parse p = new Parse(bestRel, scores[bestRel]);
		p.scores = scores;
		return p;
	}
	
	public static class Parse {
		// MPE
		public int state;
		public double score;
		
		// scores of all assignments
		public double[] scores;
		
		Parse(int state, double score) {
			this.state = state;
			this.score = score;
		}
	}
}
