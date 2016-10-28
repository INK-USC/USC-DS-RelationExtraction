package edu.uw.cs.multir.learning.algorithm;

import edu.uw.cs.multir.learning.data.MILDocument;
import edu.uw.cs.multir.util.DenseVector;
import edu.uw.cs.multir.util.SparseBinaryVector;

public class Scorer {
	private Parameters params;
	
	public Scorer() {}
	
	// scoring on mention documents, all 2*numRelation	
	public double scoreMentionRelation(MILDocument doc, int m, int rel) {
		double sum = 0;
		DenseVector p = params.relParameters[rel];
		sum += p.dotProduct(doc.features[m]);
		return sum;
	}
	
	// need to consider additional features that are dependent on rel ...
	public SparseBinaryVector getMentionRelationFeatures(MILDocument doc, int m, int rel) {
		return doc.features[m];
	}
	
	public void setParameters(Parameters params) {
		this.params = params;
	}
}
