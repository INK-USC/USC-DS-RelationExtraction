package edu.uw.cs.multir.learning.algorithm;

import java.util.Random;

import edu.uw.cs.multir.learning.data.Dataset;
import edu.uw.cs.multir.learning.data.MILDocument;
import edu.uw.cs.multir.util.DenseVector;
import edu.uw.cs.multir.util.SparseBinaryVector;

public class AveragedPerceptron {

	public int maxIterations = 50;
	public boolean computeAvgParameters = true;
	public boolean updateOnTrueY = true;
	public double delta = 1;

	private Scorer scorer;
	private Model model;
	private Random random;

	public AveragedPerceptron(Model model, Random random) {
		scorer = new Scorer();
		this.model = model;
		this.random = random;
	}

	// the following two are actually not storing weights:
	// the first is storing the iteration in which the average weights were
	// last updated, and the other is storing the next update value
	private Parameters avgParamsLastUpdatesIter;
	private Parameters avgParamsLastUpdates;

	private Parameters avgParameters;
	private Parameters iterParameters;

	public Parameters train(Dataset trainingData) {

		if (computeAvgParameters) {
			avgParameters = new Parameters();
			avgParameters.model = model;
			avgParameters.init();

			avgParamsLastUpdatesIter = new Parameters();
			avgParamsLastUpdates = new Parameters();
			avgParamsLastUpdatesIter.model = avgParamsLastUpdates.model = model;
			avgParamsLastUpdatesIter.init();
			avgParamsLastUpdates.init();
		}

		iterParameters = new Parameters();
		iterParameters.model = model;
		iterParameters.init();

		for (int i = 0; i < maxIterations; i++)
			trainingIteration(i, trainingData);

		if (computeAvgParameters) finalizeRel();
		
		return (computeAvgParameters) ? avgParameters : iterParameters;
	}

	int avgIteration = 0;

	public void trainingIteration(int iteration, Dataset trainingData) {
		System.out.println("iteration " + iteration);

		MILDocument doc = new MILDocument();

		trainingData.shuffle(random);

		trainingData.reset();

		while (trainingData.next(doc)) {

			// compute most likely label under current parameters
			Parse predictedParse = FullInference.infer(doc, scorer,
					iterParameters);

			if (updateOnTrueY || !YsAgree(predictedParse.Y, doc.Y)) {
				// if this is the first avgIteration, then we need to initialize
				// the lastUpdate vector
				if (computeAvgParameters && avgIteration == 0)
					avgParamsLastUpdates.sum(iterParameters, 1.0f);

				Parse trueParse = ConditionalInference.infer(doc, scorer,
					iterParameters);
				update(predictedParse, trueParse);
			}

			if (computeAvgParameters) avgIteration++;
		}
	}

	private boolean YsAgree(int[] y1, int[] y2) {
		if (y1.length != y2.length)
			return false;
		for (int i = 0; i < y1.length; i++)
			if (y1[i] != y2[i])
				return false;
		return true;
	}

	// a bit dangerous, since scorer.setDocument is called only inside inference
	public void update(Parse pred, Parse tru) {
		int numMentions = tru.Z.length;

		// iterate over mentions
		for (int m = 0; m < numMentions; m++) {
			int truRel = tru.Z[m];
			int predRel = pred.Z[m];

			if (truRel != predRel) {
				SparseBinaryVector v1a = scorer.getMentionRelationFeatures(
						tru.doc, m, truRel);
				updateRel(truRel, v1a, delta, computeAvgParameters);

				SparseBinaryVector v2a = scorer.getMentionRelationFeatures(
						tru.doc, m, predRel);
				updateRel(predRel, v2a, -delta, computeAvgParameters);
			}
		}
	}

	private void updateRel(int toState, SparseBinaryVector features,
			double delta, boolean useIterAverage) {
		iterParameters.relParameters[toState].addSparse(features, delta);

		if (useIterAverage) {
			DenseVector lastUpdatesIter = (DenseVector) avgParamsLastUpdatesIter.relParameters[toState];
			DenseVector lastUpdates = (DenseVector) avgParamsLastUpdates.relParameters[toState];
			DenseVector avg = (DenseVector) avgParameters.relParameters[toState];
			DenseVector iter = (DenseVector) iterParameters.relParameters[toState];
			for (int j = 0; j < features.num; j++) {
				int id = features.ids[j];
				if (lastUpdates.vals[id] != 0)
					avg.vals[id] += (avgIteration - lastUpdatesIter.vals[id])
							* lastUpdates.vals[id];

				lastUpdatesIter.vals[id] = avgIteration;
				lastUpdates.vals[id] = iter.vals[id];
			}
		}
	}

	private void finalizeRel() {
		for (int s = 0; s < model.numRelations; s++) {
			DenseVector lastUpdatesIter = (DenseVector) avgParamsLastUpdatesIter.relParameters[s];
			DenseVector lastUpdates = (DenseVector) avgParamsLastUpdates.relParameters[s];
			DenseVector avg = (DenseVector) avgParameters.relParameters[s];
			for (int id = 0; id < avg.vals.length; id++) {
				if (lastUpdates.vals[id] != 0) {
					avg.vals[id] += (avgIteration - lastUpdatesIter.vals[id])
							* lastUpdates.vals[id];
					lastUpdatesIter.vals[id] = avgIteration;
				}
			}
		}
	}
}
