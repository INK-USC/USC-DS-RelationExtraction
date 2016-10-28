package edu.uw.cs.multir.learning.algorithm;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import edu.uw.cs.multir.util.DenseVector;

public class Parameters {

	public DenseVector[] relParameters;
	
	public Model model;

	private DenseVector sum(DenseVector v1, DenseVector v2, float factor) {
		if (v1 == null && v2 == null) return null;
		else if (v2 == null) return v1.copy();
		else if (v1 == null) {
			DenseVector v = v2.copy();
			v.scale(factor);
			return v;
		}
		else return v1.sum(v2, factor);
	}
	
	public void sum(Parameters p, float factor) {
		for (int i=0; i < relParameters.length; i++)
			relParameters[i] = sum(relParameters[i], p.relParameters[i], factor);
	}
	
	public void init() {
		if (relParameters == null) {
			relParameters = new DenseVector[model.numRelations];
			System.out.println("requesting " + (8*relParameters.length*
					(long)model.numFeaturesPerRelation[0]) + " bytes");
			for (int j=0; j < relParameters.length; j++) {
				relParameters[j] = 
					new DenseVector(model.numFeatures(j));
			}
		}
	}
	
	public void reset() {
		for (int i=0; i < relParameters.length; i++)
			if (relParameters[i] != null)
				relParameters[i].reset();
	}

	public void serialize(OutputStream os) 
		throws IOException {
		DenseVector[] r = relParameters;
		for (int i=0; i < r.length; i++)
			r[i].serialize(os);
	}
	
	public void deserialize(InputStream is) 
		throws IOException {
		init();
		DenseVector[] r = relParameters;
		for (int i=0; i < r.length; i++)
			r[i].deserialize(is);
	}
	
	public void serialize(String file)
		throws IOException {
		OutputStream os = new BufferedOutputStream(new FileOutputStream(file));
		serialize(os);
		os.close();
	}
	
	public void deserialize(String file)
		throws IOException {
		InputStream is = new BufferedInputStream(new FileInputStream(file));
		deserialize(is);
		is.close();
	}
}
