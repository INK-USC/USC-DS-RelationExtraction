package edu.uw.cs.multir.learning.algorithm;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

public class Model {

	public int numRelations;
	public int[] numFeaturesPerRelation;
	
	public int numFeatures(int rel) {
		return numFeaturesPerRelation[rel];
	}
	
	public int noRelationState;
	
	public void read(String file) throws IOException {
		BufferedReader r = new BufferedReader(new InputStreamReader
				(new FileInputStream(file), "utf-8"));
		numRelations = Integer.parseInt(r.readLine());
		numFeaturesPerRelation = new int[numRelations];
		for (int i=0; i < numRelations; i++) {
			numFeaturesPerRelation[i] = Integer.parseInt(r.readLine());
		}
		r.close();
	}
	
	public void write(String file) throws IOException {
		BufferedWriter w = new BufferedWriter(new OutputStreamWriter
				(new FileOutputStream(file), "utf-8"));
		w.write(numRelations + "\n");
		for (int i=0; i < numFeaturesPerRelation.length; i++)
			w.write(numFeaturesPerRelation[i] + "\n");
		w.close();
	}
}
