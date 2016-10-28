package edu.uw.cs.multir.learning.data;

import java.util.Random;

public interface Dataset {
	
	public int numDocs();
	
	public void shuffle(Random random);
	
	public MILDocument next();
	
	public boolean next(MILDocument doc);
	
	public void reset();
}
