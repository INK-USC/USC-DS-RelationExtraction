package edu.uw.cs.multir.learning.data;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MemoryDataset implements Dataset {
	
	private MILDocument[] docs;
	private int cursor = 0;
	
	public MemoryDataset() {  }
	
	public MemoryDataset(String file) 
		throws IOException {
		MILDocument d = new MILDocument();
		List<MILDocument> l = new ArrayList<MILDocument>();
		DataInputStream dis = new DataInputStream(new BufferedInputStream
				(new FileInputStream(file)));
		while (d.read(dis)) {
			l.add(d);
			d = new MILDocument();
		}
		dis.close();
		docs = l.toArray(new MILDocument[0]);
	}
	
	public int numDocs() { return docs.length; }

	public void shuffle(Random random) {
		for (int i=0; i < docs.length; i++) {
			// pick element that we want to swap with
			int e = i + random.nextInt(docs.length - i);
			MILDocument tmp = docs[e];
			docs[e] = docs[i];
			docs[i] = tmp;
		}
	}

	public MILDocument next() { 
		if (cursor < docs.length) 
			return docs[cursor++]; 
		else return null;
	}

	public boolean next(MILDocument doc) {
		if (cursor < docs.length) {
			MILDocument d = docs[cursor++];
			doc.arg1 = d.arg1;
			doc.arg2 = d.arg2;
			doc.features = d.features;
			doc.mentionIDs = d.mentionIDs;
			doc.numMentions = d.numMentions;
			doc.Y = d.Y;
			doc.Z = d.Z;
			return true;
		}
		return false;
	}

	public void reset() {
		cursor = 0;
	}

}