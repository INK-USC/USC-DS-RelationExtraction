package edu.uw.cs.multir.learning.data;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;

import edu.uw.cs.multir.util.SparseBinaryVector;


/*
 * The purpose of this data structure is to keep all relevant information for
 * learning while using as little memory as possible. Using less memory helps
 * keeping more records in memory at the same time, thus improving speed.
 */
public class MILDocument {
	public static final int MNT_CAPACITY = 2;
	
	public String arg1, arg2;
	
	public int random = 0;
	
	// relations between arg1 and arg2, sorted by ID
	public int[] Y;
	
	// mentions of this entity pair
	public int numMentions = 0;
	public int[] mentionIDs;
	public int[] Z;
	public SparseBinaryVector[] features;
	
	public MILDocument() {
		mentionIDs = new int[MNT_CAPACITY];
		Z = new int[MNT_CAPACITY];
		features = new SparseBinaryVector[MNT_CAPACITY];
	}
	
	public void clear() {
		numMentions = 0;
	}
	
	public void setCapacity(int targetSize) {
		int[] newMentionIDs = new int[targetSize];
		int[] newZ = new int[targetSize];
		SparseBinaryVector[] newFeatures = new SparseBinaryVector[targetSize];
		if (numMentions > 0) {
			System.arraycopy(mentionIDs, 0, newMentionIDs, 0, numMentions);
			System.arraycopy(Z, 0, newZ, 0, numMentions);
			System.arraycopy(features, 0, newFeatures, 0, numMentions);
		}
		mentionIDs = newMentionIDs;
		Z = newZ;
		features = newFeatures;
	}
	
	
	public boolean read(DataInputStream dis) throws IOException {
		try {
			random = dis.readInt();
			arg1 = dis.readUTF();
			arg2 = dis.readUTF();			
			int lenY = dis.readInt();
			Y = new int[lenY];
			for (int i=0; i < lenY; i++) Y[i] = dis.readInt();
			int numMentions = dis.readInt();
			if (numMentions > mentionIDs.length) setCapacity(numMentions);
			this.numMentions = numMentions;
			for (int i=0; i < numMentions; i++) {
				mentionIDs[i] = dis.readInt();
				Z[i] = dis.readInt();
				if (features[i] == null) features[i] = new SparseBinaryVector();
				features[i].deserialize(dis);
			}
			
			//arg1 = arg2 = null;
			//mentionIDs = null;
			
			return true;
		} catch (EOFException e) { return false; }
	}
	
	public void write(DataOutputStream dos) throws IOException {
		dos.writeInt(random);
		dos.writeUTF(arg1);
		dos.writeUTF(arg2);
		dos.writeInt(Y.length);
		for (int i=0; i < Y.length; i++)
			dos.writeInt(Y[i]);
		dos.writeInt(numMentions);
		for (int i=0; i < numMentions; i++) {
			dos.writeInt(mentionIDs[i]);
			dos.writeInt(Z[i]);
			features[i].serialize(dos);
		}
	}
}