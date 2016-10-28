package edu.uw.cs.multir.util;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class SparseBinaryVector {
	public int[] ids;		// sorted
	public int num;         // the array might not be full

	public SparseBinaryVector() {
		this.ids = new int[0];
		this.num = 0;
	}
	
	public SparseBinaryVector(int[] ids, int num) {
		this.ids = ids;
		this.num = num;
	}
	
	public void reset() {
		num = 0;
	}
	
	public SparseBinaryVector copy() {
		SparseBinaryVector n = new SparseBinaryVector(new int[num], num);
		System.arraycopy(ids, 0, n.ids, 0, num);
		return n;
	}
	
	public double dotProduct(SparseBinaryVector v) {
		int i = 0, j = 0;
		
		double sum = 0;
		while (i < num && j < v.num) {
			if (ids[i] < v.ids[j])
				i++;
			else if (ids[i] > v.ids[j])
				j++;
			else {
				sum += 1;
				i++; j++;
			}
		}
		return sum;
	}
	
	public void serialize(OutputStream os) 
		throws IOException {
		DataOutputStream dos = new DataOutputStream(os);
		dos.writeInt(this.num);
		for (int i=0; i < this.num; i++) {
			dos.writeInt(this.ids[i]);
		}
	}
	
	public void deserialize(InputStream is)
		throws IOException {
		DataInputStream dis = new DataInputStream(is);
		this.num = dis.readInt();
		this.ids = new int[this.num];
		for (int i=0; i < this.num; i++) {
			this.ids[i] = dis.readInt();
		}
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		for (int i=0; i < num; i++) {
			if (i > 0) sb.append(" ");
			sb.append(ids[i]);
		}
		return sb.toString();
	}
}