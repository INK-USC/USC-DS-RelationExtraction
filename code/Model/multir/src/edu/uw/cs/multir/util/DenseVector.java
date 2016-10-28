package edu.uw.cs.multir.util;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class DenseVector {

	public double[] vals;
	
	public DenseVector(int length) {
		this.vals = new double[length];
	}
	
	public double dotProduct(SparseBinaryVector v) {
		return dotProduct(this, v);
	}

	public void reset() {
		for (int i=0; i < vals.length; i++) vals[i] = 0;
	}
	
	public static double dotProduct(DenseVector v1, SparseBinaryVector v2) {
		double sum = 0;
		for (int i=0; i < v2.num; i++) {
			sum += v1.vals[v2.ids[i]];
		}
		return sum;
	}

	public DenseVector copy() {
		DenseVector n = new DenseVector(vals.length);
		System.arraycopy(vals, 0, n.vals, 0, vals.length);
		return n;
	}

	public void scale(float factor) {
		for (int i=0; i < vals.length; i++)
			vals[i] *= factor;
	}
	
	public void addSparse(SparseBinaryVector v, double factor) {
		for (int i=0; i < v.num; i++)
			vals[v.ids[i]] += factor;
	}
	
	public static DenseVector sum(DenseVector v1, DenseVector v2, double factor) {
		DenseVector n = new DenseVector(v1.vals.length);
		for (int i=0; i < v1.vals.length; i++)
			n.vals[i] = v1.vals[i] + factor * v2.vals[i];
		return n;
	}
	
	public static DenseVector scale(DenseVector v, float factor) {
		DenseVector n = v.copy();
		n.scale(factor);
		return n;
	}

	public void serialize(OutputStream os) 
		throws IOException {
		DataOutputStream dos = new DataOutputStream(os);
		dos.writeInt(this.vals.length);
		for (int i=0; i < this.vals.length; i++) {			
			dos.writeDouble(this.vals[i]);
		}
	}
	
	public void deserialize(InputStream is)
		throws IOException {
		DataInputStream dis = new DataInputStream(is);
		int len = dis.readInt();
		this.vals = new double[len];
		for (int i=0; i < len; i++) {
			this.vals[i] = dis.readDouble();
		}
	}

	public DenseVector sum(DenseVector v, float factor) {
		return sum(this, (DenseVector)v, factor);
	}	
}
