package edu.uw.cs.multir.preprocess;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;

import edu.uw.cs.multir.learning.data.MILDocument;
import edu.uw.cs.multir.util.SparseBinaryVector;
import cc.factorie.protobuf.DocumentProtos.Relation;
import cc.factorie.protobuf.DocumentProtos.Relation.RelationMentionRef;


public class ConvertProtobufToMILDocument {

	public static void main(String[] args) throws IOException {
		// arg1 is protobuf file
		// arg2 is MIL file		
		String input = args[0];
		String output= args[1];
		String mappingFile = args[2];
		boolean writeMapping = false;
		boolean writeRelations = false;
		convert(input, output, mappingFile, writeMapping, writeRelations);
	}
	
	public static void convert(String input, String output, String mappingFile, 
			boolean writeFeatureMapping, boolean writeRelationMapping) throws IOException {
		
		// This tool can be used in two ways:
		//  1) a new Mapping is created and saved at the end
		//  2) an existing Mapping is used; non-existent relations
		//     or features are ignored
		
		Mappings m = new Mappings();
		
		if (!writeFeatureMapping || !writeRelationMapping)
			m.read(mappingFile);
		else
			// ensure that relation NA gets ID 0
			m.getRelationID("NA", true);
		
		DataOutputStream os = new DataOutputStream
			(new BufferedOutputStream(new FileOutputStream(output)));
	
	    InputStream is = new GZIPInputStream(
	    		new BufferedInputStream
	    		(new FileInputStream(input)));
	    Relation r = null;
	    MILDocument doc = new MILDocument();
	    
	    int count = 0;
    	    
	    while ((r = Relation.parseDelimitedFrom(is))!=null) {
	    	if (++count % 10000 == 0) System.out.println(count);

	    	doc.clear();
	    	
	    	doc.arg1 = r.getSourceGuid();
	    	doc.arg2 = r.getDestGuid();
	    	
	    	// set relations
	    	{
		    	String[] rels = r.getRelType().split(",");
		    	int[] irels = new int[rels.length];
		    	for (int i=0; i < rels.length; i++)
		    		irels[i] = m.getRelationID(rels[i], writeRelationMapping);
		    	Arrays.sort(irels);
		    	// ignore NA and non-mapped relations
		    	int countUnique = 0;
		    	for (int i=0; i < irels.length; i++)
		    		if (irels[i] > 0 && (i == 0 || irels[i-1] != irels[i]))
		    			countUnique++;
		    	doc.Y = new int[countUnique];
		    	int pos = 0;
		    	for (int i=0; i < irels.length; i++)
		    		if (irels[i] > 0 && (i == 0 || irels[i-1] != irels[i]))
		    			doc.Y[pos++] = irels[i];
	    	}
	    	
	    	// set mentions
	    	doc.setCapacity(r.getMentionCount());
	    	doc.numMentions = r.getMentionCount();
	    	
	    	for (int j=0; j < r.getMentionCount(); j++) {
	    		RelationMentionRef rmf = r.getMention(j);
		    	doc.Z[j] = -1;
	    		doc.mentionIDs[j] = j;
	    		SparseBinaryVector sv = doc.features[j] = new SparseBinaryVector();
	    		
	    		int[] fts = new int[rmf.getFeatureCount()];
	    		for (int i=0; i < rmf.getFeatureCount(); i++)
	    			fts[i] = m.getFeatureID(rmf.getFeature(i), writeFeatureMapping);
	    		Arrays.sort(fts);
		    	int countUnique = 0;
		    	for (int i=0; i < fts.length; i++)
		    		if (fts[i] != -1 && (i == 0 || fts[i-1] != fts[i]))
		    			countUnique++;
		    	sv.num = countUnique;
		    	sv.ids = new int[countUnique];
		    	int pos = 0;
		    	for (int i=0; i < fts.length; i++)
		    		if (fts[i] != -1 && (i == 0 || fts[i-1] != fts[i]))
		    			sv.ids[pos++] = fts[i];
	    	}
	    	doc.write(os);
	    }
		
		is.close();
		os.close();
		
		if (writeFeatureMapping || writeRelationMapping)
			m.write(mappingFile);
	}
}
