//
//  FullFctModel.cpp
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-30.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "FullFctModel.h"

void FullFctModel::Init(char* embfile, char* trainfile)
{
    fea_params.head = true;
    fea_params.word_on_path = true;
    fea_params.word_on_path_type = true;
    fea_params.postag = true;
    fea_params.dep = true;
    fea_params.ner = true;
    fea_params.sst = true;
    /*fea_params.postag = false;
    fea_params.dep = true;
    fea_params.ner = false;
    fea_params.sst = false;*/
    fea_params.context = true;
    
    fea_params.tag_fea = true;
    fea_params.dep_fea = true;
    fea_params.hyper_emb = true;
    
    fea_params.dep_path = false;
    fea_params.pos_on_path = false;
    
    fea_params.tri_conv = false;
    fea_params.linear = false;//true;
    /*
    fea_params.head = false;
    fea_params.word_on_path = false;
    fea_params.word_on_path_type = false;
    fea_params.postag = false;
    fea_params.dep = true;
    fea_params.context = false;
    
    fea_params.dep_fea = false;
    fea_params.hyper_emb = false;
    
    fea_params.dep_path = false;
    fea_params.pos_on_path = false;
    
    fea_params.tri_conv = false;
    fea_params.linear = true;*/
    
//    fea_params.head = false;
//    fea_params.word_on_path = true;
//    fea_params.word_on_path_type = false;
//    fea_params.postag = false;
//    fea_params.context = false;
//    fea_params.dep_fea = false;
//    fea_params.hyper_emb = false;
//    fea_params.dep_path = false;
//    fea_params.pos_on_path = false;
    
    emb_model = new EmbeddingModel(embfile);
    layer1_size = (int)emb_model -> layer1_size;
    emb_model_list.push_back(emb_model);

    BuildModelsFromData(trainfile);
//    InitFeatureModel((int)layer1_size, trainfile, type);
    num_models = (int)(coarse_fct_list.size() + deep_fct_list.size() + convolution_fct_list.size());
    num_labels = (int)labeldict.size();
    inst -> scores.resize(num_labels);
    
//    max_frame_slots = fea_model -> max_len - 1;
//    if (fea_model -> max_len == 0) max_frame_slots = 1;
    alpha = 1.0;
    lambda = 0.0;
}

void FullFctModel::InitSubmodels() {
    for (int i = 0; i < coarse_fct_list.size(); i++) {
        coarse_fct_list[i] -> num_labels = num_labels;
        coarse_fct_list[i] -> InitModel();
        coarse_fct_list[i] -> update_emb = update_emb;
        coarse_fct_list[i] -> adagrad = adagrad;
    }
    for (int i = 0; i < deep_fct_list.size(); i++) {
        deep_fct_list[i] -> num_labels = num_labels;
        deep_fct_list[i] -> InitModel();
        deep_fct_list[i] -> update_emb = update_emb;
        deep_fct_list[i] -> adagrad = adagrad;
    }
    for (int i = 0; i < convolution_fct_list.size(); i++) {
        convolution_fct_list[i] -> num_labels = num_labels;
        convolution_fct_list[i] -> InitModel();
        convolution_fct_list[i] -> update_emb = update_emb;
        convolution_fct_list[i] -> adagrad = adagrad;
    }
    if (fea_model != NULL) {
        fea_model -> num_labels = num_labels;
        fea_model -> Init();
        fea_model -> adagrad = adagrad;
    }
}

void FullFctModel::BuildModelsFromData(char* trainfile) {
    layer1_size = (int)emb_model_list[0] -> layer1_size;
    
//    if (type == "SEM_EVAL") {
//        labeldict["Other"] = 0;
//    }
//    else labeldict["NA"] = 0;
    //labeldict["NA"] = 0;
    labeldict["Other"] = 0;
    
    if (fea_params.linear) {
        fea_model = new FeatureModel();
    }
    else fea_model = NULL;
    
    ifstream ifs(trainfile);
    num_inst = 0;
    while (LoadInstanceInit(ifs)) {
        num_inst++;
    }
    
    labellist.resize(labeldict.size());
    for (feat2int::iterator iter = labeldict.begin(); iter != labeldict.end(); iter++) {
        labellist[iter -> second] = iter -> first;
    }
    ifs.close();
}

int FullFctModel::LoadInstanceInit(ifstream &ifs) {
    int id, model_id;
    int beg1 = 0, end1 = 0, beg2 = 0, end2 = 0;
    char line_buf[10000], line_buf2[10000];
    vector<int> trigram_id;
    trigram_id.resize(3);
    ifs.getline(line_buf, 10000, '\n');
    if (!strcmp(line_buf, "")) {
        return 0;
    }
    
    {
        istringstream iss(line_buf);
        iss >> inst -> label;
        feat2int::iterator iter = labeldict.find(inst -> label);
        if (iter == labeldict.end()) {
            id = (int)labeldict.size();
//            cout << inst->label << endl;
            labeldict[inst -> label] = id;
        }
        
        iss >> beg1; iss >> end1;
        inst -> ne1_len = end1 + 1 - beg1;
        for (int i = 0; i < inst -> ne1_len; i++) {
            iss >> inst -> ne1_words[i];
            ToLower(inst -> ne1_words[i]);
        }
        iss >> beg2; iss >> end2;
        inst -> ne2_len = end2 + 1 - beg2;
        for (int i = 0; i < inst -> ne2_len; i++) {
            iss >> inst -> ne2_words[i];
            ToLower(inst -> ne2_words[i]);
        }
    }
    {
        ifs.getline(line_buf, 10000, '\n');
        {
            ifs.getline(line_buf2, 10000, '\n');
            istringstream iss3(line_buf2);
            int tmpint;
            string ne_tag;
            iss3 >> tmpint;
            for (int i = 0; i < inst -> ne1_len; i++) {
                iss3 >> inst -> ne1_types[i];
                inst -> ne1_types[i] = ProcSenseTag(inst -> ne1_types[i]);
                if(fea_params.ner) {
                    iss3 >> ne_tag;
                    inst -> ne1_nes[i] = ProcNeTag(ne_tag);
                }
            }
        }
        {
            ifs.getline(line_buf2, 10000, '\n');
            istringstream iss4(line_buf2);
            int tmpint;
            string ne_tag;
            iss4 >> tmpint;
            for (int i = 0; i < inst -> ne2_len; i++) {
                iss4 >> inst -> ne2_types[i];
                inst -> ne2_types[i] = ProcSenseTag(inst -> ne2_types[i]);
                if (fea_params.ner) {
                    iss4 >> ne_tag;
                    inst -> ne2_nes[i] = ProcNeTag(ne_tag);
                }
            }
        }
        inst -> entitytype = inst -> ne1_types[inst -> ne1_len - 1]
        + '\t' 
        + inst -> ne2_types[inst -> ne2_len - 1];
        
        istringstream iss2(line_buf);
        int count = 0;
        string token, tag, slot_key;
        
        while (iss2 >> token) {
            ToLower(token);
            inst -> words[count] = token;
            if (fea_params.postag) {
                iss2 >> tag; tag = tag.substr(0,2);
            }
            if (fea_params.ner) {
                iss2 >> tag; inst -> word_nes[count] = ProcNeTag(tag);
            }
            if (fea_params.sst) {
                iss2 >> tag; inst -> word_types[count] = ProcSenseTag(tag);
            }
            if (fea_params.dep) {
                iss2 >> inst -> dep_paths[count];
                if (fea_params.linear) {
                    if (inst -> dep_paths[count] > 0) {
                        string feat_key = "FEAT_on_path\t" + inst -> words[count];
                        AddFeature(feat_key);
                    }
                }
            }
            
            if (fea_params.word_on_path) {
                if (count < max(beg1, beg2) && count > min(end1, end2)) {
                    slot_key = "in_between";
                    model_id = AddDeepFctModel(slot_key);
                    slot_key = "FCT_in_between";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                    slot_key = "FCT_in_between_bias";
                    //                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                    if (fea_params.postag) {
                        slot_key = "FCT_in_between_" + tag;
                        //                    AddRealFCTSlot(slot_key);
                    }
                    ostringstream oss;
                    oss << "FCT_in_between_" << (count - min(end1, end2));
                    //                AddRealFCTSlot(oss.str());
                    oss.str("");
                    oss << "FCT_in_between_" << (count - max(beg1, beg2));
                    //                AddRealFCTSlot(oss.str());
                    
                    if (fea_params.word_on_path_type) {
                        slot_key = "FCT_in_between_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                        deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                        slot_key = "FCT_in_between_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                        deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//                    slot_key = "FCT_in_between_nepair\t" + inst -> entitytype;
//                    //            AddRealFCTSlot(slot_key);
                    
                        /*slot_key = "in_between_nepair\t" + inst -> ne1_types[inst -> ne1_len - 1] + '\t' + inst -> ne2_types[inst -> ne2_len - 1];
                        model_id = AddDeepFctModel(slot_key);
                        slot_key = "FCT_bias";
                        deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);*/
            
                        slot_key = "in_between_only";
                        AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], true);
                        
                        if (fea_params.tag_fea && fea_params.sst) {
                            //slot_key = "in_between_self\t" + inst -> word_types[count];
                            //AddCoarseFctModel2List(slot_key, inst -> word_ids[count], true);
                        }
                    
                        /*slot_key = "in_between_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                        model_id = AddDeepFctModel(slot_key);
                        slot_key = "FCT_in_between_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                        deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                        
                        slot_key = "in_between_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                        model_id = AddDeepFctModel(slot_key);
                        slot_key = "FCT_in_between_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                        deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);*/

                        if (fea_params.tag_fea && fea_params.ner) {
                            slot_key = "in_between_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
                            AddCoarseFctModel2List(slot_key, inst -> word_ids[count], true);
                            slot_key = "in_between_ne1_ner\t" + inst -> ne1_nes[inst -> ne1_len - 1];
                            AddCoarseFctModel2List(slot_key, inst -> word_ids[count], true);
                            slot_key = "in_between_ne2_ner\t" + inst -> ne2_nes[inst -> ne2_len - 1];
                            AddCoarseFctModel2List(slot_key, inst -> word_ids[count], true);
                        }
                    }
                }
            }
            if (fea_params.dep_fea) {
                if (inst -> dep_paths[count] > 0) {
                    slot_key = "on_path";
                    model_id = AddDeepFctModel(slot_key);
                    slot_key = "FCT_on_path";
//                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//                    slot_key = "FCT_on_path_bias";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                    slot_key = "FCT_on_path_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                    slot_key = "FCT_on_path_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//                    slot_key = "FCT_in_between_nepair\t" + inst -> entitytype;
//                    AddRealFCTSlot(slot_key);
                    
                    /*slot_key = "on_path_nepair\t" + inst -> ne1_types[inst -> ne1_len - 1] + '\t' + inst -> ne2_types[inst -> ne2_len - 1];
                    model_id = AddDeepFctModel(slot_key);
                    slot_key = "FCT_bias";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);*/
                    slot_key = "on_path_only";
                    AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], true);
                        
                    if (fea_params.tag_fea && fea_params.sst) {
                        //slot_key = "on_path_self\t" + inst -> word_types[count];
                        //AddCoarseFctModel2List(slot_key, inst -> word_ids[count], true);
                    }
                    
                    /*slot_key = "on_path_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                    model_id = AddDeepFctModel(slot_key);
                    slot_key = "FCT_on_path_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                    
                    slot_key = "on_path_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                    model_id = AddDeepFctModel(slot_key);
                    slot_key = "FCT_on_path_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);*/
                        
                        if (fea_params.tag_fea && fea_params.ner) {
                            slot_key = "on_path_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
                            AddCoarseFctModel2List(slot_key, inst -> word_ids[count], true);
                            slot_key = "on_path_ne1_ner\t" + inst -> ne1_nes[inst -> ne1_len - 1];
                            AddCoarseFctModel2List(slot_key, inst -> word_ids[count], true);
                            slot_key = "on_path_ne2_ner\t" + inst -> ne2_nes[inst -> ne2_len - 1];
                            AddCoarseFctModel2List(slot_key, inst -> word_ids[count], true);
                        }
                }
            }
            
            if (fea_params.context) {
                if(count == beg1 - 1){
                    slot_key = "ne1_left";
                    model_id = AddDeepFctModel(slot_key);
                    
                    slot_key = "FCT_ne1_left";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//                    slot_key = "FCT_ne1_left" + inst -> ne1_types[inst -> ne1_len - 1] + "\tne1";
//                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//                    slot_key = "FCT_ne1_left" + inst -> ne2_types[inst -> ne2_len - 1] + "\tne2";
//                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                }
                if(count == beg1 - 2){
                    slot_key = "ne1_left2";
                    model_id = AddDeepFctModel(slot_key);
                    
                    slot_key = "FCT_ne1_left2";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                }
                if(count == end1 + 1){
                    slot_key = "ne1_right";
                    model_id = AddDeepFctModel(slot_key);
                    
                    slot_key = "FCT_ne1_right";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//                    slot_key = "FCT_ne1_right" + inst -> ne1_types[inst -> ne1_len - 1] + "\tne1";
//                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//                    slot_key = "FCT_ne1_right" + inst -> ne2_types[inst -> ne2_len - 1] + "\tne2";
//                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                }
                if(count == end1 + 2){
                    slot_key = "ne1_right2";
                    model_id = AddDeepFctModel(slot_key);
                    
                    slot_key = "FCT_ne1_right2";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                }
                if(count == beg2 - 1){
                    slot_key = "ne2_left";
                    model_id = AddDeepFctModel(slot_key);
                    
                    slot_key = "FCT_ne2_left";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//                    slot_key = "FCT_ne2_left" + inst -> ne2_types[inst -> ne2_len - 1] + "\tne2";
//                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//                    slot_key = "FCT_ne2_left" + inst -> ne1_types[inst -> ne1_len - 1] + "\tne1";
//                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                }
                if(count == beg2 - 2){
                    slot_key = "ne2_left2";
                    model_id = AddDeepFctModel(slot_key);
                    
                    slot_key = "FCT_ne2_left2";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                }
                if(count == end2 + 1){
                    slot_key = "ne2_right";
                    model_id = AddDeepFctModel(slot_key);
                    
                    slot_key = "FCT_ne2_right";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//                    slot_key = "FCT_ne2_right" + inst -> ne2_types[inst -> ne2_len - 1] + "\tne2";
//                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//                    slot_key = "FCT_ne2_right" + inst -> ne1_types[inst -> ne1_len - 1] + "\tne1";
//                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                }
                if(count == end2 + 2){
                    slot_key = "ne2_right2";
                    model_id = AddDeepFctModel(slot_key);
                    
                    slot_key = "FCT_ne2_right2";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                }
            }
            
            count++;
        }
        
        if (fea_params.head) {
            slot_key = "ne1_head";
            model_id = AddDeepFctModel(slot_key);
            slot_key = "FCT_ne1_head";
            deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
            //                slot_key = inst -> entitytype + "\t" + "ne1";
            //                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
            slot_key = "FCT_ne1_head" + inst -> ne1_types[inst -> ne1_len - 1] + "\tne1";
            deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//            slot_key = "FCT_ne1_head" + inst -> ne2_types[inst -> ne2_len - 1] + "\tne2";
//            deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
            
            slot_key = "ne1_head_pair";
            model_id = AddDeepFctModel(slot_key);
            slot_key = "FCT_ne1_head_pair";
            deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
            slot_key = "FCT_ne1_head_pair\t" + inst -> ne2_types[inst -> ne2_len - 1];
            deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
            
            if (fea_params.hyper_emb) {
                slot_key = "ne1_hyper";
                model_id = AddDeepFctModel(slot_key);
                slot_key = "FCT_ne1_hyper";
                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                slot_key = "FCT_ne1_hyper_pair\t" + inst -> ne2_types[inst -> ne2_len - 1];
                //deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                slot_key = "FCT_ne1_hyper\t" + inst -> ne1_types[inst -> ne1_len - 1];
                //deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);

            }
            
            slot_key = "ne1_head_only";
            AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], true);
            
            /*slot_key = "ne1_head\t" + inst -> ne1_types[inst -> ne1_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], true);
            
            slot_key = "ne1_head_pair\t" + inst -> ne2_types[inst -> ne2_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], true);*/
            
            slot_key = "ne1_head_ner\t" + inst -> ne1_nes[inst -> ne1_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], true);
            
            slot_key = "ne1_head_ner2\t" + inst -> ne2_nes[inst -> ne2_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], true);
            
            slot_key = "ne1_head_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], true);
            //slot_key = "ne1_head_nepair\t" + inst -> entitytype;
            //AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], true);
            
            slot_key = "ne2_head";
            model_id = AddDeepFctModel(slot_key);
            slot_key = "FCT_ne2_head";
            deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
            //                slot_key = inst -> entitytype + "\t" + "ne1";
            //                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//            slot_key = "FCT_ne2_head" + inst -> ne1_types[inst -> ne1_len - 1] + "\tne1";
//            deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
            slot_key = "FCT_ne2_head" + inst -> ne2_types[inst -> ne2_len - 1] + "\tne2";
            deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
            
            slot_key = "ne2_head_pair";
            model_id = AddDeepFctModel(slot_key);
            slot_key = "FCT_ne2_head_pair";
            deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
            slot_key = "FCT_ne2_head_pair\t" + inst -> ne1_types[inst -> ne1_len - 1];
            deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
            
            if (fea_params.hyper_emb) {
                slot_key = "ne2_hyper";
                model_id = AddDeepFctModel(slot_key);
                slot_key = "FCT_ne2_hyper";
                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                slot_key = "FCT_ne2_hyper_pair\t" + inst -> ne1_types[inst -> ne1_len - 1];
                //deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                slot_key = "FCT_ne2_hyper\t" + inst -> ne2_types[inst -> ne2_len - 1];
                //deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
            }
            
            slot_key = "ne2_head_only";
            AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], true);
            
            /*slot_key = "ne2_head\t" + inst -> ne2_types[inst -> ne2_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], true);
            
            slot_key = "ne2_head_pair\t" + inst -> ne1_types[inst -> ne1_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], true);*/
            
            slot_key = "ne2_head_ner\t" + inst -> ne2_nes[inst -> ne2_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], true);
            
            slot_key = "ne2_head_ner2\t" + inst -> ne1_nes[inst -> ne1_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], true);
            
            slot_key = "ne2_head_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], true);
            //slot_key = "ne2_head_nepair\t" + inst -> entitytype;
            //AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], true);
        }
        
        if (fea_params.dep_path)
        {
            ifs.getline(line_buf, 10000, '\n');
            istringstream iss3(line_buf);
            int path_len;
            string tmp_str;
            int token_pos;
            iss3 >> path_len;
            string path_len_fea;
            if (fea_params.pos_on_path) {
                ostringstream oss;
                if (path_len < 5) oss << "path_len\t" << path_len;
                else oss << "path_len\t5";
                path_len_fea = oss.str();
            }
            for (int i = 0; i < path_len - 1; i++) {
                iss3 >> tmp_str;
                iss3 >> tmp_str;
                iss3 >> tmp_str;
                iss3 >> token_pos;
                
                if (fea_params.pos_on_path) {
//                    ostringstream oss;
//                    oss << "on_path_pos\t" << inst -> entitytype << "\t" << i;
//                    AddCoarseFctModel2List(oss.str(), inst -> word_ids[token_pos], true);
                    
                    AddCoarseFctModel2List(path_len_fea, inst -> word_ids[token_pos], true);
//                    slot_key = path_len_fea + "\t" + inst -> entitytype;
//                    AddCoarseFctModel2List(slot_key, inst -> word_ids[token_pos], true);
                }
            }
        }
        
        inst -> len = count + 1;
        if (inst -> len > max_len) {max_len = inst -> len;}
        
        if (fea_params.tri_conv) {
            for (int i = 0; i < inst -> len - 1; i++) {
                if (i < max(beg1, beg2) && i > min(end1, end2)) {
                    slot_key = "tri_in_between";
                    model_id = AddDeepFctModel(slot_key);
                    slot_key = "FCT_tri_in_between_mid_bias";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                    slot_key = "FCT_tri_in_between_mid";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                    slot_key = "FCT_tri_in_between_left_bias";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                    slot_key = "FCT_tri_in_between_left";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                    slot_key = "FCT_tri_in_between_right_bias";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                    slot_key = "FCT_tri_in_between_right";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                }
            }
        }
        
        if (fea_params.linear) {
            string feat_key = "FEAT_bias";
            AddFeature(feat_key);
            feat_key = "FEAT_ne1_type\t" + inst -> ne1_types[inst -> ne1_len - 1];
            AddFeature(feat_key);
            feat_key = "FEAT_ne2_type\t" + inst -> ne2_types[inst -> ne2_len - 1];
            AddFeature(feat_key);
            feat_key = "FEAT_ne1_word\t" + inst -> ne1_words[inst -> ne1_len - 1];
            AddFeature(feat_key);
            feat_key = "FEAT_ne2_word\t" + inst -> ne2_words[inst -> ne2_len - 1];
            AddFeature(feat_key);
            feat_key = "FEAT_ne_type_pair\t" + inst -> ne1_types[inst -> ne1_len - 1] + "\t" + inst -> ne2_types[inst -> ne2_len - 1];
            AddFeature(feat_key);
            /*feat_key = "FEAT_ne_word_pair\t" + inst -> ne1_words[inst -> ne1_len - 1] + "\t" + inst -> ne2_words[inst -> ne2_len - 1];
            AddFeature(feat_key);
            feat_key = "FEAT_ne_type_inv_pair\t" + inst -> ne2_types[inst -> ne2_len - 1] + "\t" + inst -> ne1_types[inst -> ne1_len - 1];
            AddFeature(feat_key);*/
        }
    }
    
    return 1;
}

int FullFctModel::LoadInstance(ifstream &ifs, int type) {
    return LoadInstance(ifs);
    //    if (type == FCT_INST) return LoadInstance(ifs);
    //    else if (type == SEMEVAL_INST) return LoadInstanceSemEval(ifs);
    //    else return -2;
}

int FullFctModel::LoadInstance(ifstream &ifs) {
    int id, model_id;
    word2int::iterator iter2;
    int beg1 = 0, end1 = 0, beg2 = 0, end2 = 0;
    char line_buf[10000], line_buf2[10000];
    vector<int> trigram_id;
    trigram_id.resize(3);
    ifs.getline(line_buf, 10000, '\n');
    if (!strcmp(line_buf, "")) {
        return 0;
    }
    
    for (int i = 0; i < deep_fct_list.size(); i++) {
        deep_fct_list[i] -> inst -> Clear();
    }
    for (int i = 0; i < coarse_fct_list.size(); i++) {
        coarse_fct_list[i] -> inst -> Clear();
    }
    for (int i = 0; i < convolution_fct_list.size(); i++) {
        convolution_fct_list[i] -> inst -> Clear();
    }
    if (fea_model != NULL) fea_model -> inst -> Clear();
    
    {
        istringstream iss(line_buf);
        iss >> inst -> label;
        feat2int::iterator iter = labeldict.find(inst -> label);
        if (iter == labeldict.end()) inst -> label_id = -1;
        else inst -> label_id = iter -> second;
        
        iss >> beg1; iss >> end1;
        inst -> ne1_len = end1 + 1 - beg1;
        for (int i = 0; i < inst -> ne1_len; i++) {
            iss >> inst -> ne1_words[i];
            ToLower(inst -> ne1_words[i]);
        }
        for (int i = 0; i < inst -> ne1_len; i++) {
            iter2 = emb_model -> vocabdict.find(inst -> ne1_words[i]);
            if (iter2 != emb_model -> vocabdict.end()) inst -> ne1_ids[i] = iter2 -> second;
            else inst -> ne1_ids[i] = -1;
        }
        iss >> beg2; iss >> end2;
        inst -> ne2_len = end2 + 1 - beg2;
        for (int i = 0; i < inst -> ne2_len; i++) {
            iss >> inst -> ne2_words[i];
            ToLower(inst -> ne2_words[i]);
        }
        for (int i = 0; i < inst -> ne2_len; i++) {
            iter2 = emb_model -> vocabdict.find(inst -> ne2_words[i]);
            if (iter2 != emb_model -> vocabdict.end()) inst -> ne2_ids[i] = iter2 -> second;
            else inst -> ne2_ids[i] = -1;
        }
    }
    {
        ifs.getline(line_buf, 10000, '\n');
        {
            ifs.getline(line_buf2, 10000, '\n');
            istringstream iss3(line_buf2);
            int tmpint;
            iss3 >> tmpint;
            for (int i = 0; i < inst -> ne1_len; i++) {
                iss3 >> inst -> ne1_types[i];
                inst -> ne1_types[i] = ProcSenseTag(inst -> ne1_types[i]);
                if (fea_params.ner) {
                    iss3 >> inst -> ne1_nes[i]; inst -> ne1_nes[i] = ProcNeTag(inst -> ne1_nes[i]);
                }
                
                iter2 = emb_model -> vocabdict.find("SST:" + inst -> ne1_types[i]);
                if (iter2 != emb_model -> vocabdict.end()) inst -> ne1_type_ids[i] = iter2 -> second;
                else inst -> ne1_type_ids[i] = -1;
            }
        }
        {
            ifs.getline(line_buf2, 10000, '\n');
            istringstream iss4(line_buf2);
            int tmpint;
            iss4 >> tmpint;
            for (int i = 0; i < inst -> ne2_len; i++) {
                iss4 >> inst -> ne2_types[i];
                inst -> ne2_types[i] = ProcSenseTag(inst -> ne2_types[i]);
                if (fea_params.ner) {
                    iss4 >> inst -> ne2_nes[i]; inst -> ne2_nes[i] = ProcNeTag(inst -> ne2_nes[i]);
                }
                
                iter2 = emb_model -> vocabdict.find("SST:" + inst -> ne2_types[i]);
                if (iter2 != emb_model -> vocabdict.end()) inst -> ne2_type_ids[i] = iter2 -> second;
                else inst -> ne2_type_ids[i] = -1;
            }
        }
        inst -> entitytype = inst -> ne1_types[inst -> ne1_len - 1]
        + '\t' + inst -> ne2_types[inst -> ne2_len - 1];
        
        istringstream iss2(line_buf);
        int count = 0;
        string token, tag, slot_key;
        
        while (iss2 >> token) {
            ToLower(token);
            inst -> words[count] = token;
            iter2 = emb_model -> vocabdict.find(token);
            if (iter2 != emb_model -> vocabdict.end()) inst -> word_ids[count] = iter2 -> second;
            else inst -> word_ids[count] = -1;
            
            if (fea_params.postag) {
                iss2 >> tag; inst -> tags[count] = tag.substr(0,2);
            }
            if (fea_params.ner) {
                iss2 >> tag; inst -> word_nes[count] = ProcNeTag(tag);
            }
            if (fea_params.sst) {
                iss2 >> tag; inst -> word_types[count] = ProcSenseTag(tag);
            }
            if (fea_params.dep) {
                iss2 >> inst -> dep_paths[count];
                if (fea_params.linear) {
                    if (inst -> dep_paths[count] > 0) {
                        string feat_key = "FEAT_on_path\t" + inst -> words[count];
                        SearchFeature(feat_key);
                    }
                }
            }
            if (fea_params.word_on_path) {
                if (count < max(beg1, beg2) && count > min(end1, end2)) {
                    slot_key = "in_between";
                    model_id = SearchDeepFctSlot(slot_key);
                    if (model_id >= 0) {
                    FctDeepModel* p_model = deep_fct_list[model_id];
                    RealFctPathInstance* p_inst = p_model -> inst;
                    p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                    
                    slot_key = "FCT_in_between";
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
//                    slot_key = "FCT_in_between_bias";
//                    id = p_model -> SearchRealFCTSlot(slot_key);
//                    p_inst -> PushFctFea(id, p_inst -> count);
                    if (fea_params.postag) {
                        slot_key = "FCT_in_between_" + tag;
                        //                    id = SearchRealFCTSlot(slot_key);
                        //                    p_inst -> PushFCTDepFea(id, count);
                    }
                    ostringstream oss;
                    oss << "FCT_in_between_" << (count - min(end1, end2));
                    //                id = SearchRealFCTSlot(oss.str());
                    //                p_inst -> PushFCTDepFea(id, count);
                    oss.str("");
                    oss << "FCT_in_between_" << (count - max(beg1, beg2));
                    
                    slot_key = "FCT_in_between_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
                    slot_key = "FCT_in_between_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
//                    slot_key = "FCT_in_between_nepair\t" + inst -> entitytype;
                    p_inst -> count++;

                    }
                    
                    /*slot_key = "in_between_nepair\t" + inst -> ne1_types[inst -> ne1_len - 1] + '\t' + inst -> ne2_types[inst -> ne2_len - 1];
                    model_id = SearchDeepFctSlot(slot_key);
                    if (model_id >= 0) {
                        FctDeepModel* p_model = deep_fct_list[model_id];
                        RealFctPathInstance* p_inst = p_model -> inst;
                        p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                        slot_key = "FCT_bias";
                        id = p_model -> SearchRealFCTSlot(slot_key);
                        p_inst -> PushFctFea(id, p_inst -> count);
                        p_inst -> count++;
                    }*/
                    slot_key = "in_between_only";
                    AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                        
                    if (fea_params.tag_fea && fea_params.sst) {
                        //slot_key = "in_between_self\t" + inst -> word_types[count];
                        //AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                    }
                    
                    slot_key = "in_between_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                    model_id = SearchDeepFctSlot(slot_key);
                    if (model_id >= 0) {
                        FctDeepModel* p_model = deep_fct_list[model_id];
                        RealFctPathInstance* p_inst = p_model -> inst;
                    p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                    slot_key = "FCT_in_between_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
                    p_inst -> count++;
                    }
                    
                    slot_key = "in_between_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                    model_id = SearchDeepFctSlot(slot_key);
                    if (model_id >= 0) {
                        FctDeepModel* p_model = deep_fct_list[model_id];
                        RealFctPathInstance* p_inst = p_model -> inst;
                        p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                        slot_key = "FCT_in_between_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                        id = p_model -> SearchRealFCTSlot(slot_key);
                        p_inst -> PushFctFea(id, p_inst -> count);
                        p_inst -> count++;
                    }
                        if (fea_params.tag_fea && fea_params.ner) {
                            slot_key = "in_between_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
                            AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                            slot_key = "in_between_ne1_ner\t" + inst -> ne1_nes[inst -> ne1_len - 1];
                            AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                            slot_key = "in_between_ne2_ner\t" + inst -> ne2_nes[inst -> ne2_len - 1];
                            AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                        }
                }
            }
            if (fea_params.dep_fea) {
                if (inst -> dep_paths[count] > 0) {
                    slot_key = "on_path";
                    model_id = SearchDeepFctSlot(slot_key);
                    if (model_id >= 0) {
                    FctDeepModel* p_model = deep_fct_list[model_id];
                    RealFctPathInstance* p_inst = p_model -> inst;
                    p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                    slot_key = "FCT_on_path";
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
//                    slot_key = "FCT_on_path_bias";
//                    id = p_model -> SearchRealFCTSlot(slot_key);
//                    p_inst -> PushFctFea(id, p_inst -> count);
                    slot_key = "FCT_on_path_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
                    slot_key = "FCT_on_path_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
//                    slot_key = "FCT_in_between_nepair\t" + inst -> entitytype;
//                    id = p_model -> SearchRealFCTSlot(slot_key);
//                    p_inst -> PushFctFea(id, p_inst -> count);
                    p_inst -> count++;
                    }
                    
                    slot_key = "on_path_only";
                    AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                        
                    if (fea_params.tag_fea && fea_params.sst) {
                        //slot_key = "on_path_self\t" + inst -> word_types[count];
                        //AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                    }
                    
                    slot_key = "on_path_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                    model_id = SearchDeepFctSlot(slot_key);
                    if (model_id >= 0) {
                    FctDeepModel* p_model = deep_fct_list[model_id];
                    RealFctPathInstance* p_inst = p_model -> inst;
                    p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                    slot_key = "FCT_on_path_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
                    p_inst -> count++;
                    }
                    
                    slot_key = "on_path_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                    model_id = SearchDeepFctSlot(slot_key);
                    if (model_id >= 0) {
                    FctDeepModel* p_model = deep_fct_list[model_id];
                    RealFctPathInstance* p_inst = p_model -> inst;
                    p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                    slot_key = "FCT_on_path_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
                    p_inst -> count++;
                    }
                        if (fea_params.tag_fea && fea_params.ner) {
                            slot_key = "on_path_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
                            AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                            slot_key = "on_path_ne1_ner\t" + inst -> ne1_nes[inst -> ne1_len - 1];
                            AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                            slot_key = "on_path_ne2_ner\t" + inst -> ne2_nes[inst -> ne2_len - 1];
                            AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                        }
                    
                    /*slot_key = "on_path_nepair\t" + inst -> ne1_types[inst -> ne1_len - 1] + '\t' + inst -> ne2_types[inst -> ne2_len - 1];
                    model_id = SearchDeepFctSlot(slot_key);
                    if (model_id >= 0) {
                        FctDeepModel* p_model = deep_fct_list[model_id];
                        RealFctPathInstance* p_inst = p_model -> inst;
                        p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                        slot_key = "FCT_bias";
                        id = p_model -> SearchRealFCTSlot(slot_key);
                        p_inst -> PushFctFea(id, p_inst -> count);
                        p_inst -> count++;
                    }*/
                }
            }
            
            if (fea_params.context) {
                if(count == beg1 - 1){
                    slot_key = "ne1_left";
                    model_id = SearchDeepFctSlot(slot_key);
                    if (model_id >= 0) {
                    FctDeepModel* p_model = deep_fct_list[model_id];
                    RealFctPathInstance* p_inst = p_model -> inst;
                    p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                    
                    slot_key = "FCT_ne1_left";
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
                    
//                    slot_key = "FCT_ne1_left" + inst -> ne1_types[inst -> ne1_len - 1] + "\tne1";
//                    id = p_model -> SearchRealFCTSlot(slot_key);
//                    p_inst -> PushFctFea(id, p_inst -> count);
//                    slot_key = "FCT_ne1_left" + inst -> ne2_types[inst -> ne2_len - 1] + "\tne2";
//                    id = p_model -> SearchRealFCTSlot(slot_key);
//                    p_inst -> PushFctFea(id, p_inst -> count);
                    
                    p_inst -> count++;
                    }
                }
                if(count == beg1 - 2){
                    slot_key = "ne1_left2";
                    model_id = SearchDeepFctSlot(slot_key);
                    if (model_id >= 0) {
                    FctDeepModel* p_model = deep_fct_list[model_id];
                    RealFctPathInstance* p_inst = p_model -> inst;
                    p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                    
                    slot_key = "FCT_ne1_left2";
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
                    p_inst -> count++;
                    }
                }
                if(count == end1 + 1){
                    slot_key = "ne1_right";
                    if (model_id >= 0) {
                    model_id = SearchDeepFctSlot(slot_key);
                    FctDeepModel* p_model = deep_fct_list[model_id];
                    RealFctPathInstance* p_inst = p_model -> inst;
                    p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                    
                    slot_key = "FCT_ne1_right";
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
                    
//                    slot_key = "FCT_ne1_right" + inst -> ne1_types[inst -> ne1_len - 1] + "\tne1";
//                    id = p_model -> SearchRealFCTSlot(slot_key);
//                    p_inst -> PushFctFea(id, p_inst -> count);
//                    slot_key = "FCT_ne1_right" + inst -> ne2_types[inst -> ne2_len - 1] + "\tne2";
//                    id = p_model -> SearchRealFCTSlot(slot_key);
//                    p_inst -> PushFctFea(id, p_inst -> count);
                    
                    p_inst -> count++;
                    }
                }
                if(count == end1 + 2){
                    slot_key = "ne1_right2";
                    model_id = SearchDeepFctSlot(slot_key);
                    FctDeepModel* p_model = deep_fct_list[model_id];
                    RealFctPathInstance* p_inst = p_model -> inst;
                    p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                    
                    slot_key = "FCT_ne1_right2";
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
                    p_inst -> count++;
                }
                if(count == beg2 - 1){
                    slot_key = "ne2_left";
                    model_id = SearchDeepFctSlot(slot_key);
                    FctDeepModel* p_model = deep_fct_list[model_id];
                    RealFctPathInstance* p_inst = p_model -> inst;
                    p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                    
                    slot_key = "FCT_ne2_left";
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
                    
//                    slot_key = "FCT_ne2_left" + inst -> ne2_types[inst -> ne2_len - 1] + "\tne2";
//                    id = p_model -> SearchRealFCTSlot(slot_key);
//                    p_inst -> PushFctFea(id, p_inst -> count);
//                    slot_key = "FCT_ne2_left" + inst -> ne1_types[inst -> ne1_len - 1] + "\tne1";
//                    id = p_model -> SearchRealFCTSlot(slot_key);
//                    p_inst -> PushFctFea(id, p_inst -> count);
                    
                    p_inst -> count++;
                }
                if(count == beg2 - 2){
                    slot_key = "ne2_left2";
                    model_id = SearchDeepFctSlot(slot_key);
                    FctDeepModel* p_model = deep_fct_list[model_id];
                    RealFctPathInstance* p_inst = p_model -> inst;
                    p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                    
                    slot_key = "FCT_ne2_left2";
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
                    p_inst -> count++;
                }
                if(count == end2 + 1){
                    slot_key = "ne2_right";
                    model_id = SearchDeepFctSlot(slot_key);
                    FctDeepModel* p_model = deep_fct_list[model_id];
                    RealFctPathInstance* p_inst = p_model -> inst;
                    p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                    
                    slot_key = "FCT_ne2_right";
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
                    
//                    slot_key = "FCT_ne2_right" + inst -> ne2_types[inst -> ne2_len - 1] + "\tne2";
//                    id = p_model -> SearchRealFCTSlot(slot_key);
//                    p_inst -> PushFctFea(id, p_inst -> count);
//                    slot_key = "FCT_ne2_right" + inst -> ne1_types[inst -> ne1_len - 1] + "\tne1";
//                    id = p_model -> SearchRealFCTSlot(slot_key);
//                    p_inst -> PushFctFea(id, p_inst -> count);
                    
                    p_inst -> count++;                
                }
                if(count == end2 + 2){
                    slot_key = "ne2_right2";
                    model_id = SearchDeepFctSlot(slot_key);
                    FctDeepModel* p_model = deep_fct_list[model_id];
                    RealFctPathInstance* p_inst = p_model -> inst;
                    p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                    
                    slot_key = "FCT_ne2_right2";
                    id = p_model -> SearchRealFCTSlot(slot_key);
                    p_inst -> PushFctFea(id, p_inst -> count);
                    p_inst -> count++;
                }
            }
            
            count++;
        }
        inst -> len = count + 1;
        
        if (fea_params.head) {
            slot_key = "ne1_head";
            model_id = SearchDeepFctSlot(slot_key);
            FctDeepModel* p_model = deep_fct_list[model_id];
            RealFctPathInstance* p_inst = p_model -> inst;
            p_inst -> word_ids[p_inst -> count] = inst -> ne1_ids[inst -> ne1_len - 1];
            slot_key = "FCT_ne1_head";
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            //                slot_key = inst -> entitytype + "\t" + "ne1";
            //                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
            slot_key = "FCT_ne1_head" + inst -> ne1_types[inst -> ne1_len - 1] + "\tne1";
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
//            slot_key = "FCT_ne1_head" + inst -> ne2_types[inst -> ne2_len - 1] + "\tne2";
//            id = p_model -> SearchRealFCTSlot(slot_key);
//            p_inst -> PushFctFea(id, p_inst -> count);
            p_inst -> count++;
            
            slot_key = "ne1_head_pair";
            model_id = SearchDeepFctSlot(slot_key);
            p_model = deep_fct_list[model_id];
            p_inst = p_model -> inst;
            p_inst -> word_ids[p_inst -> count] = inst -> ne1_ids[inst -> ne1_len - 1];
            slot_key = "FCT_ne1_head_pair";
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            slot_key = "FCT_ne1_head_pair\t" + inst -> ne2_types[inst -> ne2_len - 1];
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            p_inst -> count++;
            
            if (fea_params.hyper_emb){
                slot_key = "ne1_hyper";
                model_id = SearchDeepFctSlot(slot_key);
                p_model = deep_fct_list[model_id];
                p_inst = p_model -> inst;
                p_inst -> word_ids[p_inst -> count] = inst -> ne1_type_ids[inst -> ne1_len - 1];
                slot_key = "FCT_ne1_hyper";
                id = p_model -> SearchRealFCTSlot(slot_key);
                p_inst -> PushFctFea(id, p_inst -> count);
                slot_key = "FCT_ne1_hyper_pair\t" + inst -> ne2_types[inst -> ne2_len - 1];
                id = p_model -> SearchRealFCTSlot(slot_key);
                //p_inst -> PushFctFea(id, p_inst -> count);
                slot_key = "FCT_ne1_hyper\t" + inst -> ne1_types[inst -> ne1_len - 1];
                id = p_model -> SearchRealFCTSlot(slot_key);
                //p_inst -> PushFctFea(id, p_inst -> count);
                p_inst -> count++;
            }
            
            slot_key = "ne1_head_only";
            AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], false);
            
            slot_key = "ne1_head\t" + inst -> ne1_types[inst -> ne1_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], false);
            
            slot_key = "ne1_head_pair\t" + inst -> ne2_types[inst -> ne2_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], false);
            
            //slot_key = "ne1_head_nepair\t" + inst -> entitytype;
            //AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], false);
            slot_key = "ne1_head_ner\t" + inst -> ne1_nes[inst -> ne1_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], false);
            
            slot_key = "ne1_head_ner2\t" + inst -> ne2_nes[inst -> ne2_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], false);
            
            slot_key = "ne1_head_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], false);
                    
//            slot_key = "ne1_head_nepair\t" + inst -> ne1_types[inst -> ne1_len - 1] + '\t' + inst -> ne2_types[inst -> ne2_len - 1];
//            model_id = SearchDeepFctSlot(slot_key);
//            if (model_id >= 0) {
//                FctDeepModel* p_model = deep_fct_list[model_id];
//                RealFctPathInstance* p_inst = p_model -> inst;
//                p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
//                slot_key = "FCT_bias";
//                id = p_model -> SearchRealFCTSlot(slot_key);
//                p_inst -> PushFctFea(id, p_inst -> count);
//                p_inst -> count++;
//            }
            
            slot_key = "ne2_head";
            model_id = SearchDeepFctSlot(slot_key);
            p_model = deep_fct_list[model_id];
            p_inst = p_model -> inst;
            p_inst -> word_ids[p_inst -> count] = inst -> ne2_ids[inst -> ne2_len - 1];
            slot_key = "FCT_ne2_head";
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            //                slot_key = inst -> entitytype + "\t" + "ne1";
            //                deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//            slot_key = "FCT_ne2_head" + inst -> ne1_types[inst -> ne1_len - 1] + "\tne1";
//            id = p_model -> SearchRealFCTSlot(slot_key);
//            p_inst -> PushFctFea(id, p_inst -> count);
            slot_key = "FCT_ne2_head" + inst -> ne2_types[inst -> ne2_len - 1] + "\tne2";
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            p_inst -> count++;
            
            slot_key = "ne2_head_pair";
            model_id = SearchDeepFctSlot(slot_key);
            p_model = deep_fct_list[model_id];
            p_inst = p_model -> inst;
            p_inst -> word_ids[p_inst -> count] = inst -> ne2_ids[inst -> ne2_len - 1];
            slot_key = "FCT_ne2_head_pair";
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            slot_key = "FCT_ne2_head_pair\t" + inst -> ne1_types[inst -> ne1_len - 1];
            id = p_model -> SearchRealFCTSlot(slot_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            p_inst -> count++;
            
            if (fea_params.hyper_emb) {
                slot_key = "ne2_hyper";
                model_id = SearchDeepFctSlot(slot_key);
                p_model = deep_fct_list[model_id];
                p_inst = p_model -> inst;
                p_inst -> word_ids[p_inst -> count] = inst -> ne2_type_ids[inst -> ne2_len - 1];
                slot_key = "FCT_ne2_hyper";
                id = p_model -> SearchRealFCTSlot(slot_key);
                p_inst -> PushFctFea(id, p_inst -> count);
                slot_key = "FCT_ne2_hyper_pair\t" + inst -> ne1_types[inst -> ne1_len - 1];
                id = p_model -> SearchRealFCTSlot(slot_key);
                //p_inst -> PushFctFea(id, p_inst -> count);
                slot_key = "FCT_ne2_hyper\t" + inst -> ne2_types[inst -> ne2_len - 1];
                id = p_model -> SearchRealFCTSlot(slot_key);
                //p_inst -> PushFctFea(id, p_inst -> count);
                p_inst -> count++;
            }

            slot_key = "ne2_head_only";
            AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], false);
            
            slot_key = "ne2_head\t" + inst -> ne2_types[inst -> ne2_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], false);
            
            slot_key = "ne2_head_pair\t" + inst -> ne1_types[inst -> ne1_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], false);
            
            //slot_key = "ne2_head_nepair\t" + inst -> entitytype;
            //AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], false);
            slot_key = "ne2_head_ner\t" + inst -> ne2_nes[inst -> ne2_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], false);
            
            slot_key = "ne2_head_ner2\t" + inst -> ne1_nes[inst -> ne1_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], false);
            
            slot_key = "ne2_head_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
            AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], false);

//            slot_key = "ne2_head_nepair\t" + inst -> ne1_types[inst -> ne1_len - 1] + '\t' + inst -> ne2_types[inst -> ne2_len - 1];
//            model_id = SearchDeepFctSlot(slot_key);
//            if (model_id >= 0) {
//                FctDeepModel* p_model = deep_fct_list[model_id];
//                RealFctPathInstance* p_inst = p_model -> inst;
//                p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
//                slot_key = "FCT_bias";
//                id = p_model -> SearchRealFCTSlot(slot_key);
//                p_inst -> PushFctFea(id, p_inst -> count);
//                p_inst -> count++;
//            }
        }
        if (fea_params.dep_path)
        {
            ifs.getline(line_buf, 10000, '\n');
            istringstream iss3(line_buf);
            int path_len;
            string tmp_str;
            int token_pos;
            iss3 >> path_len;
            string path_len_fea;
            if (fea_params.pos_on_path) {
                ostringstream oss;
                if (path_len < 5) oss << "path_len\t" << path_len;
                else oss << "path_len\t5";
                path_len_fea = oss.str();
            }
            for (int i = 0; i < path_len - 1; i++) {
                iss3 >> tmp_str;
                iss3 >> tmp_str;
                iss3 >> tmp_str;
                iss3 >> token_pos;
                
                if (fea_params.pos_on_path) {
//                    ostringstream oss;
//                    oss << "on_path_pos\t" << inst -> entitytype << "\t" << i;
//                    AddCoarseFctModel2List(oss.str(), inst -> word_ids[token_pos], false);
                    
                    AddCoarseFctModel2List(path_len_fea, inst -> word_ids[token_pos], false);
//                    slot_key = path_len_fea + "\t" + inst -> entitytype;
//                    AddCoarseFctModel2List(slot_key, inst -> word_ids[token_pos], false);
                }
            }
        }
        
        if (fea_params.tri_conv) {
            for (int i = 0; i < inst -> len - 1; i++) {
                if (i < max(beg1, beg2) && i > min(end1, end2)) {
                    
                    slot_key = "tri_in_between";
                    if (i - 1 >= 0) trigram_id[0] = inst -> word_ids[i - 1];
                    else trigram_id[0] = -1;
                    trigram_id[1] = inst -> word_ids[i];
                    if (i + 1 < inst -> len - 1) trigram_id[2] = inst -> word_ids[i + 1];
                    else trigram_id[2] = -1;
                    AddConvolutionModel2List(slot_key, trigram_id, false);
                }
            }
        }
        
        if (fea_params.linear) {
            string feat_key = "FEAT_bias";
            SearchFeature(feat_key);
            feat_key = "FEAT_ne1_type\t" + inst -> ne1_types[inst -> ne1_len - 1];
            SearchFeature(feat_key);
            feat_key = "FEAT_ne2_type\t" + inst -> ne2_types[inst -> ne2_len - 1];
            SearchFeature(feat_key);
            feat_key = "FEAT_ne1_word\t" + inst -> ne1_words[inst -> ne1_len - 1];
            SearchFeature(feat_key);
            feat_key = "FEAT_ne2_word\t" + inst -> ne2_words[inst -> ne2_len - 1];
            SearchFeature(feat_key);
            feat_key = "FEAT_ne_type_pair\t" + inst -> ne1_types[inst -> ne1_len - 1] + "\t" + inst -> ne2_types[inst -> ne2_len - 1];
            SearchFeature(feat_key);
            /*feat_key = "FEAT_ne_word_pair\t" + inst -> ne1_words[inst -> ne1_len - 1] + "\t" + inst -> ne2_words[inst -> ne2_len - 1];
            SearchFeature(feat_key);
            feat_key = "FEAT_ne_type_inv_pair\t" + inst -> ne2_types[inst -> ne2_len - 1] + "\t" + inst -> ne1_types[inst -> ne1_len - 1];
            SearchFeature(feat_key);*/
        }
    }
    return 1;
}

//int FullFctModel::AddDeepFctModel2List(string slot_key, string fea_key, bool add) {
//    if (add) {
//        slot_key = "ne2_head";
//        model_id = AddDeepFctModel(slot_key);
//        slot_key = "FCT_ne2_head";
//        deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//    }
//}

int FullFctModel::AddCoarseFctModel2List(string slot_key, int word_id, bool add) {
    int model_id, id;
    string fea_key;
    if (add) {
        model_id = AddDeepFctModel(slot_key);
        fea_key = "FCT_bias";
        deep_fct_list[model_id] -> AddRealFCTSlot(fea_key);
        return model_id;
    }
    else {
        model_id = SearchDeepFctSlot(slot_key);
        if (model_id >= 0) {
            FctDeepModel* p_model = deep_fct_list[model_id];
            RealFctPathInstance* p_inst = p_model -> inst;
            p_inst -> word_ids[p_inst -> count] = word_id;
            fea_key = "FCT_bias";
            id = p_model -> SearchRealFCTSlot(fea_key);
            p_inst -> PushFctFea(id, p_inst -> count);
            p_inst -> count++;
        }
        return model_id;
    }
}

int FullFctModel::AddConvolutionModel2List(string slot_key, vector<int> word_id_vec, bool add) {
    int model_id;
    string fea_key;
    if (add) {
        model_id = AddConvolutionModel(slot_key, (int)word_id_vec.size());
        return model_id;
    }
    else {
        model_id = SearchConvolutionSlot(slot_key);
        if (model_id >= 0) {
            FctConvolutionModel* p_model = convolution_fct_list[model_id];
            FctConvolutionInstance* p_inst = p_model -> inst;
            for (int i = 0; i < word_id_vec.size(); i++) {
                p_inst -> ngram_ids[p_inst -> num_ngram][i] = word_id_vec[i];
            }
            p_inst -> num_ngram++;
        }
        return model_id;
    }
}


string FullFctModel::ProcSenseTag(string input_type) {
    size_t idx = input_type.find_first_of(".");
    string ret = input_type.substr(idx + 1);
    return ret; 
}

string FullFctModel::ProcNeTag(string input_type) {
    size_t idx = input_type.find_first_of(":");
    string ret = input_type.substr(idx + 1);
    //idx = input_type.find_first_of(":");
    idx = ret.find_first_of(":");
    if (idx != -1) {
        ret = ret.substr(0, idx);
    }
    return ret; 
}

string FullFctModel::ToLower(string& s) {
    for (int i = 0; i < s.length(); i++) {
        if (s[i] >= 'A' && s[i] <= 'Z') s[i] += 32;
    }
    return s;
}

int FullFctModel::AddCoarseFctModel(string slot_key) {
//    int id;
//    feat2int::iterator iter = slotdict.find(slot_key);
//    if (iter == slotdict.end()) {
//        id = (int)slotdict.size();
//        slotdict[slot_key] = id;
//        slotlist.push_back(slot_key);
//        cout << slot_key << "\t" << id << endl;
//        return id;
//    }
//    return iter -> second;
    return -1;
}

int FullFctModel::AddDeepFctModel(string slot_key) {
    int id;
    feat2int::iterator iter = slot2deep_model.find(slot_key);
    if (iter == slot2deep_model.end()) {
        id = (int)slot2deep_model.size();
        slot2deep_model[slot_key] = id;
        FctDeepModel* p_deep_model = new FctDeepModel(emb_model_list[0]);
        deep_fct_list.push_back(p_deep_model);
        deep_slot_list.push_back(slot_key);
        cout << slot_key << "\t" << id << endl;
        return id;
    }
    return iter -> second;
}

int FullFctModel::AddConvolutionModel(string slot_key, int length) {
    int id;
    feat2int::iterator iter = slot2convolution_model.find(slot_key);
    if (iter == slot2convolution_model.end()) {
        id = (int)slot2convolution_model.size();
        slot2convolution_model[slot_key] = id;
        FctConvolutionModel* p_conv_model = new FctConvolutionModel(emb_model_list[0], length);
        convolution_fct_list.push_back(p_conv_model);
        convolution_slot_list.push_back(slot_key);
        cout << slot_key << "\t" << id << endl;
        return id;
    }
    return iter -> second;
}

int FullFctModel::SearchCoarseFctSlot(string slot_key) {
    feat2int::iterator iter = slot2coarse_model.find(slot_key);
    if (iter == slot2coarse_model.end()) return -1;
    else return iter -> second;
}

int FullFctModel::SearchDeepFctSlot(string slot_key) {
    feat2int::iterator iter = slot2deep_model.find(slot_key);
    if (iter == slot2deep_model.end()) return -1;
    else return iter -> second;
}

int FullFctModel::SearchConvolutionSlot(string slot_key) {
    feat2int::iterator iter = slot2convolution_model.find(slot_key);
    if (iter == slot2convolution_model.end()) return -1;
    else return iter -> second;
}

int FullFctModel::SearchFeature(string feat_key) {
    feat2int::iterator iter = fea_model -> feat_dict.find(feat_key);
    if (iter == fea_model -> feat_dict.end()) return -1;
    else {
        fea_model -> inst -> AddFea(iter -> second);
        return iter -> second;
    }
}

int FullFctModel::AddFeature(string feat_key) {
    int id;
    //cout << feat_key << endl;
    feat2int::iterator iter = fea_model -> feat_dict.find(feat_key);
    if (iter == fea_model -> feat_dict.end()) {
        id = (int)fea_model -> feat_dict.size();
        fea_model -> feat_dict[feat_key] = id;
        cout << feat_key << "\t" << id << endl;
        return id;
    }
    return iter -> second;
}

void FullFctModel::ForwardProp()
{
    real sum;
    int c;
    for (int i = 0; i < num_labels; i++) {
        inst -> scores[i] = 0.0;
    }
    for (int i = 0; i < coarse_fct_list.size(); i++) {
        coarse_fct_list[i] -> ForwardProp(inst);
    }
    for (int i = 0; i < deep_fct_list.size(); i++) {
        if (deep_fct_list[i] -> inst -> count <= 0) continue;
        deep_fct_list[i] -> ForwardProp(inst);
    }
    for (int i = 0; i < convolution_fct_list.size(); i++) {
        if (convolution_fct_list[i] -> inst -> num_ngram <= 0) continue;
        convolution_fct_list[i] -> ForwardProp(inst);
    }
    if (fea_model != NULL) fea_model -> ForwardProp(inst);
    sum = 0.0;
    for (c = 0; c < num_labels; c++) {
        float tmp;
        if (inst -> scores[c] > MAX_EXP) tmp = exp(MAX_EXP);
        else if (inst -> scores[c] < MIN_EXP) tmp = exp(MIN_EXP);
        else tmp = exp(inst -> scores[c]);
        inst -> scores[c] = tmp;
        sum += inst -> scores[c];
        
    }
    for (c = 0; c < num_labels; c++) {
        inst -> scores[c] /= sum;
    }
}

void FullFctModel::BackProp()
{
    alpha_old = alpha;
//    alpha = alpha * ( 1 - eta * lambda );
    real eta_real = eta / alpha;
    for (int i = 0; i < coarse_fct_list.size(); i++) {
        coarse_fct_list[i] -> BackProp(inst, eta_real);
    }
    for (int i = 0; i < deep_fct_list.size(); i++) {
        if (deep_fct_list[i] -> inst -> count <= 0) continue;
        deep_fct_list[i] -> BackProp(inst, eta_real);
    }
    for (int i = 0; i < convolution_fct_list.size(); i++) {
        if (convolution_fct_list[i] -> inst -> num_ngram <= 0) continue;
        convolution_fct_list[i] -> BackProp(inst, eta_real);
    }
    if (fea_model != NULL) fea_model -> BackProp(inst, eta_real);
}

void FullFctModel::TrainData(string trainfile, string devfile, int type) {
    if (deep_fct_list.size() != 0) {
        printf("L-emb1: %lf\n", deep_fct_list[0] -> label_emb[0]);
        printf("L-emb2: %lf\n", deep_fct_list[0] -> label_emb[layer1_size]);
        printf("L-emb1: %lf\n", deep_fct_list[0] -> fct_fea_emb[0]);
        printf("L-emb2: %lf\n", deep_fct_list[0] -> fct_fea_emb[layer1_size]);
    }
    if(convolution_fct_list.size() != 0) {
        printf("L-emb1: %lf\n", convolution_fct_list[0] -> label_emb[0]);
        printf("L-emb2: %lf\n", convolution_fct_list[0] -> label_emb[layer1_size]);
        printf("L-emb1: %lf\n", convolution_fct_list[0] -> W0[0]);
        printf("L-emb2: %lf\n", convolution_fct_list[0] -> W0[layer1_size]);
    }
    int count = 0;
    int total = num_inst * iter;
    for (int i = 0; i < iter; i++) {
        cout << "Iter " << i << endl;
        cur_iter = i;
        ifstream ifs(trainfile.c_str());
        //int count = 0;
        while (LoadInstance(ifs, type)) {
            ForwardProp();
            BackProp();
            count++;
            
            if (count % 100 == 0) {
                WeightDecay(eta, lambda);
            }
//            if (lambda_prox != 0.0)
//                if (count % 100 == 0) {
//                    real eta_real = eta / alpha;
//                    for (int a = 0; a < num_labels * num_slots * layer1_size; a++) {
//                        if (0 < label_emb[a] && label_emb[a] < eta_real / sqrt(params_g[a]) * lambda_prox) label_emb[a] = 0;
//                        else if (0 > label_emb[a] && label_emb[a] > -1 * eta_real / sqrt(params_g[a]) * lambda_prox) label_emb[a] = 0;
//                        else if (label_emb[a] > 0) label_emb[a] -= eta_real / sqrt(params_g[a]) * lambda_prox;
//                        else label_emb[a] += eta_real / sqrt(params_g[a]) * lambda_prox;
//                    }
//                    if (update_emb) {
//                        real* tmp_p = emb_model -> syn0;
//                        real* tmp_gp = emb_model -> params_g;
//                        for (int a = 0; a < emb_model -> vocab_size * layer1_size; a++) {
//                            if (0 < tmp_p[a] && tmp_p[a] < eta_real / sqrt(tmp_gp[a]) * lambda_prox) tmp_p[a] = 0;
//                            else if (0 > tmp_p[a] && tmp_p[a] > -1 * eta_real / sqrt(tmp_gp[a]) * lambda_prox) tmp_p[a] = 0;
//                            else if (tmp_p[a] > 0) tmp_p[a] -= eta_real / sqrt(tmp_gp[a]) * lambda_prox;
//                            else tmp_p[a] += eta_real / sqrt(tmp_gp[a]) * lambda_prox;
//                        }
//                    }
//                }
//            eta = eta0 * (1 - count / (double)(total + 1));
//            if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        }
        if(!adagrad) eta = eta0 * (1 - count / (double)(total + 1));
        if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        ifs.close();
        //if(i >= 5) for (int a = 0; a < num_labels * num_slots * layer1_size; a++) label_emb[a] = (1 -lambda_prox) * label_emb[a];
        //if(i >= 5) if(update_emb) for (int a = 0; a < emb_model->vocab_size * layer1_size; a++) emb_model -> syn0[a] = (1 - lambda_prox) * emb_model -> syn0[a];
        if (deep_fct_list.size() != 0) {
            printf("L-emb1: %lf\n", deep_fct_list[0] -> label_emb[0]);
            printf("L-emb2: %lf\n", deep_fct_list[0] -> label_emb[layer1_size]);
            printf("L-emb1: %lf\n", deep_fct_list[0] -> fct_fea_emb[0]);
            printf("L-emb2: %lf\n", deep_fct_list[0] -> fct_fea_emb[layer1_size]);
        }
        if(convolution_fct_list.size() != 0) {
            printf("L-emb1: %lf\n", convolution_fct_list[0] -> label_emb[0]);
            printf("L-emb2: %lf\n", convolution_fct_list[0] -> label_emb[layer1_size]);
            printf("L-emb1: %lf\n", convolution_fct_list[0] -> W0[0]);
            printf("L-emb2: %lf\n", convolution_fct_list[0] -> W0[layer1_size]);
        }
        EvalData(trainfile, type);
        EvalData(devfile, type);
    }
    if (deep_fct_list.size() != 0) {
        printf("L-emb1: %lf\n", deep_fct_list[0] -> label_emb[0]);
        printf("L-emb2: %lf\n", deep_fct_list[0] -> label_emb[layer1_size]);
    }
    if(convolution_fct_list.size() != 0) {
        printf("L-emb1: %lf\n", convolution_fct_list[0] -> label_emb[0]);
        printf("L-emb2: %lf\n", convolution_fct_list[0] -> label_emb[layer1_size]);
    }
        
    //ofs.close();
}

void FullFctModel::EvalData(string trainfile, int type) {
    int total = 0;
    int right = 0;
    int positive = 0;
    int tp = 0;
    int pos_pred = 0;
    double max, max_p;
    real prec, rec;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs, type)) {
        if (inst -> label_id == -1) {
            continue;
        }
        //continue;
        total++;
        if (inst -> label_id != 0) positive++;
        ForwardProp();
        max = -1;
        max_p = -1;
        for (int i = 0; i < num_labels; i++){
            if (inst -> scores[i] > max) {
                max = inst -> scores[i];
                max_p = i;
            }
        }
        if (max_p != 0) pos_pred ++;
        if (max_p == inst -> label_id) {
            right++;
            if (inst -> label_id != 0) tp++;
        }
    }
    cout << right << " " << tp << " "  << positive << " " << pos_pred << endl;
    cout << "Acc: " << (float)right / total << endl;
    rec = (real)tp / positive;
    prec = (real)tp / pos_pred;
    cout << "Prec: " << prec << endl;
    cout << "Rec:" << rec << endl;
    real f1 = 2 * prec * rec / (prec + rec);
    cout << "F1:" << f1 << endl;
    cout << std::setprecision(4) << prec * 100 << "\t" << rec * 100 << "\t" << f1 * 100 << endl;
    ifs.close();
}

void FullFctModel::EvalData(string trainfile, string outfile, int type) {
    int total = 0;
    int right = 0;
    int positive = 0;
    int tp = 0;
    int pos_pred = 0;
    int id = 8000;
    double max, max_p;
    real prec, rec;
    ifstream ifs(trainfile.c_str());
    ofstream ofs(outfile.c_str());
    while (LoadInstance(ifs, type)) {
        if (inst -> label_id == -1) {
            continue;
        }
        //continue;
        total++;
        if (inst -> label_id != 0) positive++;
        ForwardProp();
        max = -1;
        max_p = -1;
        for (int i = 0; i < num_labels; i++){
            if (inst -> scores[i] > max) {
                max = inst -> scores[i];
                max_p = i;
            }
        }
        ofs << (id + total) << "\t" << labellist[max_p] << endl;// << "\t" << inst -> scores[max_p] << endl;
        if (max_p != 0) pos_pred ++;
        if (max_p == inst -> label_id) {
            right++;
            if (inst -> label_id != 0) tp++;
        }
    }
    cout << right << " " << tp << " "  << positive << " " << pos_pred << endl;
    cout << "Acc: " << (float)right / total << endl;
    rec = (real)tp / positive;
    prec = (real)tp / pos_pred;
    cout << "Prec: " << prec << endl;
    cout << "Rec:" << rec << endl;
    real f1 = 2 * prec * rec / (prec + rec);
    cout << "F1:" << f1 << endl;
    cout << std::setprecision(4) << prec * 100 << "\t" << rec * 100 << "\t" << f1 * 100 << endl;
    ifs.close();
    ofs.close();
}

void FullFctModel::PrintModelInfo() {
    cout << "Number of Labels: " << num_labels << endl;
    cout << "Number of Instances: " << num_inst << endl;
    cout << "Number of Models: " << num_models << endl;
    if (fea_model != NULL) cout << "Number of Features: " << fea_model -> num_fea << endl;
//    cout << "Number of FCT Slots: " << deep_fct_list[0] -> fct_slotdict.size() << endl;
    cout << "Max length of sentences: " << max_len << endl;
    
    for (int i = 0; i < deep_fct_list.size(); i++) {
//        cout << "Submodel: " << deep_slot_list[i] << endl;
//        deep_fct_list[i] -> PrintModelInfo();
    }
    
    for (int i = 0; i < coarse_fct_list.size(); i++) {
//        cout << "Submodel: " << coarse_slot_list[i] << endl;
    }

}

void FullFctModel::WeightDecay(real eta_real, real lambda) {
    for (int i = 0; i < deep_fct_list.size(); i++) {
        deep_fct_list[i] -> WeightDecay(eta_real, lambda);
    }
}

