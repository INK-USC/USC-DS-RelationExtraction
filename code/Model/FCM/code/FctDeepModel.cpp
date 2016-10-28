//
//  FctDeepModel.cpp
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-30.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "FctDeepModel.h"

void FctDeepModel::Init(char* embfile, char* trainfile, int type)
{
//    fea_model -> InitFeatureModel((int)layer1_size, trainfile, type);
//    num_fea = fea_model -> num_fea;
//    num_labels = fea_model -> labeldict.size();
//    num_frames = (int)fea_model -> framedict.size();
//    num_slots = (int)fea_model -> slotdict.size();
//    label_emb = (real*)malloc(sizeof(real) * num_labels * num_slots * layer1_size);
//    params_g = (real*)malloc(sizeof(real) * num_labels * num_slots * layer1_size);
    
//    int num_fct_slots = fea_model -> fct_slotdict.size();
//    if (num_fct_slots > 0) {
//        fct_fea_emb = (real*)malloc(sizeof(real) * fea_model -> fct_slotdict.size() * layer1_size);
//        fct_params_g = (real*)malloc(sizeof(real) * fea_model -> fct_slotdict.size() * layer1_size);
//        for (int a = 0; a < num_fct_slots * layer1_size; a++) fct_fea_emb[a] = 0.0;
//        for (int a = 0; a < layer1_size; a++) fct_fea_emb[a] = 1.0;
//        for (int a = 0; a < num_fct_slots * layer1_size; a++) fct_params_g[a] = 1.0;
//    }
//    for (int a = 0; a < num_labels * num_slots * layer1_size; a++) label_emb[a] = 0.0;// (rand() / (real)RAND_MAX - 0.5) / layer1_size;// 0.0;
//    for (int a = 0; a < num_labels * num_slots * layer1_size; a++) params_g[a] = 1.0;
//    inst -> scores.resize(num_labels);
//    
//    path_emb = (real*) malloc(sizeof(real) * layer1_size);
//    part_path_emb = (real*) malloc(sizeof(real) * layer1_size);
//    
//    max_frame_slots = fea_model -> max_len - 1;
//    if (fea_model -> max_len == 0) max_frame_slots = 1;
//    delete[] emb_p;
//    delete[] part_emb_p;
//    emb_p = new real[layer1_size * max_frame_slots];
//    part_emb_p = new real[layer1_size * max_frame_slots];
//    part_emb_ne1 = new real[layer1_size];
//    part_emb_ne2 = new real[layer1_size];
//    
//    alpha = 1.0;
//    lambda = 0.0;
}

void FctDeepModel::InitModel()
{
    //    num_labels = fea_model -> labeldict.size();
    int num_fct_slots = (int)fct_slotdict.size();
    if (num_fct_slots > 0) {
        label_emb = (real*) malloc(sizeof(real) * num_labels * layer1_size);
        params_g = (real*) malloc(sizeof(real) * num_labels * layer1_size);
        fct_fea_emb = (real*)malloc(sizeof(real) * fct_slotdict.size() * layer1_size);
        fct_params_g = (real*)malloc(sizeof(real) * fct_slotdict.size() * layer1_size);
        for (int a = 0; a < num_fct_slots * layer1_size; a++) fct_fea_emb[a] = 0.0;
        for (int a = 0; a < layer1_size; a++) fct_fea_emb[a] = 1.0;
        for (int a = 0; a < num_fct_slots * layer1_size; a++) fct_params_g[a] = 1.0;
    }
    for (int a = 0; a < num_labels * layer1_size; a++) label_emb[a] = 0.0;// (rand() / (real)RAND_MAX - 0.5) / layer1_size;// 0.0;
    for (int a = 0; a < num_labels * layer1_size; a++) params_g[a] = 1.0;
    //    inst -> scores.resize(num_labels); // full model
    path_emb = (real*) malloc(sizeof(real) * layer1_size);
    part_path_emb = (real*) malloc(sizeof(real) * layer1_size);
}

int FctDeepModel::AddRealFCTSlot(string slot_key) {
    int id;
    feat2int::iterator iter = fct_slotdict.find(slot_key);
    if (iter == fct_slotdict.end()) {
        id = (int)fct_slotdict.size();
        fct_slotdict[slot_key] = id;
        fct_slotlist.push_back(slot_key);
        cout << slot_key << "\t" << id << endl;
        return id;
    }
    return iter -> second;
}

int FctDeepModel::SearchRealFCTSlot(string slot_key) {
    feat2int::iterator iter = fct_slotdict.find(slot_key);
    if (iter == fct_slotdict.end()) return -1;
    else return iter -> second;
}

void FctDeepModel::ForwardEmb()
{
    int a;
    int id;
    long long l1, l2;
    //real norm1 = 1.0, norm2 = 1.0;
    word2int::iterator iter;
    for (a = 0; a < layer1_size; a++) path_emb[a] = 0.0;
    for (int i = 0; i < inst->count; i++) {
        for (int j = 0; j < inst -> fct_nums_fea[i]; j++) {
            id = inst -> fct_fea_ids[i][j];
            if (id != -1 && inst -> word_ids[i] != -1) {
                l1 = id * layer1_size;
                l2 = inst -> word_ids[i] * layer1_size;
                for (a = 0; a < layer1_size; a++) path_emb[a] += emb_model->syn0[a + l2] * fct_fea_emb[a + l1];
            }
        }
    }
}

void FctDeepModel::ForwardOutputs(BaseInstance* b_inst)
{
    int a, c;
    long long l1;
    real sum;
    //real norm1 = 1.0, norm2 = 1.0;
    word2int::iterator iter;
    for (c = 0; c < num_labels; c++) {
        sum = 0.0;
        l1 = c * layer1_size;
        for (a = 0; a < layer1_size; a++) sum += path_emb[a] * label_emb[a + l1];
        
        b_inst -> scores[c] += sum;
    }
}

long FctDeepModel::BackPropPhrase(BaseInstance* b_inst, real eta_real) {
    int a, c, y;
    long long l1;
    
    for (a = 0; a < layer1_size; a++) part_path_emb[a] = 0.0;
    for (c = 0; c < num_labels; c++) {
        if (b_inst -> label_id == c) y = 1;
        else y = 0;
        
        l1 = c * layer1_size;
        for (a = 0; a < layer1_size; a++) part_path_emb[a] += (y - b_inst->scores[c]) * label_emb[a + l1];
//        for (a = 0; a < layer1_size; a++) {
//            if(isnan(part_path_emb[a])) {
//                cerr << part_path_emb[a] << endl;
//            }
//        }

        BackPropFea(b_inst, c, y, eta_real);
    }
    return 0;
}

void FctDeepModel::BackPropFea(BaseInstance* b_inst, int c, int y, real eta_real) {
    long a;
    long l1 = c * layer1_size;
    if (!adagrad) for (a = 0; a < layer1_size; a++) label_emb[a + l1] += eta_real * (y - b_inst->scores[c]) * path_emb[a];
    else {
        for (a = 0; a < layer1_size; a++) params_g[a + l1] += (y - b_inst->scores[c]) * path_emb[a] * (y - b_inst->scores[c]) * path_emb[a];
        for (a = 0; a < layer1_size; a++) label_emb[a + l1] += eta_real / sqrt(params_g[a + l1]) * ( (y - b_inst->scores[c]) * path_emb[a] );
    }
}

long FctDeepModel::WeightDecay(real eta_real, real lambda) {
    int a, c;
    long l1;
    
    for (c = 0; c < num_labels; c++) {
        l1 = c * layer1_size;
        if (!adagrad) for (a = 0; a < layer1_size; a++) label_emb[a + l1] -= eta_real * lambda * label_emb[a + l1];
        else {
            for (a = 0; a < layer1_size; a++) params_g[a + l1] += (lambda * label_emb[a + l1]) * (lambda * label_emb[a + l1]);
            for (a = 0; a < layer1_size; a++) label_emb[a + l1] -= eta_real / sqrt(params_g[a + l1]) * ( lambda * label_emb[a + l1] );
        }
    }
    
    for (c = 1; c < fct_slotdict.size(); c++) {
        l1 = c * layer1_size;
        if (!adagrad) for (a = 0; a < layer1_size; a++) fct_fea_emb[a + l1] -= eta_real * lambda * fct_fea_emb[a + l1];
        else {
            for (a = 0; a < layer1_size; a++) fct_params_g[a + l1] += (lambda * fct_fea_emb[a + l1]) * (lambda * fct_fea_emb[a + l1]);
            for (a = 0; a < layer1_size; a++) fct_fea_emb[a + l1] -= eta_real / sqrt(fct_params_g[a + l1]) * (lambda * fct_fea_emb[a + l1]);
        }
    }
    
    return 0;
}

//todo:
//void RealFCT::BackPropFCTFea(int fea_id, int c, int y, real eta_real) {}

void FctDeepModel::ForwardProp(BaseInstance* b_inst)
{   
    ForwardEmb();
    ForwardOutputs(b_inst);
}

void FctDeepModel::BackProp(BaseInstance* b_inst, real eta_real)
{
    int a;
    long long l1, l2;
    int id;
    
    BackPropPhrase(b_inst, eta_real);
    
    for (int i = 0; i < inst -> count; i++) {
        for (int j = 0; j < inst -> fct_nums_fea[i]; j++) {
            id = inst -> fct_fea_ids[i][j];
            if (id != -1 && inst -> word_ids[i] != -1) {
                l1 = id * layer1_size;
                l2 = inst -> word_ids[i] * layer1_size;
                if (id != 0) {
                    if (!adagrad) for (a = 0; a < layer1_size; a++) fct_fea_emb[a + l1] += eta_real * part_path_emb[a] * emb_model->syn0[a + l2];
                    else {
                        for (a = 0; a < layer1_size; a++) fct_params_g[a + l1] += (part_path_emb[a] * emb_model->syn0[a + l2]) * (part_path_emb[a] * emb_model->syn0[a + l2]);
                        for (a = 0; a < layer1_size; a++) fct_fea_emb[a + l1] += eta_real / sqrt(fct_params_g[a + l1]) * (part_path_emb[a] * emb_model->syn0[a + l2]);
                    }
                }
                if (update_emb) {
                    if (!adagrad) for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l2] += eta_real * part_path_emb[a] * fct_fea_emb[a + l1];
                    else {
                        for (a = 0; a < layer1_size; a++) emb_model->params_g[a + l2] += (part_path_emb[a] * fct_fea_emb[a + l1]) * (part_path_emb[a] * fct_fea_emb[a + l1]);
                        for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l2] += eta_real / sqrt(emb_model->params_g[a + l2]) * (part_path_emb[a] * fct_fea_emb[a + l1]);
                    }
                }
            }
        }
    }
}

void FctDeepModel::PrintModelInfo() {
    cout << "Number of FCT Slots: " << fct_slotdict.size() << endl;
}

