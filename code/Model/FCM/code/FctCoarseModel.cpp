//
//  FctCoarseModel.cpp
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-30.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "FctCoarseModel.h"

void FctCoarseModel::Init(char* embfile, char* trainfile, int type)
{}

void FctCoarseModel::InitModel()
{
    for (int a = 0; a < num_labels * layer1_size; a++) label_emb[a] = 0.0;// (rand() / (real)RAND_MAX - 0.5) / layer1_size;// 0.0;
    for (int a = 0; a < num_labels * layer1_size; a++) params_g[a] = 1.0;
}

void FctCoarseModel::ForwardOutputs(BaseInstance* b_inst)
{
    int a, c;
    long long l1, l2;
    real sum;
    //real norm1 = 1.0, norm2 = 1.0;
    word2int::iterator iter;
    for (c = 0; c < num_labels; c++) {
        for (int i = 0; i < inst->count; i++) {
            if (inst -> word_ids[i] != -1) {
                l1 = c * layer1_size;
                l2 = inst -> word_ids[i] * layer1_size;
                for (a = 0; a < layer1_size; a++) sum += emb_model->syn0[a + l2] * label_emb[a + l1];
            }
        }
        b_inst -> scores[c] += sum;
    }
}

long FctCoarseModel::BackPropPhrase(BaseInstance* b_inst, real eta_real) {
    int a, c, y;
    long long l1, l2;
    
    for (a = 0; a < layer1_size * inst -> count; a++) {
        part_emb_p[a] = 0.0;
    }
    for (c = 0; c < num_labels; c++) {
        if (b_inst -> label_id == c) y = 1;
        else y = 0;
        
        for (int i = 0; i < inst->count; i++) {
            if (inst -> word_ids[i] != -1) {
                l1 = c * layer1_size;
                l2 = inst -> word_ids[i] * layer1_size;
                for (a = 0; a < layer1_size; a++) part_emb_p[a + i * layer1_size] += (y - b_inst->scores[c]) * label_emb[a + l1];
            }
        }
        for (int i = 0; i < inst -> count; i++) {
            if (inst -> word_ids[i] != -1) {
                BackPropFea(b_inst, (int)inst -> word_ids[i], c, y, eta_real);
            }
        }
    }
    return 0;
}

void FctCoarseModel::BackPropFea(BaseInstance* b_inst, int word_id, int c, int y, real eta_real) {
    long a;
    long l1 = c * layer1_size;
    long l2 = word_id * layer1_size;
    if (!adagrad) for (a = 0; a < layer1_size; a++) label_emb[a + l1] += eta_real * (y - b_inst->scores[c]) * emb_model->syn0[a + l2];
    else {
        for (a = 0; a < layer1_size; a++) params_g[a + l1] += (y - b_inst->scores[c]) * emb_model->syn0[a + l2] * (y - b_inst->scores[c]) * emb_model->syn0[a + l2];
        for (a = 0; a < layer1_size; a++) label_emb[a + l1] += eta_real / sqrt(params_g[a + l1]) * ( (y - b_inst->scores[c]) * emb_model->syn0[a + l2] );
    }
}

void FctCoarseModel::ForwardProp(BaseInstance* b_inst)
{    
    ForwardOutputs(b_inst);
}

void FctCoarseModel::BackProp(BaseInstance* b_inst, real eta_real)
{
    int a;
    long long l1, l2;

    BackPropPhrase(b_inst, eta_real);
    
    if (update_emb) {
        for (int i = 0; i < inst -> count; i++) {
            if (inst -> word_ids[i] >= 0) {
                l1 = inst -> word_ids[i] * layer1_size;
                l2 = layer1_size * i;
                if (!adagrad) for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta_real * part_emb_p[a + l2];
                else {
                    for (a = 0; a < layer1_size; a++) emb_model->params_g[a + l1] += part_emb_p[a + l2] * part_emb_p[a + l2];
                    for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta_real / sqrt(emb_model->params_g[a + l1]) * part_emb_p[a + l2];
                }
            }
        }
    }
}
