//
//  FctConvolutionModel.cpp
//  fct_re_git
//
//  Created by gflfof gflfof on 14-9-7.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "FctConvolutionModel.h"

void FctConvolutionModel::Init(char* embfile, char* trainfile, int type)
{
}

void FctConvolutionModel::InitModel()
{
    label_emb = (real*) malloc(sizeof(real) * num_labels * layer1_size);
    params_g = (real*) malloc(sizeof(real) * num_labels * layer1_size);
    W0 = (real*)malloc(sizeof(real) * length * layer1_size * layer1_size);
    W0_g = (real*)malloc(sizeof(real) * length * layer1_size * layer1_size);
    
    for (int a = 0; a < length * layer1_size * layer1_size; a++) W0[a] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;//0.0;
    for (int i = 0; i < length; i++) {
        for (int a = 0; a < layer1_size; a++) W0[(a + layer1_size * i) * layer1_size + a] = 1;
    }
    for (int a = 0; a < length * layer1_size * layer1_size; a++) W0_g[a] = 1.0;
    
    for (int a = 0; a < num_labels * layer1_size; a++) label_emb[a] = 0.0;
    for (int a = 0; a < num_labels * layer1_size; a++) params_g[a] = 1.0;
    
    path_emb = (real*) malloc(sizeof(real) * layer1_size);
    part_path_emb = (real*) malloc(sizeof(real) * layer1_size);
}

void FctConvolutionModel::ForwardEmb()
{
    int a, b;
    long long l1, l2, l3;
    //real norm1 = 1.0, norm2 = 1.0;
    word2int::iterator iter;
    for (a = 0; a < layer1_size; a++) path_emb[a] = 0.0;

    for (int i = 0; i < inst -> num_ngram; i++) {
        for (int j = 0; j < length; j++) {
            l1 = j * layer1_size * layer1_size;
            l2 = inst -> ngram_ids[i][j] * layer1_size;
//            for (a = 0; a < layer1_size; a++) path_emb[a] += emb_model->syn0[a + l2] * W0[a + l1];
            for (a = 0; a < layer1_size; a++) {
                l3 = l1 + a * layer1_size;
                for (b = 0; b < layer1_size; b++) {
                    path_emb[a] += emb_model->syn0[b + l2] * W0[b + l3];
                }
            }
        }
    }
    Sigmoid();
}

void FctConvolutionModel::Sigmoid() {
    int a;
    real tmp;
    for (a = 0; a < layer1_size; a++) {
        tmp = exp(-path_emb[a]);
        tmp = 1.0 / (1 + tmp);
        path_emb[a] = tmp;
    }
}

void FctConvolutionModel::SigmoidGradient() {
    int a;
    real tmp;
    for (a = 0; a < layer1_size; a++) {
        tmp = (1 - path_emb[a]) * path_emb[a];
        part_path_emb[a] = part_path_emb[a] * tmp;
    }
}

void FctConvolutionModel::ForwardOutputs(BaseInstance* b_inst)
{
    int a, c;
    long long l1;
    real sum;
    word2int::iterator iter;
    for (c = 0; c < num_labels; c++) {
        sum = 0.0;
        l1 = c * layer1_size;
        for (a = 0; a < layer1_size; a++) sum += path_emb[a] * label_emb[a + l1];
        
        b_inst -> scores[c] += sum;
    }
}

long FctConvolutionModel::BackPropPhrase(BaseInstance* b_inst, real eta_real) {
    int a, c, y;
    long long l1;
    
    for (a = 0; a < layer1_size; a++) part_path_emb[a] = 0.0;
    for (c = 0; c < num_labels; c++) {
        if (b_inst -> label_id == c) y = 1;
        else y = 0;
        
        l1 = c * layer1_size;
        for (a = 0; a < layer1_size; a++) part_path_emb[a] += (y - b_inst->scores[c]) * label_emb[a + l1];
        
        BackPropFea(b_inst, c, y, eta_real);
    }
    
    SigmoidGradient();
    return 0;
}

void FctConvolutionModel::BackPropFea(BaseInstance* b_inst, int c, int y, real eta_real) {
    long a;
    long l1 = c * layer1_size;
    if (!adagrad) for (a = 0; a < layer1_size; a++) label_emb[a + l1] += eta_real * (y - b_inst->scores[c]) * path_emb[a];
    else {
        for (a = 0; a < layer1_size; a++) params_g[a + l1] += (y - b_inst->scores[c]) * path_emb[a] * (y - b_inst->scores[c]) * path_emb[a];
        for (a = 0; a < layer1_size; a++) label_emb[a + l1] += eta_real / sqrt(params_g[a + l1]) * ( (y - b_inst->scores[c]) * path_emb[a] );
    }
}

long FctConvolutionModel::WeightDecay(real eta_real, real lambda) {
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
    
    for (c = 0; c < length; c++) {
        l1 = c * layer1_size;
        if (!adagrad) for (a = 0; a < layer1_size; a++) W0[a + l1] -= eta_real * lambda * W0[a + l1];
        else {
            for (a = 0; a < layer1_size; a++) W0_g[a + l1] += (lambda * W0[a + l1]) * (lambda * W0[a + l1]);
            for (a = 0; a < layer1_size; a++) W0[a + l1] -= eta_real / sqrt(W0_g[a + l1]) * (lambda * W0[a + l1]);
        }
    }
    
    return 0;
}

void FctConvolutionModel::ForwardProp(BaseInstance* b_inst)
{   
    ForwardEmb();
    ForwardOutputs(b_inst);
}

void FctConvolutionModel::BackProp(BaseInstance* b_inst, real eta_real)
{
    int a, b;
    long long l1, l2, l3;
    
    BackPropPhrase(b_inst, eta_real);
    
    for (int i = 0; i < inst -> num_ngram; i++) {
        for (int j = 0; j < length; j++) {
            l1 = j * layer1_size * layer1_size;
            l2 = inst -> ngram_ids[i][j] * layer1_size;
            
            for (a = 0; a < layer1_size; a++) {
                l3 = l1 + a * layer1_size;
                if (!adagrad) for (b = 0; b < layer1_size; b++) W0[b + l3] += eta_real * part_path_emb[a] * emb_model->syn0[b + l2];
                else {
                    for (b = 0; b < layer1_size; b++) W0_g[b + l3] += (part_path_emb[a] * emb_model->syn0[b + l2]) * (part_path_emb[a] * emb_model->syn0[b + l2]);
                    for (b = 0; b < layer1_size; b++) W0[b + l3] += eta_real / sqrt(W0_g[b + l3]) * (part_path_emb[a] * emb_model->syn0[b + l2]);
                }
            }
        }
    }
}

void FctConvolutionModel::PrintModelInfo() {
    cout << "Number of FCT Slots: " << length << endl;
}
