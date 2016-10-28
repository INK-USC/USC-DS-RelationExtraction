//
//  FeatureModel.cpp
//  fct_re_git
//
//  Created by gflfof gflfof on 14-10-14.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include "FeatureModel.h"

void FeatureModel::Init() {
    cout << feat_dict.size() << endl;
    num_fea = (int)feat_dict.size();
    if (num_fea > 0) {
        label_emb = (real*) malloc(sizeof(real) * num_labels * num_fea);
        params_g = (real*) malloc(sizeof(real) * num_labels * num_fea);
        
        for (unsigned long a = 0; a < num_labels * num_fea; a++) label_emb[a] = 0.0;
        for (unsigned long a = 0; a < num_labels * num_fea; a++) params_g[a] = 1.0;
    }
}

void FeatureModel::ForwardOutputs(BaseInstance* b_inst)
{
    unsigned long c;
    long long l1;
    real sum;
    //real norm1 = 1.0, norm2 = 1.0;
    word2int::iterator iter;
    for (c = 0; c < num_labels; c++) {
        sum = 0.0;
        for (int i = 0; i < inst->num_fea; i++) {
            if (inst -> fea_ids[i] != -1) {
                l1 = c * num_fea + inst -> fea_ids[i];
                sum += label_emb[l1];
            }
        }
        b_inst -> scores[c] += sum;
    }
}

long FeatureModel::BackPropPhrase(BaseInstance* b_inst, real eta_real) {
    int c, y;
    long long l1;
    
    for (c = 0; c < num_labels; c++) {
        if (b_inst -> label_id == c) y = 1;
        else y = 0;
        
        for (int i = 0; i < inst->num_fea; i++) {
            if (inst -> fea_ids[i] != -1) {
                l1 = c * num_fea + inst -> fea_ids[i];
                params_g[l1] += (y - b_inst->scores[c]) * (y - b_inst->scores[c]);
            }
        }
        for (int i = 0; i < inst -> num_fea; i++) {
            if (inst -> fea_ids[i] != -1) {
                l1 = c * num_fea + inst -> fea_ids[i];
                label_emb[l1] += eta_real / sqrt(params_g[l1]) * (y - b_inst->scores[c]);
            }
        }
    }
    return 0;
}

void FeatureModel::ForwardProp(BaseInstance* b_inst)
{
    ForwardOutputs(b_inst);
}

void FeatureModel::BackProp(BaseInstance* b_inst, real eta_real)
{
    BackPropPhrase(b_inst, eta_real);
}
