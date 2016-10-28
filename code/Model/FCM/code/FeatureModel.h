//
//  FeatureModel.h
//  fct_re_git
//
//  Created by gflfof gflfof on 14-10-14.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef fct_re_git_FeatureModel_h
#define fct_re_git_FeatureModel_h

#include "BaseComponentModel.h"
#include <tr1/unordered_map>
#include "Instances.h"
#include <math.h>
#include <stdlib.h>

#define MAX_EXP 86
#define MIN_EXP -86

class FeatureModel
{
public:
    real alpha;
    real lambda;
    
    unsigned long num_labels;
    real* label_emb;
    real* params_g;
    
    real eta0;
    real eta;
    int iter;
    int cur_iter;
    unsigned long num_fea;
    feat2int feat_dict;
    
    bool adagrad;
    FeatureInstance* inst;
    
    FeatureModel() {
        num_fea = 0;
        inst = new FeatureInstance();
    }
    
    ~FeatureModel(){
        delete inst;
        delete label_emb;
//        delete label_bias;
    }
    
    void ForwardOutputs(BaseInstance* b_inst);
    long BackPropPhrase(BaseInstance* b_inst, real eta_real);
    
    void ForwardProp(BaseInstance* b_inst);
    void BackProp(BaseInstance* b_inst, real eta_real);
    
    void Init();
};


#endif
