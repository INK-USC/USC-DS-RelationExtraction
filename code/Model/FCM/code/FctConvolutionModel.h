//
//  FctConvolutionModel.h
//  fct_re_git
//
//  Created by gflfof gflfof on 14-9-7.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef fct_re_git_FctConvolutionModel_h
#define fct_re_git_FctConvolutionModel_h

#include "BaseComponentModel.h"

class FctConvolutionModel: public BaseComponentModel
{
public:    
    FctConvolutionModel() {inst = new FctConvolutionInstance();}
    ~FctConvolutionModel() {}
    
    FctConvolutionInstance* inst;
    
    feat2int fct_slotdict;
    vector<string> fct_slotlist;
    
    int length;
    
    int hidden_size;
    
    real* W0;
    real* W0_g;
    
    real* bias;
    real* bias_g;
    
    real* path_emb;
    real* part_path_emb;
    
    FctConvolutionModel(EmbeddingModel* emb_model, int length) {
        this -> emb_model = emb_model;
        layer1_size = emb_model -> layer1_size;
        inst = new FctConvolutionInstance();
        this -> length = length;
    }
    
    void ForwardEmb();
    void Sigmoid();
    void SigmoidGradient();
    
    void ForwardOutputs(BaseInstance* b_inst);
    long BackPropPhrase(BaseInstance* b_inst, real eta_real);
    
    void Init(char* embfile, char* trainfile, int type);
    void InitModel();
    
    void ForwardProp(BaseInstance* b_inst);
    void BackProp(BaseInstance* b_inst, real eta_real);
    
    void BackPropFea(BaseInstance* b_inst, int class_id, int correct, real eta_real);
    void PrintModelInfo();
    
    long WeightDecay(real eta_real, real lambda);
};


#endif
