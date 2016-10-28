//
//  FctCoarseModel.h
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-30.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef RE_FCT_FctCoarseModel_h
#define RE_FCT_FctCoarseModel_h

#include "BaseComponentModel.h"

class FctCoarseModel: public BaseComponentModel
{
public:
    
    CoarseFctInstance* inst;
    real* part_emb_p;
    
    FctCoarseModel() {inst = new CoarseFctInstance();};
    ~FctCoarseModel() {}
    
    FctCoarseModel(EmbeddingModel* emb_model) {
        this -> emb_model = emb_model;
        layer1_size = emb_model -> layer1_size;
        inst = new CoarseFctInstance();
    }
    
    void ForwardOutputs(BaseInstance* b_inst);
    long BackPropPhrase(BaseInstance* b_inst, real eta_real);
    
    void Init(char* embfile, char* trainfile, int type);
    void InitModel();
    
    void ForwardProp(BaseInstance* b_inst);
    void BackProp(BaseInstance* b_inst, real eta_real);
    void BackPropFea(BaseInstance* b_inst, int word_id, int class_id, int correct, real eta_real);
};


#endif
