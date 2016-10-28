//
//  FctDeepModel.h
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-30.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef RE_FCT_FctDeepModel_h
#define RE_FCT_FctDeepModel_h

#include "BaseComponentModel.h"

class FctDeepModel: public BaseComponentModel
{
public:    
    FctDeepModel() {inst = new RealFctPathInstance();}
    ~FctDeepModel() {}
    
    RealFctPathInstance* inst;
    
    feat2int fct_slotdict;
    vector<string> fct_slotlist;
    
    real* fct_fea_emb;
    real* fct_params_g;
    
    real* path_emb;
    real* part_path_emb;
    
    FctDeepModel(EmbeddingModel* emb_model) {
        this -> emb_model = emb_model;
        layer1_size = emb_model -> layer1_size;
        inst = new RealFctPathInstance();
//        Init(embfile, trainfile, type);
    }
    
    //void EvalData(string trainfile);
    
//    int LoadInstance(ifstream& ifs, int type);
//    int LoadInstanceRealFCT(ifstream& ifs);
//    int LoadInstanceInit(BaseInstance* b_inst);
    
    int AddRealFCTSlot(string slot_key);
    int SearchRealFCTSlot(string slot_key);
    
    void ForwardEmb();
    void ForwardOutputs(BaseInstance* b_inst);
    long BackPropPhrase(BaseInstance* b_inst, real eta_real);
    
    void Init(char* embfile, char* trainfile, int type);
    void InitModel();
    
    void ForwardProp(BaseInstance* b_inst);
    void BackProp(BaseInstance* b_inst, real eta_real);
    
    void BackPropFea(BaseInstance* b_inst, int class_id, int correct, real eta_real);
//    void BackPropFCTFea(int fea_id, int c, int y, real eta_real);
//    void BackPropEmb();
    void PrintModelInfo();
    
    long WeightDecay(real eta_real, real lambda);
};


#endif
