//
//  FullFctModel.h
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-30.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef RE_FCT_FullFctModel_h
#define RE_FCT_FullFctModel_h

#include "Instances.h"
#include "FctCoarseModel.h"
#include "FctDeepModel.h"
#include "FctConvolutionModel.h"
#include "FeatureModel.h"

class FeaParams {
public:
    bool path;
    bool type;
    bool postag;
    bool dep;
    bool ner;
    bool sst;
    bool context;
    bool tag_fea;
    bool dep_fea;
    bool word_on_path;
    bool word_on_path_type;
    bool head;
    bool hyper_emb;
    
    bool dep_path;
    bool pos_on_path;
    
    bool tri_conv;
    bool linear;
    
    void PrintValue() {
        cout << "path:" << path << endl;
        cout << "type:" << type << endl;
        cout << "postag:" << postag << endl;
        cout << "dep:" << dep << endl;
        cout << "context:" << context << endl;
        cout << "dep_fea:" << dep_fea << endl;
        cout << "head:" << head << endl;
        cout << "word_on_path:" << word_on_path << endl;
        cout << "word_on_path_type:" << word_on_path_type << endl;
        cout << "hyper_emb:" << hyper_emb << endl;
        cout << "dep_path:" << dep_path << endl;
        cout << "pos_on_path:" << pos_on_path << endl;
        
        cout << "tri_conv:" << tri_conv << endl; 
    }
};

class FullFctModel
{
public:
    string type;
    bool adagrad;
    bool update_emb;
    
    vector<FctCoarseModel*> coarse_fct_list;
    vector<FctDeepModel*> deep_fct_list;
    vector<FctConvolutionModel*> convolution_fct_list;
    vector<EmbeddingModel*> emb_model_list;
    EmbeddingModel* emb_model;
    
    FeatureModel* fea_model;
    
    int num_models;
    int layer1_size;
    int num_inst;
    int max_len;
    
    feat2int slot2coarse_model;
    vector<string> coarse_slot_list;
    feat2int slot2deep_model;
    vector<string> deep_slot_list;
    
    feat2int slot2convolution_model;
    vector<string> convolution_slot_list;
    
    feat2int labeldict;
    vector<string> labellist;
    
    int num_labels;
    BaseInstance* inst;
    FeaParams fea_params;
    
    real eta0;
    real eta;
    real eta_real;
    real alpha_old;
    real alpha;
    real lambda;
    real lambda_prox;
    
    int iter;
    int cur_iter;
    
    FullFctModel() {inst = new BaseInstance();}
    ~FullFctModel() {}
    
    FullFctModel(char* embfile, char* trainfile) {
        type = "SEM_EVAL";
        inst = new BaseInstance();
        Init(embfile, trainfile);
    }
    
    void BuildModelsFromData(char* trainfile);
    void InitSubmodels();
    
    //void EvalData(string trainfile);
    
    int LoadInstance(ifstream& ifs);
    int LoadInstanceInit(ifstream& ifs);
    int LoadInstance(ifstream& ifs, int type);
    int LoadInstanceSemEval(ifstream& ifs);
    int SearchCoarseFctSlot(string slot_key);
    int AddCoarseFctModel(string slot_key);
    int SearchDeepFctSlot(string slot_key);
    int AddDeepFctModel(string slot_key);
    
    int SearchFeature(string feat_key);
    int AddFeature(string feat_key);
    
    int SearchConvolutionSlot(string slot_key);
    int AddConvolutionModel(string slot_key, int length);
    
//    void AddHeadFctModels();
//    void AddInBetweenFctModels();
//    void AddDepPathFctModels();
//    void AddFctModelsFromFile();
    
//    int AddDeepFctModel2List(string slot_key, string fea_key, bool add);
    
    int AddCoarseFctModel2List(string slot_key, int count, bool add);
    int AddConvolutionModel2List(string slot_key, vector<int> word_id_vec, bool add);
    
    string ProcSenseTag(string input_type);
    string ProcNeTag(string input_type);
    string ToLower(string& s);
    
    void Init(char* embfile, char* traindata);
//    void Init(char* embfile, char* trainfile, int type);
    
    void ForwardProp();
    void BackProp();
    
//    void BackPropFea(int fea_id, int word_id, int class_id, int correct, real eta_real);
//    void BackPropWord(int fea_id, int word_id);
    
    virtual void TrainData(string trainfile, string devfile, int type);
    virtual void EvalData(string trainfile, int type);
    virtual void EvalData(string trainfile, string outfile, int type);
    
    void PrintModelInfo();
    void WeightDecay(real eta_real, real lambda);
};


#endif
