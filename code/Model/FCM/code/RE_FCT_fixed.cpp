//
//  RE_FCT.cpp
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-31.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sstream>
#include <limits>
#include "FullFctModel.h"

#define SEM_EVAL 1
#define ACE_2005 2

char train_file[MAX_STRING], dev_file[MAX_STRING], res_file[MAX_STRING];
char output_file[MAX_STRING], param_file[MAX_STRING];
char clus_file[MAX_STRING], baseemb_file[MAX_STRING], freq_file[MAX_STRING];
char model_file[MAX_STRING];
char feature_file[MAX_STRING];
char trainsub_file[MAX_STRING];
int iter = 1;
int finetuning = 1;
real alpha = 0.01;

int ArgPos(char *str, int argc, char **argv);

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    //int i;
    output_file[0] = 0;
    string dir = "/Users/gflfof/Desktop/new work/path embedding/2014summer/data";
    if (true) {
        strcpy(train_file, argv[1]);
        strcpy(dev_file, argv[2]);
        strcpy(res_file, argv[3]);
        strcpy(baseemb_file, argv[4]);
        
        FullFctModel* plearner = new FullFctModel(baseemb_file, train_file);
        plearner -> adagrad = true;
        plearner -> update_emb = false;
        plearner -> InitSubmodels();
        plearner -> PrintModelInfo();
        
        plearner -> iter = atoi(argv[5]);
        plearner -> eta = plearner -> eta0 = atof(argv[6]);
        plearner -> lambda = 0;
        plearner -> lambda_prox = 0;
        
        //        plearner -> EvalData(dev_file, SEMEVAL_INST);
        plearner -> TrainData(train_file, dev_file, SEM_EVAL);
        //plearner -> TrainData(train_file, train_file);
        
        //        plearner -> EvalData(train_file, REALFCT_INST);
        plearner -> EvalData(dev_file, res_file, SEM_EVAL);
        
        cout << "end" << endl;
        return 0;
    }
}
