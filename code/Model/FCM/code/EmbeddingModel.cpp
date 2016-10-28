//
//  EmbeddingModel.cpp
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-30.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include <sstream>
#include <algorithm>
#include "EmbeddingModel.h"

const long long max_w = 500;

//int pairCompare(const void *a, const void *b) {
////    if (secondElem.first - firstElem.first > 0) return 1;
////    else if (secondElem.first - firstElem.first < 0) return -1;
////    else return 0;
//    return ((pair<int, string> *)b)->first - ((pair<int, string> *)a)->first;
//}

bool pairCompare(pair<int, string> a, pair<int, string> b);

bool pairCompare(pair<int, string> a, pair<int, string> b) {
    return a.first > b.first;
}

int EmbeddingModel::InitEmb(char* freqfile, int dim)
{
    long long words, a, b;
    char line_buf[1000];
    string word;
    
    words = 0;
    word2int::iterator iter;
    ifstream ifs(freqfile);
    ifs.getline(line_buf, 1000, '\n');
    while (strcmp(line_buf, "") != 0) {
        words ++;
        ifs.getline(line_buf, 1000, '\n');    
    }
    ifs.close();
    
    layer1_size = dim;
    syn0 = (real *)malloc(words * layer1_size * sizeof(real));
    syn1neg = (real *)malloc(words * layer1_size * sizeof(real));
    
    if (syn0 == NULL || syn1neg == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * layer1_size * sizeof(float) / 1048576);
        return -1;
    }
    
    ifs.open(freqfile);
    for (b = 0; b < words; b++) {
        ifs.getline(line_buf, 1000, '\n');
        istringstream iss(line_buf);
        iss >> word;
        vocabdict[word] = (int)b;
        vocablist.push_back(word);
        for (a = 0; a < layer1_size; a++) syn0[a + layer1_size * b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
        for (a = 0; a < layer1_size; a++) syn1neg[a + layer1_size * b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
    }
    ifs.close();
    return 0;
}

int EmbeddingModel::LoadEmb(string modelname)
{
    //printf("!!!\n");
    long long words, size, a, b;
    float len;
    char ch;
    FILE *f = fopen(modelname.c_str(), "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);
    layer1_size = size;
    words += 42;
    //syn1neg = (float *)malloc(words * size * sizeof(float));
    syn0 = (real *)malloc(words * size * sizeof(real));
    params_g = (real *)malloc(words * size * sizeof(real));
    for (a = 0; a < words * size; a++) params_g[a] = 1.0;
    
    if (syn0 == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(real) / 1048576);
        return -1;
    }
    
    char tmpword[max_w * 2];
    for (b = 0; b < words - 42; b++) {
        fscanf(f, "%s%c", tmpword, &ch);
        if (feof(f)) {
            break;
        }
        //cout << tmpword << endl;
        word2int::iterator iter = vocabdict.find(string(tmpword));
        if (iter == vocabdict.end()) {
            vocabdict[string(tmpword)] = (int)b;
        }
        vocablist.push_back(tmpword);
        if (vocablist.size() != b + 1) {
            cout << "here" << endl;
        }
        float tmp;
        for (a = 0; a < size; a++) {
            fread(&tmp, sizeof(float), 1, f);
            syn0[a + b * size] = tmp;
        }
        //for (a = 0; a < size; a++) fread(&syn1neg[a + b * size], sizeof(real), 1, f);
        len = 0.0;
        for (a = 0; a < size; a++) len += syn0[a + b * size] * syn0[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) syn0[a + b * size] /= len;
    }
    for (b = 0; b < 42; b++) {
        string key = "SST:" + sstags[b];
        word2int::iterator iter = vocabdict.find(key);
        if (iter == vocabdict.end()) {
            vocabdict[key] = (int)(words - 42 + b);
        }
        vocablist.push_back(key);
        //word2int::iterator iter2 = vocabdict.find(sstags[b]);
        //if (iter2 == vocabdict.end()) {
            for (a = 0; a < size; a++) syn0[a + (words - 42 + b) * size] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
        //}
        //else {
        //    for (a = 0; a < size; a++) syn0[a + (words - 42 + b) * size] = syn0[a + iter2 -> second * size];
        //}
    }
    fclose(f);
    //printf("!!!\n");
    //exit(0);
    return 0;
}

int EmbeddingModel::LoadEmbUnnorm(string modelname)
{
    long long words, size, a, b;
    char ch;
    FILE *f = fopen(modelname.c_str(), "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);
    layer1_size = size;
    //syn1neg = (float *)malloc(words * size * sizeof(float));
    syn0 = (real *)malloc(words * size * sizeof(real));
    
    if (syn0 == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(real) / 1048576);
        return -1;
    }
    
    char tmpword[max_w * 2];
    for (b = 0; b < words; b++) {
        fscanf(f, "%s%c", tmpword, &ch);
        if (feof(f)) {
            break;
        }
        //cout << tmpword << endl;
        word2int::iterator iter = vocabdict.find(string(tmpword));
        if (iter == vocabdict.end()) {
            vocabdict[string(tmpword)] = (int)b;
        }
        vocablist.push_back(tmpword);
        float tmp;
        for (a = 0; a < size; a++) {
            fread(&tmp, sizeof(float), 1, f);
            syn0[a + b * size] = tmp;
        }
    }
    fclose(f);
    return 0;
}

void EmbeddingModel::SaveEmb(string modelfile) {
    int b;
    FILE* fileout = fopen(modelfile.c_str(), "wb");
    fprintf(fileout, "%ld %lld\n", vocab_size, layer1_size);
    for (int i = 0; i < vocab_size; i++) {
        fprintf(fileout, "%s ", vocablist[i].c_str());
        //fwrite(&syn1neg[i * layer1_size], sizeof(real), layer1_size, fileout);
        for (b = 0; b < layer1_size; b++) fwrite(&syn0[i * layer1_size + b], sizeof(real), 1, fileout);
        fprintf(fileout, "\n");
    }
    fclose(fileout);
}

void EmbeddingModel::SaveEmbTxt(string modelfile, real alpha) {
    int b;
    FILE* fileout = fopen(modelfile.c_str(), "w");
    for (int i = 0; i < vocab_size; i++) {
        fprintf(fileout, "%s", vocablist[i].c_str());
        for (b = 0; b < layer1_size; b++) fprintf(fileout, "\t%f", alpha * syn0[i * layer1_size + b]);
        fprintf(fileout, "\n");
    }
    fclose(fileout);
}

void EmbeddingModel::SaveLM(string modelfile) {
    int b;
    FILE* fileout = fopen(modelfile.c_str(), "wb");
    fprintf(fileout, "%ld %lld\n", vocab_size, layer1_size);
    for (int i = 0; i < vocab_size; i++) {
        fprintf(fileout, "%s ", vocablist[i].c_str());
        //fwrite(&syn1neg[i * layer1_size], sizeof(real), layer1_size, fileout);
        for (b = 0; b < layer1_size; b++) fwrite(&syn0[i * layer1_size + b], sizeof(real), 1, fileout);
        for (b = 0; b < layer1_size; b++) fwrite(&syn1neg[i * layer1_size + b], sizeof(real), 1, fileout);
        fprintf(fileout, "\n");
    }
    fclose(fileout);
}

void EmbeddingModel::SaveLM(string modelfile, int type) {
    int b;
    FILE* fileout = fopen(modelfile.c_str(), "wb");
    fprintf(fileout, "%ld %lld\n", vocab_size, layer1_size);
    for (int i = 0; i < vocab_size; i++) {
        fprintf(fileout, "%s ", vocablist[i].c_str());
        //fwrite(&syn1neg[i * layer1_size], sizeof(real), layer1_size, fileout);
        for (b = 0; b < layer1_size; b++) fwrite(&syn0[i * layer1_size + b], sizeof(real), 1, fileout);
        //for (b = 0; b < layer1_size; b++) fprintf(fileout, "%f ", syn0[i * layer1_size + b]);
        //if (type == LM) for (b = 0; b < layer1_size; b++) fwrite(&syn1neg[i * layer1_size + b], sizeof(real), 1, fileout);
        fprintf(fileout, "\n");
    }
    
    //if (type == HSLM) 
    for (int i = 0; i < vocab_size - 1; i++) {
        for (b = 0; b < layer1_size; b++) fwrite(&syn1[i * layer1_size + b], sizeof(real), 1, fileout);
        //for (b = 0; b < layer1_size; b++) fprintf(fileout, "%f ", syn1[i * layer1_size + b]);
        fprintf(fileout, "\n");
    }
    fclose(fileout);
}

void EmbeddingModel::InitFromTrainFile(char* train_file) {
    LearnVocabFromTrainFile(train_file);
    InitNet();
}

void EmbeddingModel::LearnVocabFromTrainFile(char* train_file) {
    char word[MAX_STRING];
    //FILE *fin;
    long long i;
    
    ifstream ifs(train_file);
    vocab_size = 0;
    vocabdict.clear();
    word2int::iterator iter;
    
    while (1) {
        //ReadWord(word, fin);
        ifs >> word;
        
        if (ifs.eof()) break;
        
        iter = vocabdict.find(word);
        if (iter == vocabdict.end()) {
            vocabdict[word] = 1;
            vocab_size++;
        } else vocabdict[word]++;
    }
    SortVocab();
    //    FILE* fileout;
    //    fileout = fopen("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt199407.trial.vocab2", "w");
    //    for (i = 0; i < vocab_size; i++) {
    //        fprintf(fileout, "%s\t%d\n", vocablist[i].c_str(), freqlist[i]);
    //    }
    //    fclose(fileout);
    //    vocablist.clear();
    //    freqlist.clear();
    //    vocabdict.clear();
    FILE* filein;
    char tmpstr[1000];
    int freq;
    filein = fopen("/Users/gflfof/Desktop/new work/phrase_embedding/PhraseEmb/test/nyt199407.trial.freq1", "r");
    fscanf(filein, "%d\n", &vocab_size);
    for (i = 0; i < vocab_size; i++) {
        fscanf(filein, "%s\t%d\n", tmpstr, &freq);
        freqlist.push_back(freq);
        vocablist.push_back(tmpstr);
        vocabdict[tmpstr] = (int)i;
    }
    fclose(filein);
    printf("Vocab size: %ld\n", vocab_size);
}

void EmbeddingModel::InitNet() {
    long long a, b;
    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
        syn1neg[a * layer1_size + b] = 0;
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
        syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
}

void EmbeddingModel::InitNetHS() {
    long long a,b;
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
        syn1[a * layer1_size + b] = 0;
}

void EmbeddingModel::SortVocab() {
    int i;
    vector< pair<int, string> > tmp;
    tmp.push_back(make_pair(50000, "</s>"));
    int count = 1;
    for (word2int::iterator iter = vocabdict.begin(); iter != vocabdict.end(); iter++) {
        if (iter -> second >= 5) count ++;
        tmp.push_back(make_pair(iter->second, iter->first));
    }
    cout << count << endl;
    sort(&tmp[1], &tmp[tmp.size()]);
    //    sort(&tmp[1], &tmp[tmp.size() - 1], pairCompare);
    //    qsort(&tmp[1], vocab_size - 1, sizeof(pair<int, string>), pairCompare);
    
    vocabdict.clear();
    vocabdict["</s>"] = 0;
    vocablist.push_back("</s>");
    freqlist.push_back(0);
    vocab_size = 1;
    
    for (i = (int)tmp.size() - 1; i >= 0; i--) {
        if (tmp[i].first < 5) break;
        vocabdict[tmp[i].second] = (int)(vocab_size++);
        vocablist.push_back(tmp[i].second);
        freqlist.push_back(tmp[i].first);
    }
    //    for (int i = 1; i < tmp.size(); i++) {
    //        if (tmp[i].first < 5) break;
    //        vocabdict[tmp[i].second] = vocab_size++;;
    //        vocablist.push_back(tmp[i].second);
    //        freqlist.push_back(tmp[i].first);
    //    }
}

