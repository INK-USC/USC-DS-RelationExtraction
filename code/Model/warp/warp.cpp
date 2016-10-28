//
//  warp.c
//  Warp
//
//  Created by wenqihe on 1/11/16.
//  Copyright © 2016 wenqihe. All rights reserved.
//


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#define BUFFER_SIZE 512
#define SMALL_BUFFER_SIZE 16
#define LEARNING_RATE 0.01
#define MAX_ITER 50

int count_lines(char *);
double** malloc_matrix_double(int, int);
void free_matrix_double(double**, int);
int* get_negatives(int *, int, int);
double gradient(double **, double **, int*, int*, int*, int, int, int, int);
void print_matrix(char *, double**, int, int);
void print(char*, double**, int, int);
double rank(int);
double dot(double*, double*, int);
void load_data(char*, int**, int*);

double lr;
int max_iter;
int num_threads;
int feature_count;
int type_count;
int mention_count;
int embed_size;
double **A;
double **B;
int** train_x;
int* x_count;
int** train_y;
int* y_count;
int iter;
double error = 0;

void *train_BCD_thread(void *id);

int main(int argc, const char * argv[]) {
    if ( argc != 7 ) {
        /* We print argv[0] assuming it is the program name */
        printf( "usage: %s -INDIR -OUTDIR -EMBED_SIZE -LR -MAX_ITER -N\n", argv[0] );
    }
    else 
    {
        const char *indir = argv[1];
		const char *outdir = argv[2];
        embed_size = atoi(argv[3]);
        lr = atof(argv[4]);
        max_iter = atoi(argv[5]);
        num_threads = atoi(argv[6]);
        char filename[BUFFER_SIZE];
        
        time_t t;
        srand((unsigned) time(&t));
        
        /* Load number of features, types, and mentions */
        snprintf(filename, sizeof(filename), "data/intermediate/%s/em/feature.txt", indir);
        feature_count = count_lines(filename);
        snprintf(filename, sizeof(filename), "data/intermediate/%s/em/type.txt", indir);
        type_count = count_lines(filename);
        snprintf(filename, sizeof(filename), "data/intermediate/%s/em/mention.txt", indir);
        mention_count = count_lines(filename);
        printf("type: %d, feature: %d, mention: %d\n",type_count, feature_count, mention_count);
        
        /* Initialize matrix A and B*/
        A = malloc_matrix_double(feature_count, embed_size);
        B = malloc_matrix_double(type_count, embed_size);
        printf("Fininsh initialize matrix A and B\n");
             
        snprintf(filename, sizeof(filename), "data/intermediate/%s/em/mention_feature.txt", indir);
        train_x = (int **)malloc(mention_count * sizeof(int *));
        x_count = (int *)malloc(mention_count * sizeof(int));
        for(int i = 0; i < mention_count; i++){
            train_x[i] = (int *)malloc(BUFFER_SIZE * sizeof(int));
            if(train_x[i] == NULL){
                printf("out of memory!\n");
                exit(EXIT_FAILURE);
            }
        }
        load_data(filename, train_x, x_count);
        snprintf(filename, sizeof(filename), "data/intermediate/%s/em/mention_type.txt", indir);
        train_y = (int **)malloc(mention_count * sizeof(int *));
        y_count = (int *)malloc(mention_count * sizeof(int));
        for(int i = 0; i < mention_count; i++){
            train_y[i] = (int *)malloc(SMALL_BUFFER_SIZE * sizeof(int));
            if(train_y[i] == NULL){
                printf("out of memory!\n");
                exit(EXIT_FAILURE);
            }
        }
        load_data(filename, train_y, y_count);
        ;
        
        printf("Start training process\n");
        long a;
        pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
				for (iter = 0; iter != MAX_ITER; iter++)
        {
            error = 0;
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, train_BCD_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            printf("Iter:%d, error:%f\n",iter, error);
        }
        /* Save matrix A and B*/
        snprintf(filename, sizeof(filename), "data/results/%s/em/emb_warp_feature.txt", outdir);
        print_matrix(filename, A, feature_count, embed_size);
        snprintf(filename, sizeof(filename), "data/results/%s/em/emb_warp_type.txt", outdir);
        print_matrix(filename, B, type_count, embed_size);
        free_matrix_double(A, feature_count);
        return 0;
    }
}

void *train_BCD_thread(void *id) {
    long long tid = (long long)id;
    int begin = mention_count / num_threads * tid;
    int end = mention_count / num_threads * (tid + 1);
    if (tid == num_threads) end = mention_count;
    double thread_error = 0;
    for (int i = begin; i != end; i++)
    {
        int *MF =train_x[i];
        int *MT = train_y[i];
        int f_number = x_count[i];
        int t_number = y_count[i];
        int *NT =get_negatives(MT, type_count, type_count-t_number);
        thread_error += gradient(A, B, MF, MT, NT, embed_size, f_number, t_number, type_count-t_number);
        free(NT);
    }
    printf("Thread:%lld, Iter:%d, DONE\n", tid, iter);
    error += thread_error;
    pthread_exit(NULL);
}

void print_matrix(char * filename, double** array, int nrows, int ncolumns) {
    FILE *file = fopen( filename, "w" );
    if ( file == NULL ){
        printf( "Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fprintf(file,"%d %d\n", nrows, ncolumns);
    for (int i = 0;i<nrows;i++){
				fprintf(file, "line\t%d\t", i);
        for (int j = 0; j<ncolumns; j++) {
            fprintf(file, "%f ",array[i][j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void load_data(char *filename, int** data, int* count){
    FILE *file = fopen(filename,"r");
    char line[BUFFER_SIZE];
    int mid=0,fid=0,weight=0;
    int i = 0;
    int index = 0;
    int f_number = 0;
    while(fgets(line, sizeof(line), file) != NULL){
        sscanf(line, "%d\t%d\t%d",&mid, &fid, &weight);
        // printf("Read line feature: %d, %d\n", mid, fid);
        if(mid == index){/* Still in the same mention, append feature list */
            data[i][f_number]=fid;
            ++f_number;
        }else{
            count[i] = f_number;
            ++i;
            index = mid;
            data[i][0]=fid;
            f_number = 1;
        }
    }
    count[i] = f_number;
    
    fclose(file);
}

void print(char* message, double** array, int nrows, int ncolumns) {
    printf("%s\n",message);
    for (int i = 0;i<nrows;i++){
        for (int j = 0; j<ncolumns; j++) {
            printf("%f ",array[i][j]);
        }
        printf("\n");
    }
}

int count_lines(char * filename) {
    FILE *file = fopen( filename, "r" );
    if ( file == NULL ){
        printf( "Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    else{
        char line[BUFFER_SIZE];
        int count = 0;
        while(fgets(line, sizeof(line), file) != NULL){
            sscanf(line, "%*s\t%d", &count);
        }
        fclose(file);
        return count+1;
    }
}

double** malloc_matrix_double(int nrows, int ncolumns) {
    double **array;
    array = (double **)malloc(nrows * sizeof(double *));
    int i,j;
    if(array == NULL){
        printf("out of memory!\n");
        exit(EXIT_FAILURE);
    }
    for(i = 0; i < nrows; i++){
        array[i] = (double *)malloc(ncolumns * sizeof(double));
        if(array[i] == NULL){
            printf("out of memory!\n");
            exit(EXIT_FAILURE);
        }
    }
    for (i = 0; i<nrows; ++i) {
        for (j = 0; j<ncolumns;++j){
            array[i][j] =((double) rand() / (RAND_MAX));
        }
    }
    return array;
}

void free_matrix_double(double** array, int nrows) {
    for(int i = 0; i < nrows; i++)
        free(array[i]);
    free(array);
}

int compare (const void * a, const void * b){
    return ( *(int*)a - *(int*)b );
}

int* get_negatives(int* positive_types, int type_count, int nt_number){
    int *negative = (int *)malloc(nt_number*sizeof(int));
    qsort(positive_types, type_count-nt_number, sizeof(int), compare);
    int i = 0;
    int j = 0;
    int p = 0;
    for (; i<type_count; ++i) {
        if (positive_types[p] == i) {
            ++p;
            if (p>=type_count-nt_number) {
                ++i;
                break;
            }
        }else{
            negative[j] = i;
            ++j;
        }
        
    }
    while (i<type_count) {
        negative[j] = i;
        ++j;
        ++i;
    }
    return negative;
}

double gradient(double ** A, double ** B, int* features, int* positive_types, int* negative_types, int embed_size, int f_number, int pt_number, int nt_number){
    double dA[embed_size]; /* 1*h vector */
    double dB[pt_number+nt_number][embed_size]; /* t*h matrix, s */
    double *Ax = (double *)malloc(embed_size*sizeof(double)); /* 1*h vector */
    double error;
    int i,j;
    for (i = 0;i<embed_size; ++i ) {
        dA[i] = 0;
    }
    for (i = 0; i<pt_number+nt_number; ++i) {
        for (j = 0; j<embed_size; ++j) {
            dB[i][j] = 0;
        }
    }
    for (i = 0; i<embed_size; ++i) {
        Ax[i]=0;
    }
    for (i = 0; i<f_number; ++i) {
        int f = features[i];
        for (j = 0;j<embed_size;++j){
            Ax[j] += A[f][j];
        }
    }
    for (i = 0; i<pt_number; ++i) {
        int p_sample = positive_types[i];
        double s1  = dot(Ax, B[p_sample],embed_size);
        int n_sample = -1;
        double s2 = 0;
        int N = 1;
        /* Sample negative example */
        for (; N<=nt_number; ++N) {
            int temp = rand() % nt_number;
            s2 = dot(Ax, B[negative_types[temp]], embed_size);
            if (s1 - s2 <1) {
                n_sample = negative_types[temp];
                break;
            }
        }
        //        printf("Positive : %d, Score1: %f, negative sampling:%d with score %f\n", p_sample,s1, n_sample, s2);
        if (n_sample != -1) {
            double L = rank(nt_number/N);
            error += (1+s2-s1)*L;
            /* Update dA and dB */
            int k;
            for(k = 0;k<embed_size;++k){
                dA[k] += L*(B[p_sample][k]-B[n_sample][k]);
                dB[p_sample][k] += L * Ax[k];
                dB[n_sample][k] -= L * Ax[k];
            }
        }
    }
    
    /* Update A and B , and normalize */
    for (i = 0; i<pt_number+nt_number; ++i) {
        double factor = 0;
        for (j = 0; j<embed_size; ++j) {
            B[i][j] += LEARNING_RATE * dB[i][j];
            factor += B[i][j] * B[i][j];
        }
        if (factor > 1) {
            for (j = 0; j<embed_size; ++j) {
                B[i][j] /= sqrt(factor);
            }
        }
    }
    for (i = 0; i<f_number; ++i) {
        int f = features[i];
        double factor = 0;
        for (j = 0;j<embed_size;++j){
            A[f][j] += LEARNING_RATE * dA[j];
            factor += A[f][j] * A[f][j];
        }
        if (factor > 1) {
            for (j = 0;j<embed_size;++j){
                A[f][j] /= sqrt(factor);
            }
        }
        
    }
    
    free(Ax);
    return error;
}

double rank(int k){
    double loss = 0;
    int i ;
    for (i=1; i<=k; ++i) {
        loss += 1.0/i;
    }
    return loss;
}

double dot(double*Ax , double* Bi, int len){
    double result = 0;
    int i;
    for (i = 0; i<len; ++i) {
        result += Ax[i] * Bi[i];
    }
    return result;
}
