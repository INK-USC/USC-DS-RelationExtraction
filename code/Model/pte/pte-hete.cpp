#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include "linelib.h"

char data[MAX_STRING], task[MAX_STRING];
char file_path[MAX_STRING], output_path[MAX_STRING];
int binary = 0, num_threads = 1, vector_size = 100, negative = 5;
long long samples = 1, edge_count_actual;
real alpha = 0.025, starting_alpha;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;
ransampl_ws* ws;

line_node node_M, node_F, node_Y;
line_link link_MF, link_MY, link_FY;


void *TrainModelThread(void *id) {
    long long edge_count = 0, last_edge_count = 0;
    unsigned long long next_random = (long long)id;
    real *error_vec_u = (real *)calloc(vector_size, sizeof(real));
    real *error_vec_v = (real *)calloc(vector_size, sizeof(real));

    while (1)
    {
        //judge for exit
        if (edge_count > samples / num_threads + 2) break;
        
        if (edge_count - last_edge_count>10000)
        {
            edge_count_actual += edge_count - last_edge_count;
            last_edge_count = edge_count;
            printf("%cAlpha: %f Progress: %.3lf%%", 13, alpha, (real)edge_count_actual / (real)(samples + 1) * 100);
            fflush(stdout);
            alpha = starting_alpha * (1 - edge_count_actual / (real)(samples + 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }

        link_MF.train_sample(error_vec_u, error_vec_v, alpha, gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r), next_random);
        link_FY.train_sample(error_vec_u, error_vec_v, alpha, gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r), next_random);        

        edge_count++;
        edge_count++;

    }
    free(error_vec_u);
    free(error_vec_v);
    pthread_exit(NULL);
}

void TrainModel() {
    long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    starting_alpha = alpha;
    char file_name[MAX_STRING];
    

    sprintf(file_name, "%smention.txt", file_path);
    node_M.init(file_name, vector_size);
    sprintf(file_name, "%sfeature.txt", file_path);
    node_F.init(file_name, vector_size);
    sprintf(file_name, "%stype.txt", file_path);
    node_Y.init(file_name, vector_size);

    // printf("menion-feature + mention-type.\n");    
    // sprintf(file_name, "%smention_feature.txt", file_path);
    // link_MF.init(file_name, &node_M, &node_F, negative);
    // sprintf(file_name, "%smention_type.txt", file_path);
    // link_MY.init(file_name, &node_M, &node_Y, negative);
    
    printf("menion-feature + feature-type.\n");
    sprintf(file_name, "%smention_feature.txt", file_path);
    link_MF.init(file_name, &node_M, &node_F, negative);
    sprintf(file_name, "%sfeature_type.txt", file_path);
    link_FY.init(file_name, &node_F, &node_Y, negative);

    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);
    
    clock_t start = clock();
    printf("Training process:\n");
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);
    
    sprintf(file_name, "%semb_pte_mention.txt", output_path);
    node_M.output(file_name, binary);
    sprintf(file_name, "%semb_pte_feature.txt", output_path);
    node_F.output(file_name, binary);
    sprintf(file_name, "%semb_pte_type.txt", output_path);
    node_Y.output(file_name, binary);
}


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
    int i;
    if (argc == 1) {
        printf("PTE-Heterogeneous\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-data <path>\n");
        printf("\t\tData (FIGER / BBN)\n");
        printf("\t-task <path>\n");
        printf("\t\tTask (reduce_label_noise / typing)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of embedding; default is 100\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 5 - 10 (0 = not used)\n");
        printf("\t-samples <int>\n");
        printf("\t\tSet the number of training samples as <int>Million\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        return 0;
    }
    if ((i = ArgPos((char *)"-data", argc, argv)) > 0) strcpy(data, argv[i + 1]);
    if ((i = ArgPos((char *)"-task", argc, argv)) > 0) strcpy(task, argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) samples = atoi(argv[i + 1])*(long long)(1000000);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    sprintf(file_path, "Intermediate/%s/", data);
    sprintf(output_path, "Results/%s/", data);
    TrainModel();
    return 0;
}