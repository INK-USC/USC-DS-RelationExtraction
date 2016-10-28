#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include "hplelib.h"

char data[MAX_STRING], task[MAX_STRING];
char file_path[MAX_STRING], output_path[MAX_STRING], mode = 'j';
int binary = 0, num_threads = 10, vector_size = 50, negative = 5, iters = 100, iter;
long long total_count = 0, total_count_iter = 0, node_count_actual, samples;
int triple_size = 0, margin = 1;
real starting_lr = 0.25, alpha = 0.0001, lr;
double trans_weight = 1.0;
long long tripe_cnt = 0;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;
ransampl_ws* ws;

line_node node_em_M, node_em_F, node_em_Y;
line_link link_em_MF, link_em_MY, link_em_FY;

line_node node_rm_M, node_rm_F, node_rm_Y;
line_link link_rm_MF, link_rm_MY, link_rm_FY;

line_triple trip;

// double weight[7] = {1, 1, 1, 1, 1, 1, 1};
double weight[3] = {1, 1, 1};

double func_rand_num()
{
    return gsl_rng_uniform(gsl_r);
}

void *train_em_thread(void *id)
{
    long long count = 0, last_count = 0;
    unsigned long long next_random = (long long)id;
    int em_node_size = node_em_M.get_num_nodes();
    real *error_vec_u = (real *)calloc(vector_size, sizeof(real));
    real *error_vec_v = (real *)calloc(vector_size, sizeof(real));
    
    while (1)
    {
        //judge for exit
        if (count > samples / num_threads + 2) break;
        
        if (count - last_count > 1000)
        {
            total_count += count - last_count;
            total_count_iter += count - last_count;
            last_count = count;
            printf("%cTransW: %f LR: %f/%f Epoch: %d/%d Progress: %.3lf%%", 13, trans_weight, lr, starting_lr, iter, iters, (real)total_count_iter / (real)(samples + 1) * 100);
            fflush(stdout);
        }
            
        link_em_MY.train_miniBatch_ple(error_vec_u, lr, alpha, next_random, em_node_size * func_rand_num());
        link_em_MF.train_miniBatch_sg_edge_sampling(error_vec_u, error_vec_v, lr, next_random, func_rand_num);
        link_em_FY.train_miniBatch_sg_edge_sampling(error_vec_u, error_vec_v, lr, next_random, func_rand_num);

        count += 1;
    }
    free(error_vec_u);
    free(error_vec_v);
    pthread_exit(NULL);
}


void *train_rm_thread(void *id)
{
    long long count = 0, last_count = 0, trip_cnt = 0;
    unsigned long long next_random = (long long)id;
    int rm_node_size = node_rm_M.get_num_nodes();
    real *error_vec_u = (real *)calloc(vector_size, sizeof(real));
    real *error_vec_v = (real *)calloc(vector_size, sizeof(real));
    
    while (1)
    {
        //judge for exit
        if (count > samples / num_threads + 2) break;
        
        if (count - last_count > 1000)
        {
            total_count += count - last_count;
            total_count_iter += count - last_count;
            last_count = count;
            printf("%cTransW: %f LR: %f/%f Epoch: %d/%d Progress: %.3lf%%", 13, trans_weight, lr, starting_lr, iter, iters, (real)total_count_iter / (real)(samples + 1) * 100);
            fflush(stdout);
        }
        
        link_rm_MY.train_miniBatch_ple(error_vec_u, lr, alpha, next_random, rm_node_size * func_rand_num());
        link_rm_MF.train_miniBatch_sg_edge_sampling(error_vec_u, error_vec_v, lr, next_random, func_rand_num);
        link_rm_FY.train_miniBatch_sg_edge_sampling(error_vec_u, error_vec_v, lr, next_random, func_rand_num);

        trip.train_batch_relation(triple_size * func_rand_num(), lr, margin, func_rand_num);
        trip_cnt += 1;
        
        count += 1;
    }
    tripe_cnt += trip_cnt;
    free(error_vec_u);
    free(error_vec_v);
    pthread_exit(NULL);
}


void *train_miniBatch_thread(void *id)
{
    long long count = 0, last_count = 0, trip_cnt = 0;
    unsigned long long next_random = (long long)id;
    int em_node_size = node_em_M.get_num_nodes();
    int rm_node_size = node_rm_M.get_num_nodes();
    real *error_vec_u = (real *)calloc(vector_size, sizeof(real));
    real *error_vec_v = (real *)calloc(vector_size, sizeof(real));
    
    while (1)
    {
        //judge for exit
        if (count > samples / num_threads + 2) break;
        
        if (count - last_count > 1000)
        {
            total_count += count - last_count;
            total_count_iter += count - last_count;
            last_count = count;
            printf("%cTransW: %f LR: %f/%f Epoch: %d/%d Progress: %.3lf%%", 13, trans_weight, lr, starting_lr, iter, iters, (real)total_count_iter / (real)(samples + 1) * 100);
            fflush(stdout);
        }
        
        int netid = (int)(ransampl_draw(ws, func_rand_num(), func_rand_num()));        
    
        // if (netid == 0) link_em_MY.train_miniBatch_ple(error_vec_u, lr, alpha, next_random, em_node_size * func_rand_num());
        // if (netid == 1) link_em_MF.train_miniBatch_sg_edge_sampling(error_vec_u, error_vec_v, lr, next_random, func_rand_num);
        // if (netid == 2) link_em_FY.train_miniBatch_sg_edge_sampling(error_vec_u, error_vec_v, lr, next_random, func_rand_num);
        // if (netid == 3) link_rm_MY.train_miniBatch_ple(error_vec_u, lr, alpha, next_random, rm_node_size * func_rand_num());
        // if (netid == 4) link_rm_MF.train_miniBatch_sg_edge_sampling(error_vec_u, error_vec_v, lr, next_random, func_rand_num);
        // if (netid == 5) link_rm_FY.train_miniBatch_sg_edge_sampling(error_vec_u, error_vec_v, lr, next_random, func_rand_num);
        // if (netid == 6) {
        //     trip.train_batch(triple_size * func_rand_num(), lr, margin, func_rand_num);
        //     trip_cnt += 1;
        // }

        if (netid == 0) 
        {
            link_em_MY.train_miniBatch_ple(error_vec_u, lr, alpha, next_random, em_node_size * func_rand_num());
            link_em_MF.train_miniBatch_sg_edge_sampling(error_vec_u, error_vec_v, lr, next_random, func_rand_num);
            link_em_FY.train_miniBatch_sg_edge_sampling(error_vec_u, error_vec_v, lr, next_random, func_rand_num);
        }
        if (netid == 1) 
        {
            link_rm_MY.train_miniBatch_ple(error_vec_u, lr, alpha, next_random, rm_node_size * func_rand_num());
            link_rm_MF.train_miniBatch_sg_edge_sampling(error_vec_u, error_vec_v, lr, next_random, func_rand_num);
            link_rm_FY.train_miniBatch_sg_edge_sampling(error_vec_u, error_vec_v, lr, next_random, func_rand_num);
        }
        if (netid == 2) 
        {
            trip.train_batch(triple_size * func_rand_num(), lr, margin, func_rand_num);
            trip_cnt += 1;
        }

        
        count += 1;
    }
    tripe_cnt += trip_cnt;
    free(error_vec_u);
    free(error_vec_v);
    pthread_exit(NULL);
}


void TrainModel() {
    long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    char file_name[MAX_STRING];

    // weight[6] = trans_weight;
    // ws = ransampl_alloc(7);

    weight[2] = trans_weight;
    ws = ransampl_alloc(3);

    ransampl_set(ws, weight);
    printf("Mode: %c\n", mode);
    
    // entity mention nodes
    sprintf(file_name, "%sem/mention.txt", file_path);
    node_em_M.init(file_name, vector_size);
    sprintf(file_name, "%sem/feature.txt", file_path);
    node_em_F.init(file_name, vector_size);
    sprintf(file_name, "%sem/type.txt", file_path);
    node_em_Y.init(file_name, vector_size);
    printf("#EM: %d, #EM feature: %d, #EM type: %d\n", node_em_M.get_num_nodes(), node_em_F.get_num_nodes(), node_em_Y.get_num_nodes());

    // entity mention links
    sprintf(file_name, "%sem/mention_feature.txt", file_path);
    link_em_MF.init(file_name, &node_em_M, &node_em_F, negative); // graph is inversed
    sprintf(file_name, "%sem/mention_type.txt", file_path);
    link_em_MY.init(file_name, &node_em_M, &node_em_Y, negative);
    sprintf(file_name, "%sem/feature_type.txt", file_path);
    link_em_FY.init(file_name, &node_em_F, &node_em_Y, negative);
    printf("MF_em: %d, MY_em: %d, FY_em: %d\n", link_em_MF.get_num_edges(), link_em_MY.get_num_edges(), link_em_FY.get_num_edges());

    // relation mention nodes
    sprintf(file_name, "%srm/mention.txt", file_path);
    node_rm_M.init(file_name, vector_size);
    sprintf(file_name, "%srm/feature.txt", file_path);
    node_rm_F.init(file_name, vector_size);
    sprintf(file_name, "%srm/type.txt", file_path);
    node_rm_Y.init(file_name, vector_size);
    printf("#RM: %d, #RM feature: %d, #RM type: %d\n", node_rm_M.get_num_nodes(), node_rm_F.get_num_nodes(), node_rm_Y.get_num_nodes());

    // relation mention links
    sprintf(file_name, "%srm/mention_feature.txt", file_path);
    link_rm_MF.init(file_name, &node_rm_M, &node_rm_F, negative); // graph is inversed
    sprintf(file_name, "%srm/mention_type.txt", file_path);
    link_rm_MY.init(file_name, &node_rm_M, &node_rm_Y, negative);
    sprintf(file_name, "%srm/feature_type.txt", file_path);
    link_rm_FY.init(file_name, &node_rm_F, &node_rm_Y, negative);
    printf("MF_rm: %d, MY_rm: %d, FY_rm: %d\n", link_rm_MF.get_num_edges(), link_rm_MY.get_num_edges(), link_rm_FY.get_num_edges());

    sprintf(file_name, "%srm/triples.txt", file_path);
    trip.init(file_name, &node_em_M, &node_em_M, &node_rm_M);
    triple_size = trip.get_triple_size();
    printf("#RM triple: %d\n", triple_size);

    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);
    
    clock_t start = clock();
    printf("Training process:\n");
    if (mode == 'p')
    {
        for (iter = 0; iter != iters; iter++)
        {
            lr = starting_lr * (1 - iter / (real)(iters));        
            total_count = iter * samples;
            total_count_iter = 0;        
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, train_em_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
        }
        for (iter = 0; iter != iters; iter++)
        {
            lr = starting_lr * (1 - iter / (real)(iters));        
            total_count = iter * samples;
            total_count_iter = 0;        
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, train_rm_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
        }
    }
    if (mode == 'j')
    {
        for (iter = 0; iter != iters; iter++)
        {
            lr = starting_lr * (1 - iter / (real)(iters));        
            total_count = iter * samples;
            total_count_iter = 0;        
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, train_miniBatch_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
        }
    }

    
    printf("\n");
    printf("%%iters on TransE: %.3lf%%\n", (real)(tripe_cnt) / (real)(iters*samples) * 100);
    clock_t finish = clock();
    printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC / (double)(num_threads));
    
    if(mode == 'j')
    {
        sprintf(file_name, "%sem/emb_retype_mention.txt", output_path);
        node_em_M.output(file_name, binary);
        sprintf(file_name, "%sem/emb_retype_feature.txt", output_path);
        node_em_F.output(file_name, binary);
        sprintf(file_name, "%sem/emb_retype_type.txt", output_path);
        node_em_Y.output(file_name, binary);    
        sprintf(file_name, "%srm/emb_retype_mention.txt", output_path);
        node_rm_M.output(file_name, binary);
        sprintf(file_name, "%srm/emb_retype_feature.txt", output_path);
        node_rm_F.output(file_name, binary);
        sprintf(file_name, "%srm/emb_retype_type.txt", output_path);
        node_rm_Y.output(file_name, binary);
    }
    if(mode == 'p')
    {
        sprintf(file_name, "%sem/emb_retypeTwo_mention.txt", output_path);
        node_em_M.output(file_name, binary);
        sprintf(file_name, "%sem/emb_retypeTwo_feature.txt", output_path);
        node_em_F.output(file_name, binary);
        sprintf(file_name, "%sem/emb_retypeTwo_type.txt", output_path);
        node_em_Y.output(file_name, binary);    
        sprintf(file_name, "%srm/emb_retypeTwo_mention.txt", output_path);
        node_rm_M.output(file_name, binary);
        sprintf(file_name, "%srm/emb_retypeTwo_feature.txt", output_path);
        node_rm_F.output(file_name, binary);
        sprintf(file_name, "%srm/emb_retypeTwo_type.txt", output_path);
        node_rm_Y.output(file_name, binary);
    }
    
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
        printf("ReType-full\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-data <path>\n");
        printf("\t\tData (FIGER / BBN)\n");
        printf("\t-mode <string>\n");
        printf("\t\tTraining mode (m: mini-batch / b: block-wise coordinate descent)\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of embedding; default is 100\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 5 - 10 (0 = not used)\n");
        printf("\t-iters <int>\n");
        printf("\t\tSet the number of iterations as <int>\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the value of weight decay (default 0.0001)\n");
        printf("\t-lr <float>\n");
        printf("\t\tSet the value of learning rate (default 0.025)\n");
        printf("\t-transWeight <float>\n");
        printf("\t\tWeight on TransE term (default 1.0)\n");
        return 0;
    }
    if ((i = ArgPos((char *)"-data", argc, argv)) > 0) strcpy(data, argv[i + 1]);
    if ((i = ArgPos((char *)"-mode", argc, argv)) > 0) mode = argv[i + 1][0];
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) samples = (long long)(atof(argv[i + 1])*1000000);
    if ((i = ArgPos((char *)"-iters", argc, argv)) > 0) iters = atoi(argv[i + 1])*1;
    if ((i = ArgPos((char *)"-lr", argc, argv)) > 0) starting_lr = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-transWeight", argc, argv)) > 0) trans_weight = atof(argv[i + 1]);
    sprintf(file_path, "data/intermediate/%s/", data);
    sprintf(output_path, "data/results/%s/", data);
    lr = starting_lr;
    TrainModel();
    return 0;
}
