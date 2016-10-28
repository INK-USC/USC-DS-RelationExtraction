#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <Eigen/Dense>
#include "ransampl.h"

#define MAX_STRING 200
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
const int neg_table_size = 1e8;

typedef float real;

typedef Eigen::Matrix< real, Eigen::Dynamic,
Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign >
BLPMatrix;

typedef Eigen::Matrix< real, 1, Eigen::Dynamic,
Eigen::RowMajor | Eigen::AutoAlign >
BLPVector;

struct struct_node {
    //char *word;
    char word[MAX_STRING];
};

class line_link;

class line_node
{
protected:
    struct struct_node *node;  // store the name of every node
    int node_size, vector_size;  // the number of nodes and the embedding dimension
    char node_file[MAX_STRING];
    real *_vec;  // the node embedding
    Eigen::Map<BLPMatrix> vec;
    
public:
    line_node();
    ~line_node();
    
    int get_vector_dim();
    int get_num_nodes();
    real *get_vector();
    struct struct_node *get_node();
    void init(char *file_name, int vector_dim);
    void output(char *file_name, int binary);
    
    friend line_link;
};

class line_link
{
protected:
    line_node *node_u, *node_v;  // the two sets of nodes
    real *expTable;  // store the value of exp function before training
    ransampl_ws* ws;  // edge sampler
    long long edge_cnt;  // the number of edges
    int neg_samples;  // the number of negative samples
    int *neg_table_u, *neg_table_v;  // negative sampling table
    int *edge_u, *edge_v;
    double *edge_w;
    double *dgr_u, *dgr_v;
    char link_file[MAX_STRING];
public:
    line_link();
    ~line_link();
    
    long long get_edge_cnt();
    void init(char *file_name, line_node *p_u, line_node *p_v, int negative);
    void train_sample(real *_error_vec_u, real *_error_vec_v, real alpha, double rand_num_1, double rand_num_2, unsigned long long &rand_index);
};