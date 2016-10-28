#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <set>
#include <Eigen/Dense>
#include "ransampl.h"

#define MAX_STRING 1000
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

struct struct_neighbor {
    int index;
    double wei;
};

struct triple
{
    int h, r, t;
    friend bool operator < (triple t1, triple t2)
    {
        if (t1.h == t2.h)
        {
            if (t1.r == t2.r) return t1.t < t2.t;
            return t1.r < t2.r;
        }
        return t1.h < t2.h;
    }
};

class line_link;
class line_triple;

class line_node
{
protected:
    struct struct_node *node;  // store the name of every node
    int node_size, vector_size;  // the number of nodes and the embedding dimension
    char node_file[MAX_STRING];
    real *_vec, *_err;  // the node embedding
    Eigen::Map<BLPMatrix> vec, err;
    int *grad_cnt;
    
public:
    line_node();
    ~line_node();
    
    int get_vector_dim();
    int get_num_nodes();
    real *get_vector();
    struct struct_node *get_node();
    void init(const char *file_name, int vector_dim);
    void output(const char *file_name, int binary);
    void update_vec();
    void update_vec_transe();
    void update_vec_ple(real lr, real alpha); // for partial label learning
    
    friend line_link;
    friend line_triple;
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
    
    std::vector<struct_neighbor> *graph;
    std::set<int> *nb_set;
    
    int get_num_edges();
    void init(const char *file_name, line_node *p_u, line_node *p_v, int negative);
    void train_miniBatch_sg(real *_error_vec_u, real *_error_vec_v, real lr, unsigned long long &rand_index, int u);
    void train_miniBatch_sg_edge_sampling(real *_error_vec_u, real *_error_vec_v, real lr, unsigned long long &rand_index, double (*func_rand_num)());
    void train_miniBatch_ple(real *_error_vec_u, real lr, real alpha, unsigned long long &rand_index, int u);
    void train_BCD_sg(real lr, unsigned long long &rand_index, int u);
    void train_BCD_ple(real lr, real alpha, unsigned long long &rand_index, int u);
};

class line_triple
{
protected:
    line_node *node_h, *node_t, *node_r;  // the two sets of nodes
    long long triple_cnt;  // the number of edges
    int *triple_h, *triple_t, *triple_r;
    char triple_file[MAX_STRING];
    std::set<triple> appear;
    std::vector<triple> *data;
    
    void train_sample(real lr, int h, int t, int r, int nh, int nt, int nr);
    void train_sample_relation(real lr, int h, int t, int r, int nh, int nt, int nr);
    
public:
    line_triple();
    ~line_triple();
    
    void init(const char *file_name, line_node *p_h, line_node *p_t, line_node *p_r);
    void train_bcd(int h, real lr, real margin, double (*func_rand_num)());
    void train_batch(int triple_id, real lr, real margin, double (*func_rand_num)());
    void train_batch_relation(int triple_id, real lr, real margin, double (*func_rand_num)());
    long long get_triple_size();
};

real sigmoid(real x);


