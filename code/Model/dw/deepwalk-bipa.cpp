#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <pthread.h>
#include <gsl/gsl_rng.h>


#define MAX_STRING 200
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
    double cn;
    int gid;
    char word[MAX_STRING];
};

char data[MAX_STRING], task[MAX_STRING];
char file_path[MAX_STRING], output_path[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, window = 5, num_threads = 1, negative = 5;
int *vocab_hash;
long long vocab_size = 0, layer1_size = 100;
long long feature_size, type_size;
long long fy_size;
long long word_count_actual = 0, word_count_iter = 0, total_word_count = 0;
real alpha = 0.025, starting_alpha;
real *syn0, *syn1, *expTable;
clock_t start;
const int table_size = 1e8;
int *table;

int epoch = 0;
int walk_length = 40, walks_per_vertex = 40;
std::vector<int> vertex_set;
std::vector<int> *neighbor;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

void InitUnigramTable() {
    int a, i;
    double train_words_pow = 0;
    real d1, power = 0.75;
    table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
    i = 0;
    d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
    for (a = 0; a < table_size; a++) {
        table[a] = i;
        if (a / (real)table_size > d1) {
            i++;
            d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
        }
        if (i >= vocab_size) i = vocab_size - 1;
    }
}

// Returns hash value of a word
int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

void LearnVocabFromTrainFile() {
    char file_name[MAX_STRING], str[2 * MAX_STRING + 10000];
    FILE *fin;
    int pst, gid;
    
    feature_size = 0;
    sprintf(file_name, "%sfeature.txt", file_path);
    fin = fopen(file_name, "rb");
    while (fgets(str, sizeof(str), fin))
    {
        int len = strlen(str);
        int k = 0;
        for (k = 0; k != len; k++) if (str[k] == '\t')
            break;
        pst = atoi(str + k + 1);
        if (pst >= feature_size) feature_size = pst + 1;
    }
    fclose(fin);
    
    type_size = 0;
    sprintf(file_name, "%stype.txt", file_path);
    fin = fopen(file_name, "rb");
    while (fgets(str, sizeof(str), fin))
    {
        int len = strlen(str);
        int k = 0;
        for (k = 0; k != len; k++) if (str[k] == '\t')
            break;
        pst = atoi(str + k + 1);
        if (pst >= type_size) type_size = pst + 1;
    }
    fclose(fin);
    
    vocab_size = feature_size + type_size;
    
    vocab = (struct vocab_word *)calloc(vocab_size, sizeof(struct vocab_word));
    
    // read node name and node id
    sprintf(file_name, "%sfeature.txt", file_path);
    fin = fopen(file_name, "rb");
    while (fgets(str, sizeof(str), fin))
    {
        int len = strlen(str);
        int k = 0;
        for (k = 0; k != len; k++) if (str[k] == '\t')
            break;
        str[k] = 0;
        gid = atoi(str + k + 1);
        pst = gid;
        
        unsigned int length = strlen(str) + 1;
        if (length > MAX_STRING) length = MAX_STRING;
        str[MAX_STRING - 1] = 0;
        strcpy(vocab[pst].word, str);
        vocab[pst].cn = 0;
        vocab[pst].gid = gid;
    }
    fclose(fin);
    
    sprintf(file_name, "%stype.txt", file_path);
    fin = fopen(file_name, "rb");
    while (fgets(str, sizeof(str), fin))
    {
        int len = strlen(str);
        int k = 0;
        for (k = 0; k != len; k++) if (str[k] == '\t')
            break;
        str[k] = 0;
        gid = atoi(str + k + 1);
        pst = gid + feature_size;
        
        unsigned int length = strlen(str) + 1;
        if (length > MAX_STRING) length = MAX_STRING;
        str[MAX_STRING - 1] = 0;
        strcpy(vocab[pst].word, str);
        vocab[pst].cn = 0;
        vocab[pst].gid = gid;
    }
    fclose(fin);
    
    printf("Vocab size: %lld\n", vocab_size);
    printf("Feature size: %lld\n", feature_size);
    printf("Type size: %lld\n", type_size);
}

void InitNet() {
    long long a, b;
    
    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn0 == NULL) { printf("Memory allocation failed\n"); exit(1); }
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
        syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
    
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) { printf("Memory allocation failed\n"); exit(1); }
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
        syn1[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
    
    neighbor = new std::vector<int>[vocab_size];
    for (int k = 0; k != vocab_size; k++)
        vertex_set.push_back(k);
}

void ReadNet()
{
    char file_name[MAX_STRING];
    FILE *fi;
    int u, v;
    double wei;
    fy_size = 0;
    
    sprintf(file_name, "%sfeature_type.txt", file_path);
    fi = fopen(file_name, "rb");
    while (1)
    {
        if (fscanf(fi, "%d %d %lf", &u, &v, &wei) != 3) break;
        if (fy_size % 100000 == 0)
        {
            printf("%lldK%c", fy_size / 1000, 13);
            fflush(stdout);
        }
        
        u += 0;
        v += feature_size;
        neighbor[u].push_back(v);
        neighbor[v].push_back(u);
        vocab[u].cn += wei;
        vocab[v].cn += wei;
        
        fy_size++;
    }
    fclose(fi);
    
    printf("Feature-Type: %lld\n", fy_size);
}

void *TrainModelThread(void *id) {
    long long a, c, d, word, last_word, sentence_pst = 0;
    long long word_count = 0, last_word_count = 0;
    long long l1, l2, tid, target, label;
    unsigned long long next_random = (long long)id;
    int *sen = (int *)malloc((walk_length + 5) * sizeof(int));
    int cv, lv, begin, end;
    real f, g;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    
    tid = (long long)id;
    begin = vocab_size / num_threads * tid;
    end = vocab_size / num_threads * (tid + 1);
    if (tid == num_threads - 1) end = vocab_size;
    
    for (int T = begin; T != end; T++)
    {
        if (word_count - last_word_count > 10)
        {
            word_count_actual += word_count - last_word_count;
            word_count_iter += word_count - last_word_count;
            last_word_count = word_count;
            printf("%cEpoch: %d/%d Progress: %.3lf%% Alpha: %f",
                   13, epoch, walks_per_vertex, (real)word_count_iter / (real)(vocab_size + 1) * 100, alpha);
            fflush(stdout);
            alpha = starting_alpha * (1 - word_count_actual / (real)(total_word_count + 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
        
        cv = vertex_set[T];
        for (int k = 0; k != walk_length; k++)
        {
            sen[k] = cv;
            lv = cv;
            a = neighbor[lv].size();
            if (a == 0) cv = vocab_size * gsl_rng_uniform(gsl_r);
            else cv = neighbor[lv][a * gsl_rng_uniform(gsl_r)];
        }
        
        for (sentence_pst = 0; sentence_pst != walk_length; sentence_pst++)
        {
            last_word = sen[sentence_pst];
            l1 = last_word * layer1_size;
            
            for (a = 0; a < window * 2 + 1; a++)
            {
                if (a == window) continue;
                
                c = sentence_pst - window + a;
                if (c < 0) continue;
                if (c >= walk_length) continue;
                word = sen[c];
                
                for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                
                for (d = 0; d < negative + 1; d++)
                {
                    if (d == 0)
                    {
                        target = word;
                        label = 1;
                    }
                    else
                    {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        target = table[(next_random >> 16) % table_size];
                        if (target == 0) target = next_random % (vocab_size - 1) + 1;
                        if (target == word) continue;
                        label = 0;
                    }
                    l2 = target * layer1_size;
                    f = 0;
                    for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
                    if (f > MAX_EXP) g = (label - 1) * alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * alpha;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
                    for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
                }
                
                // Learn weights input -> hidden
                for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
            }
        }
        word_count++;
    }
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void TrainModel() {
    long a;
    FILE *fo;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    starting_alpha = alpha;
    LearnVocabFromTrainFile();
    InitNet();
    ReadNet();
    InitUnigramTable();
    
    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);
    
    total_word_count = walks_per_vertex * vocab_size;
    start = clock();
    printf("Training process:\n");
    for (epoch = 0; epoch != walks_per_vertex; epoch++)
    {
        word_count_iter = 0;
        std::random_shuffle(vertex_set.begin(), vertex_set.end());
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    }
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);
    
    char file_name[MAX_STRING];
    
    sprintf(file_name, "%semb_dw_bipartite_feature.txt", output_path);
    fo = fopen(file_name, "wb");
    if (binary == 1)
    {
        fprintf(fo, "%lld %lld\n", feature_size, layer1_size);
        for (int a = 0; a != feature_size; a++)
        {
            fprintf(fo, "%s ", vocab[a].word);
            for (int b = 0; b != layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
            fprintf(fo, "\n");
        }
    }
    if (binary == 0)
    {
        fprintf(fo, "%lld %lld\n", feature_size, layer1_size);
        for (int k = 0; k != feature_size; k++)
        {
            fprintf(fo, "%s\t%d\t", vocab[k].word, vocab[k].gid);
            for (int c = 0; c != layer1_size; c++) fprintf(fo, "%lf ", syn0[k * layer1_size + c]);
            fprintf(fo, "\n");
        }
    }
    fclose(fo);
    
    sprintf(file_name, "%semb_dw_bipartite_type.txt", output_path);
    fo = fopen(file_name, "wb");
    if (binary == 1)
    {
        fprintf(fo, "%lld %lld\n", type_size, layer1_size);
        for (int a = feature_size; a != feature_size + type_size; a++)
        {
            fprintf(fo, "%s ", vocab[a].word);
            for (int b = 0; b != layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
            fprintf(fo, "\n");
        }
    }
    if (binary == 0)
    {
        fprintf(fo, "%lld %lld\n", type_size, layer1_size);
        for (int k = feature_size; k != feature_size + type_size; k++)
        {
            fprintf(fo, "%s\t%d\t", vocab[k].word, vocab[k].gid);
            for (int c = 0; c != layer1_size; c++) fprintf(fo, "%lf ", syn0[k * layer1_size + c]);
            fprintf(fo, "\n");
        }
    }
    fclose(fo);
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
        
        return 0;
    }
    if ((i = ArgPos((char *)"-data", argc, argv)) > 0) strcpy(data, argv[i + 1]);
    if ((i = ArgPos((char *)"-task", argc, argv)) > 0) strcpy(task, argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iterations", argc, argv)) > 0) walks_per_vertex = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-length", argc, argv)) > 0) walk_length = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    sprintf(file_path, "Intermediate/%s/", data);
    sprintf(output_path, "Results/%s/", data);

    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    TrainModel();
    return 0;
}