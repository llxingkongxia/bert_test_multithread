#include "src/bert.h"
#include "src/tokenizer.h"
#include "src/model.pb.h"
#include "utils/cpubm_x86.hpp"
#include "utils/table.hpp"
#include "utils/smtl.hpp"
#include <fstream>
#include <iostream>
#include <chrono>

using namespace std;
using namespace lh;

struct Thread_params {
public:
   shared_ptr<Bert<float>> bert = nullptr;
   shared_ptr<FullTokenizer> tokenizer = nullptr;
   size_t pre_batch_size = 100;
   size_t pre_seq_len = 512;
   size_t num_heads = 12;
   size_t embedding_size = 768;
   size_t head_hidden_size = 64;
   size_t intermediate_ratio = 4;
   size_t num_layers = 12;
};

static double get_time(struct timespec *start,
	struct timespec *end)
{
	return end->tv_sec - start->tv_sec +
		(end->tv_nsec - start->tv_nsec) * 1e-9;
}

static void parse_thread_pool(char *sets,
    vector<int> &set_of_threads)
{
    if (sets[0] != '[')
    {
        return;
    }
    int pos = 1;
    int left = 0, right = 0;
    int state = 0;
    while (sets[pos] != ']' && sets[pos] != '\0')
    {
        if (state == 0)
        {
            if (sets[pos] >= '0' && sets[pos] <= '9')
            {
                left *= 10;
                left += (int)(sets[pos] - '0');
            }
            else if (sets[pos] == ',')
            {
                set_of_threads.push_back(left);
                left = 0;
            }
            else if (sets[pos] == '-')
            {
                right = 0;
                state = 1;
            }
        }
        else if (state == 1)
        {
            if (sets[pos] >= '0' && sets[pos] <= '9')
            {
                right *= 10;
                right += (int)(sets[pos] - '0');
            }
            else if (sets[pos] == ',')
            {
                int i;
                for (i = left; i <= right; i++)
                {
                    set_of_threads.push_back(i);
                }
                left = 0;
                state = 0;
            }
        }
        pos++;
    }
    if (sets[pos] != ']')
    {
        return;
    }
    if (state == 0)
    {
        set_of_threads.push_back(left);
    }
    else if (state == 1)
    {
        int i;
        for (i = left; i <= right; i++)
        {
            set_of_threads.push_back(i);
        }
    }
}

static void thread_func(void *params)
{
#if 1
    Thread_params *myparams = (Thread_params *)params;

    vector<string> input_string = {u8"how are you! i am very happy to see you guys, please give me five ok? thanks", u8"this is some jokes, please tell somebody else that reputation to user privacy protection. There is no central authority or supervisor having overall manipulations over others, which makes Bitcoin favored by many. Unlike lling piles of identity information sheets before opening bank accounts, users of Bitcoin need only a pseudonym, a.k.a an address or a hashed public key, to participate the system."};

    vector<vector<string>> input_tokens(2);
    for (int i = 0; i < 2; i++)
    {
        myparams->tokenizer->tokenize(input_string[i].c_str(), &input_tokens[i], 128);
        input_tokens[i].insert(input_tokens[i].begin(), "[CLS]");
        input_tokens[i].push_back("[SEP]");
    }
    uint64_t mask[2];
    for (int i = 0; i < 2; i++)
    {
        mask[i] = input_tokens[i].size();
        for (int j = input_tokens[i].size(); j < 128; j++)
        {
            input_tokens[i].push_back("[PAD]");
        }
    }
    uint64_t input_ids[256];
    uint64_t position_ids[256];
    uint64_t type_ids[256];
    for (int i = 0; i < 2; i++)
    {
        myparams->tokenizer->convert_tokens_to_ids(input_tokens[i], input_ids + i * 128);
        for (int j = 0; j < 128; j++)
        {
            position_ids[i * 128 + j] = j;
            type_ids[i * 128 + j] = 0;
        }
    }

    float out[2 * 128 * myparams->embedding_size];
    float pool_out[2 * myparams->embedding_size];

    auto begin = chrono::system_clock::now();
    for(int i = 0; i < 10; i++) myparams->bert->compute(2, 128, input_ids, position_ids, type_ids, mask, out, pool_out);
    auto end = chrono::system_clock::now();
    
    //cout<<"time: "<<chrono::duration_cast<chrono::milliseconds>(end-begin).count() <<endl;
#endif
}

int main(int argc, char *argv[])
{

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s --thread_pool=[xxx]\n", argv[0]);
        fprintf(stderr, "[xxx] indicates all cores to benchmark.\n");
        fprintf(stderr, "Example: [0,3,5-8,13-15].\n");
        fprintf(stderr, "Notice: there must NOT be any spaces.\n");
        exit(0);
    }

    if (strncmp(argv[1], "--thread_pool=", 14))
    {
        fprintf(stderr, "Error: You must set --thread_pool parameter.\n");
        fprintf(stderr, "Usage: %s --thread_pool=[xxx]\n", argv[0]);
        fprintf(stderr, "[xxx] indicates all cores to benchmark.\n");
        fprintf(stderr, "Example: [0,3,5-8,13-15].\n");
        fprintf(stderr, "Notice: there must NOT be any spaces.\n");
        exit(0);
    }

    vector<int> set_of_threads;
    parse_thread_pool(argv[1] + 14, set_of_threads);

    Thread_params params;
    Model model;
    Graph<float> graph;
    fstream input("/home/liulei/bert_test/model/model.proto", ios::in | ios::binary);
    if (!model.ParseFromIstream(&input))
    {
        //printf("can not read protofile\n");
        return 1;
    }
    for (int i = 0; i < model.param_size(); i++)
    {
        Model_Paramter paramter = model.param(i);
        int size = 1;
        vector<size_t> dims(paramter.n_dim());
        for (int j = 0; j < paramter.n_dim(); j++)
        {
            int dim = paramter.dim(j);
            size *= dim;
            dims[j] = dim;
        }
        float *data = new float[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = paramter.data(i);
        }
        graph[paramter.name()] = make_pair(dims, data);
    }
    google::protobuf::ShutdownProtobufLibrary();
    //cout << "load paramter from protobuf successly!" << endl;

    size_t pre_batch_size = 100;
    size_t pre_seq_len = 512;
    size_t num_heads = 12;
    size_t embedding_size = 768;
    size_t head_hidden_size = 64;
    size_t intermediate_ratio = 4;
    size_t num_layers = 12;
    vector<string> names;
    names.push_back("bert.embeddings.word_embeddings.weight");
    names.push_back("bert.embeddings.position_embeddings.weight");
    names.push_back("bert.embeddings.token_type_embeddings.weight");
    names.push_back("bert.embeddings.LayerNorm.gamma");
    names.push_back("bert.embeddings.LayerNorm.beta");
    for (int idx = 0; idx < num_layers; idx++)
    {
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.query.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.query.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.key.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.key.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.value.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.self.value.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.output.dense.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.output.dense.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.output.LayerNorm.gamma");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".attention.output.LayerNorm.beta");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".intermediate.dense.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".intermediate.dense.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".output.dense.weight");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".output.dense.bias");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".output.LayerNorm.gamma");
        names.push_back("bert.encoder.layer." + to_string(idx) + ".output.LayerNorm.beta");
    }
    names.push_back("bert.pooler.dense.weight");
    names.push_back("bert.pooler.dense.bias");

    params.bert.reset(new Bert<float>(names, graph, pre_batch_size, pre_seq_len, embedding_size, num_heads, head_hidden_size, intermediate_ratio, num_layers));
    params.tokenizer.reset(new FullTokenizer("/home/liulei/bert_test/model/bert-base-uncased-vocab.txt"));
    params.pre_batch_size = pre_batch_size;
    params.pre_seq_len = pre_seq_len;
    params.num_heads = num_heads;
    params.embedding_size = embedding_size;
    params.head_hidden_size = head_hidden_size;
    params.intermediate_ratio = intermediate_ratio;
    params.num_layers = num_layers;

    cout << "init model from pb file and tokenizer successly!" << endl;


    //cpubm_do_bench(set_of_threads, bert.compute);

    int num_threads = set_of_threads.size();
    smtl_handle sh;
    smtl_init(&sh, set_of_threads);

    struct timespec start, end;
    double time_used, perf;

    for (int i = 0; i < num_threads; i++)
    {
        smtl_add_task(sh, thread_func, &params);
    }
    smtl_begin_tasks(sh);
    smtl_wait_tasks_finished(sh);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (int i = 0; i < num_threads; i++)
    {
        smtl_add_task(sh, thread_func, &params);
    }
    smtl_begin_tasks(sh);
    smtl_wait_tasks_finished(sh);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    time_used = get_time(&start, &end);
    printf("time_used=%f\n", time_used);

    smtl_fini(sh);
}
