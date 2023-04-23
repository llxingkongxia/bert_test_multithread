#include "table.hpp"
#include "smtl.hpp"
#include "cpubm_x86.hpp"

#include <cstdint>
#include <ctime>
#include <vector>
#include <sstream>
#include <iomanip>
using namespace std;

static double get_time(struct timespec *start,
	struct timespec *end)
{
	return end->tv_sec - start->tv_sec +
		(end->tv_nsec - start->tv_nsec) * 1e-9;
}

typedef struct
{
    std::string isa;
    std::string type;
    std::string dim;
    int64_t loop_time;
    int64_t comp_pl;
    void (*bench)(int64_t);
} cpubm_x86_t;

static vector<cpubm_x86_t> bm_list;

static void thread_func(void *params)
{
    cpubm_x86_t *bm = (cpubm_x86_t*)params;
    bm->bench(bm->loop_time);
}

static void cpubm_x86_one(smtl_handle sh)
{
    struct timespec start, end;
    double time_used, perf;

    int i;
    int num_threads = smtl_num_threads(sh);

	// warm up
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	time_used = get_time(&start, &end);
    
    printf("time=%f\n",time_used);
}

void cpubm_do_bench(std::vector<int> &set_of_threads)
{
    int i;

    int num_threads = set_of_threads.size();

    printf("Number Threads: %d\n", num_threads);
    printf("Thread Pool Binding:");
    for (i = 0; i < num_threads; i++)
    {
        printf(" %d", set_of_threads[i]);
    }
    printf("\n");

    // set thread pool
    smtl_handle sh;
    smtl_init(&sh, set_of_threads);

    // traverse task list
    cpubm_x86_one(sh);

    smtl_fini(sh);
}

