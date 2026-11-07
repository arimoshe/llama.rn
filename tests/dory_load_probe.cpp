// Loads a model through llama_rn_context::loadModel with the settings the
// AIDungeonMaster app uses, so a load that fails in the app can be reproduced
// on a host. Usage:
//   dory_load_probe <model.gguf> [loops] [cache_mode] [n_ctx] [q4kv 0|1] [flash 0|1] [float_loops 0|1]
// float_loops sends the loop count as a float, the way a JavaScript number
// arrives before JSIParams narrows it.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "common.h"
#include "rn-llama.h"

using namespace rnllama;

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [loops] [cache_mode] [n_ctx] [q4kv] [flash] [float_loops]\n", argv[0]);
        return 2;
    }
    const int loops = argc > 2 ? atoi(argv[2]) : 2;
    const std::string mode = argc > 3 ? argv[3] : "last_loop";
    const int n_ctx = argc > 4 ? atoi(argv[4]) : 16384;
    const bool q4kv = argc > 5 ? atoi(argv[5]) != 0 : true;
    const bool flash = argc > 6 ? atoi(argv[6]) != 0 : true;
    const bool float_loops = argc > 7 ? atoi(argv[7]) != 0 : false;

    common_params params;
    params.model.path = argv[1];
    params.n_ctx = n_ctx;
    params.n_parallel = 1;
    params.cpuparams.n_threads = 4;
    params.n_gpu_layers = 0;
    params.ctx_shift = false;
    if (q4kv) {
        params.cache_type_k = GGML_TYPE_Q4_0;
        params.cache_type_v = GGML_TYPE_Q4_0;
    }
    params.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    if (loops > 0) {
        llama_model_kv_override o{};
        strcpy(o.key, "dory.num_loops");
        if (float_loops) {
            o.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
            o.val_f64 = loops;
        } else {
            o.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
            o.val_i64 = loops;
        }
        params.kv_overrides.push_back(o);
        llama_model_kv_override m{};
        strcpy(m.key, "dory.recurrent_kv_cache_mode");
        m.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        strncpy(m.val_str, mode.c_str(), sizeof(m.val_str) - 1);
        params.kv_overrides.push_back(m);
        params.kv_overrides.emplace_back();
        params.kv_overrides.back().key[0] = 0;
    }

    llama_rn_context ctx;
    const bool ok = ctx.loadModel(params);
    printf("load %s (loops=%d%s mode=%s n_ctx=%d kv=%s flash=%d)\n",
        ok ? "OK" : "FAILED", loops, float_loops ? " as float" : "", mode.c_str(), n_ctx,
        q4kv ? "q4_0" : "f16", flash ? 1 : 0);
    return ok ? 0 : 1;
}
