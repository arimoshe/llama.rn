// kv_overrides as the app sends them: through JSIParams' parser, with every
// number a double the way a JavaScript number arrives over JSI. Checks the
// types the parser picks, and, given a model path, that a load applies them.
//   kv_overrides_test [model.gguf]

#include <cstdio>
#include <cstring>

#include "JSIParams.h"
#include "rn-llama.h"

using namespace rnllama_jsi;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

static const llama_model_kv_override * find(const common_params & p, const char * key) {
    for (const auto & o : p.kv_overrides) if (strcmp(o.key, key) == 0) return &o;
    return nullptr;
}

int main(int argc, char ** argv) {
    json params = json::object();
    params["model"] = argc > 1 ? argv[1] : "unused.gguf";
    json kv = json::object();
    kv["dory.num_loops"] = 2.0;                        // a JS 2
    kv["dory.recurrent_kv_cache_mode"] = "last_loop";
    kv["some.scale"] = 0.5;
    kv["some.flag"] = true;
    params["kv_overrides"] = kv;

    common_params p;
    parseCommonParams(params, p);

    const auto * loops = find(p, "dory.num_loops");
    CHECK(loops && loops->tag == LLAMA_KV_OVERRIDE_TYPE_INT && loops->val_i64 == 2);
    const auto * mode = find(p, "dory.recurrent_kv_cache_mode");
    CHECK(mode && mode->tag == LLAMA_KV_OVERRIDE_TYPE_STR && strcmp(mode->val_str, "last_loop") == 0);
    const auto * scale = find(p, "some.scale");
    CHECK(scale && scale->tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT && scale->val_f64 == 0.5);
    const auto * flag = find(p, "some.flag");
    CHECK(flag && flag->tag == LLAMA_KV_OVERRIDE_TYPE_BOOL && flag->val_bool);
    CHECK(!p.kv_overrides.empty() && p.kv_overrides.back().key[0] == 0);

    common_params none;
    json bare = json::object();
    bare["model"] = "unused.gguf";
    parseCommonParams(bare, none);
    CHECK(none.kv_overrides.empty());

    if (argc > 1) {
        p.n_ctx = 2048;
        p.cpuparams.n_threads = 4;
        p.n_gpu_layers = 0;
        rnllama::llama_rn_context ctx;
        const bool ok = ctx.loadModel(p);
        CHECK(ok);
        printf("load %s\n", ok ? "OK" : "FAILED");
    }
    printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
