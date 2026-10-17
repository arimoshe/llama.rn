// How well can a mixture-of-experts model's routing be guessed ahead of time?
// Runs prompts through the model, records each MoE layer's router input and
// real choice for every generated token, and scores guesses for layer L+k made
// from layer L's router input (rescaled by the two layers' ffn_norm weights,
// then run through layer L+k's router). Also simulates the expert cache (LRU
// at a byte budget) to show how many of its misses a prefetch would cover and
// how many extra reads it would cost.
//
// Built against vendor/llama.cpp on its own (not part of the test CMake):
//   cmake -S vendor/llama.cpp -B build-probe -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_APP=OFF \
//         -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
//   ninja -C build-probe llama
//   g++ -O2 -std=c++17 tests/moe_prefetch_probe.cpp -Ivendor/llama.cpp/include -Ivendor/llama.cpp/ggml/include \
//       build-probe/src/libllama.a build-probe/ggml/src/libggml{,-cpu,-base}.a -fopenmp -lpthread -ldl
//
// LFM2.5-24B-A2B on the app's benchmark prompts (Oct 2026): 1 layer ahead,
// top 4 caught 83-86% of the real experts (top 6: 92-95%; the previous
// token's: 26-31%); at a 2.9 GB cache, waited-for reads per token fell from
// ~65 to ~14 for ~20% more bytes read.
//
// usage: prefetch_probe model.gguf budget_mb prompt.txt [prompt.txt...]
//   each prompt file: system text, a line "---USER---", user text, a line
//   "---N---" and the number of tokens to generate.
#include "llama.h"
#include "gguf.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static int n_layer, n_expert, n_used, dense_lead, gating, n_embd;
static std::vector<std::vector<float>> router;   // [layer] n_expert x n_embd
static std::vector<std::vector<float>> normw;    // [layer] n_embd
static std::vector<std::vector<float>> bias;     // [layer] n_expert
static double expert_bytes;                      // gate+up+down of one expert in one layer

static std::vector<float> read_tensor(FILE * f, gguf_context * g, const char * name, int64_t * ne0 = nullptr) {
    int64_t id = gguf_find_tensor(g, name);
    if (id < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    ggml_type type = gguf_get_tensor_type(g, id);
    size_t size = gguf_get_tensor_size(g, id);
    size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
    std::vector<uint8_t> raw(size);
    fseeko(f, off, SEEK_SET);
    if (fread(raw.data(), 1, size, f) != size) { fprintf(stderr, "short read %s\n", name); exit(1); }
    int64_t n = (int64_t) (size / ggml_type_size(type) * ggml_blck_size(type));
    std::vector<float> out(n);
    if (type == GGML_TYPE_F32) memcpy(out.data(), raw.data(), n * 4);
    else ggml_get_type_traits(type)->to_float(raw.data(), out.data(), n);
    if (ne0) *ne0 = n;
    return out;
}

static void load_router(const char * path) {
    gguf_init_params p = { /*no_alloc*/ true, nullptr };
    gguf_context * g = gguf_init_from_file(path, p);
    std::string arch = gguf_get_val_str(g, gguf_find_key(g, "general.architecture"));
    auto u32 = [&](const std::string & k, int def) {
        int64_t id = gguf_find_key(g, (arch + "." + k).c_str());
        return id < 0 ? def : (int) gguf_get_val_u32(g, id);
    };
    n_layer = u32("block_count", 0);
    n_expert = u32("expert_count", 0);
    n_used = u32("expert_used_count", 0);
    dense_lead = u32("leading_dense_block_count", 0);
    gating = u32("expert_gating_func", 1);
    n_embd = u32("embedding_length", 0);
    fprintf(stderr, "%s: %d layers (%d dense first), %d experts, %d used, gating %d, n_embd %d\n",
            arch.c_str(), n_layer, dense_lead, n_expert, n_used, gating, n_embd);
    FILE * f = fopen(path, "rb");
    router.resize(n_layer); normw.resize(n_layer); bias.resize(n_layer);
    double exp_total = 0;
    char name[128];
    for (int il = 0; il < n_layer; il++) {
        snprintf(name, sizeof name, "blk.%d.ffn_norm.weight", il);
        normw[il] = read_tensor(f, g, name);
        if (il < dense_lead) continue;
        snprintf(name, sizeof name, "blk.%d.ffn_gate_inp.weight", il);
        router[il] = read_tensor(f, g, name);
        snprintf(name, sizeof name, "blk.%d.exp_probs_b.bias", il);
        if (gguf_find_tensor(g, name) >= 0) bias[il] = read_tensor(f, g, name);
        else bias[il].assign(n_expert, 0.0f);
        for (const char * t : {"gate", "up", "down"}) {
            snprintf(name, sizeof name, "blk.%d.ffn_%s_exps.weight", il, t);
            exp_total += gguf_get_tensor_size(g, gguf_find_tensor(g, name));
        }
    }
    fclose(f);
    expert_bytes = exp_total / ((n_layer - dense_lead) * (double) n_expert);
    fprintf(stderr, "experts: %.2f GB in all, %.2f MB per expert per layer (gate+up+down)\n",
            exp_total / 1e9, expert_bytes / 1e6);
    gguf_free(g);
}

// selection scores from logits, as build_moe_ffn picks them
static void scores(int il, const float * logits, std::vector<float> & s) {
    s.assign(logits, logits + n_expert);
    if (gating == 2) {
        for (auto & v : s) v = 1.0f / (1.0f + expf(-v));
    } else if (gating == 1) {
        float m = *std::max_element(s.begin(), s.end()), sum = 0;
        for (auto & v : s) { v = expf(v - m); sum += v; }
        for (auto & v : s) v /= sum;
    }
    for (int e = 0; e < n_expert; e++) s[e] += bias[il][e];
}

static std::vector<int> topk(const std::vector<float> & s, int k) {
    std::vector<int> idx(s.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return s[a] > s[b]; });
    idx.resize(k);
    return idx;
}

// guess of layer `to`'s ranking from layer `from`'s ffn_norm output
static std::vector<float> guess_scores(int from, int to, const float * x_from) {
    std::vector<float> x(n_embd);
    for (int i = 0; i < n_embd; i++) {
        float w = normw[from][i];
        x[i] = w != 0.0f ? x_from[i] / w * normw[to][i] : 0.0f;
    }
    std::vector<float> logits(n_expert);
    const float * R = router[to].data();
    for (int e = 0; e < n_expert; e++) {
        double acc = 0;
        for (int i = 0; i < n_embd; i++) acc += (double) R[(size_t) e * n_embd + i] * x[i];
        logits[e] = (float) acc;
    }
    std::vector<float> s;
    scores(to, logits.data(), s);
    return s;
}

// what the graph callback captures for one decode call
struct Capture {
    int n_tokens = 0;
    std::map<int, std::vector<float>> norm;    // layer -> n_embd x n_tokens
    std::map<int, std::vector<float>> logits;  // layer -> n_expert x n_tokens
};
static Capture cap;

static bool cb_eval(ggml_tensor * t, bool ask, void *) {
    const char * n = t->name;
    int il = -1;
    bool is_norm = false;
    if (sscanf(n, "model.layers.{}.ffn_norm-%d", &il) == 1) is_norm = true;
    else if (sscanf(n, "ffn_moe_logits-%d", &il) != 1) return false;
    if (ask) return true;
    // router inputs are only needed for generated tokens
    if (is_norm && t->ne[1] > 1) return true;
    std::vector<float> buf(ggml_nelements(t));
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) { fprintf(stderr, "unexpected %s\n", n); return true; }
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    // a prompt larger than n_ubatch arrives in several parts
    auto & v = (is_norm ? cap.norm : cap.logits)[il];
    v.insert(v.end(), buf.begin(), buf.end());
    return true;
}

// one generated token: per MoE layer, its router input and its real choice
struct Step {
    std::vector<std::vector<float>> x;     // [layer] n_embd
    std::vector<std::vector<int>> chosen;  // [layer] n_used
};

struct Run {
    std::string name;
    std::vector<std::vector<int>> prompt_counts;  // [layer] per-expert uses over the prompt
    std::vector<std::vector<int>> prompt_order;   // [layer] experts in order of first use in the prompt
    std::vector<Step> steps;
};

static std::string slurp(const char * p) { std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str(); }

static Run run_prompt(llama_model * model, const char * file) {
    Run run;
    run.name = file;
    std::string all = slurp(file);
    size_t u = all.find("\n---USER---\n"), nn = all.find("\n---N---\n");
    std::string sys = all.substr(0, u), user = all.substr(u + 12, nn - u - 12);
    int n_gen = atoi(all.c_str() + nn + 9);

    llama_chat_message msgs[2] = { { "system", sys.c_str() }, { "user", user.c_str() } };
    const char * tmpl = llama_model_chat_template(model, nullptr);
    std::vector<char> buf(all.size() * 2 + 4096);
    int len = llama_chat_apply_template(tmpl, msgs, 2, true, buf.data(), buf.size());
    std::string prompt(buf.data(), len);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = 8192; cp.n_batch = 2048; cp.n_ubatch = 512; cp.n_threads = 4;
    cp.cb_eval = cb_eval; cp.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cp);
    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true);
    toks.resize(n);
    fprintf(stderr, "%s: %d prompt tokens, generating %d\n", file, n, n_gen);

    run.prompt_counts.assign(n_layer, std::vector<int>(n_expert, 0));
    run.prompt_order.assign(n_layer, {});
    auto * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch b = llama_batch_get_one(toks.data(), toks.size());
    llama_token tok;
    std::string out;
    for (int i = 0; i <= n_gen; i++) {
        cap = Capture();
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); break; }
        const int nt = b.n_tokens;
        if (i == 0) {
            for (auto & [il, lg] : cap.logits) {
                // the last layer only keeps the output row
                int rows = (int) (lg.size() / n_expert);
                std::vector<float> s;
                for (int r = 0; r < rows; r++) {
                    scores(il, lg.data() + (size_t) r * n_expert, s);
                    for (int e : topk(s, n_used)) {
                        if (run.prompt_counts[il][e]++ == 0) run.prompt_order[il].push_back(e);
                    }
                }
            }
        } else if (nt == 1) {
            Step st;
            st.x.assign(n_layer, {});
            st.chosen.assign(n_layer, {});
            for (int il = dense_lead; il < n_layer; il++) {
                if (!cap.norm.count(il) || !cap.logits.count(il)) { fprintf(stderr, "layer %d not captured\n", il); exit(1); }
                st.x[il] = cap.norm[il];
                std::vector<float> s;
                scores(il, cap.logits[il].data(), s);
                st.chosen[il] = topk(s, n_used);
            }
            run.steps.push_back(std::move(st));
        }
        if (i == n_gen) break;
        tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        char pb[256];
        int pl = llama_token_to_piece(vocab, tok, pb, sizeof pb, 0, true);
        out.append(pb, pl);
        b = llama_batch_get_one(&tok, 1);
    }
    fprintf(stderr, "output: %.200s\n", out.c_str());
    llama_sampler_free(smpl);
    llama_free(ctx);
    return run;
}

// LRU over (layer, expert) units, each one expert's gate+up+down in one layer
struct Lru {
    size_t cap;
    std::list<int> order;  // front = most recent
    std::unordered_map<int, std::list<int>::iterator> pos;
    bool has(int k) const { return pos.count(k) > 0; }
    void touch(int k) {
        auto it = pos.find(k);
        if (it != pos.end()) order.erase(it->second);
        order.push_front(k);
        pos[k] = order.begin();
        while (order.size() > cap) { pos.erase(order.back()); order.pop_back(); }
    }
    bool full() const { return order.size() >= cap; }
};

struct SimResult { double misses = 0, covered = 0, extra = 0, tokens = 0; };

// warm: "first" fills the cache in order of first use in the prompt while it
// has room (what the engine does now); "freq" keeps the prompt's most used.
// prefetch: 0 = none, else guesses for layer L+ahead made at layer L, top `width`.
static SimResult simulate(const Run & run, size_t cap, const char * warm, int ahead, int width) {
    Lru c{cap};
    const int L = n_layer;
    if (strcmp(warm, "first") == 0) {
        // the engine walks layer by layer through each prompt batch; approximate
        // with one pass in layer order, experts in order of first use
        for (int il = dense_lead; il < L && !c.full(); il++)
            for (int e : run.prompt_order[il]) { if (c.full()) break; c.touch(il * n_expert + e); }
    } else {
        std::vector<std::pair<int, int>> all;  // (count, key)
        for (int il = dense_lead; il < L; il++)
            for (int e = 0; e < n_expert; e++)
                if (run.prompt_counts[il][e]) all.push_back({run.prompt_counts[il][e], il * n_expert + e});
        std::sort(all.begin(), all.end(), [](auto & a, auto & b) { return a.first < b.first; });
        for (auto & [cnt, k] : all) c.touch(k);  // most used touched last = kept
    }
    SimResult r;
    std::set<int> inflight;  // prefetched, not yet used
    for (const Step & st : run.steps) {
        for (int il = dense_lead; il < L; il++) {
            // the guess for il+ahead, issued now
            if (ahead > 0 && il + ahead < L) {
                auto s = guess_scores(il, il + ahead, st.x[il].data());
                for (int e : topk(s, width)) {
                    int k = (il + ahead) * n_expert + e;
                    if (!c.has(k)) { r.extra += 1; inflight.insert(k); c.touch(k); }
                }
            }
            for (int e : st.chosen[il]) {
                int k = il * n_expert + e;
                if (inflight.count(k)) { r.covered += 1; r.extra -= 1; inflight.erase(k); }
                else if (!c.has(k)) r.misses += 1;
                c.touch(k);
            }
        }
        inflight.clear();
        r.tokens += 1;
    }
    return r;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: prefetch_probe model.gguf budget_mb prompt.txt...\n"); return 1; }
    llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_WARN) fputs(t, stderr); }, nullptr);
    load_router(argv[1]);
    const double budget_mb = atof(argv[2]);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.use_extra_bufts = false;  // no repacked copy: keep the weights in the mapping
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;

    std::vector<Run> runs;
    for (int i = 3; i < argc; i++) runs.push_back(run_prompt(model, argv[i]));

    const int K = n_used;
    printf("\n== guess accuracy: share of the %d real experts in the guess (generation tokens) ==\n", K);
    printf("%-28s", "guess");
    for (auto & r : runs) printf(" %14.14s", r.name.c_str());
    printf("\n");
    // previous token, same layer
    {
        printf("%-28s", "previous token, same layer");
        for (auto & r : runs) {
            double hit = 0, tot = 0;
            for (size_t t = 1; t < r.steps.size(); t++)
                for (int il = dense_lead; il < n_layer; il++) {
                    std::set<int> prev(r.steps[t - 1].chosen[il].begin(), r.steps[t - 1].chosen[il].end());
                    for (int e : r.steps[t].chosen[il]) { hit += prev.count(e); tot++; }
                }
            printf(" %13.0f%%", 100 * hit / tot);
        }
        printf("\n");
    }
    for (int ahead : {0, 1, 2, 3}) {
        for (int width : {K, K + 2, 2 * K}) {
            char label[64];
            snprintf(label, sizeof label, "%s, top %d", ahead == 0 ? "own layer (check)" : (std::to_string(ahead) + " layer(s) ahead").c_str(), width);
            if (ahead == 0 && width != K) continue;
            printf("%-28s", label);
            for (auto & r : runs) {
                double hit = 0, tot = 0;
                for (auto & st : r.steps)
                    for (int il = dense_lead; il + ahead < n_layer; il++) {
                        auto s = guess_scores(il, il + ahead, st.x[il].data());
                        auto g = topk(s, width);
                        std::set<int> gs(g.begin(), g.end());
                        for (int e : st.chosen[il + ahead]) { hit += gs.count(e); tot++; }
                    }
                printf(" %13.0f%%", 100 * hit / tot);
            }
            printf("\n");
        }
    }

    const size_t cap = (size_t) (budget_mb * 1e6 / expert_bytes * 1.048576);
    printf("\n== cache at %.0f MB (%zu of %d expert slots), per generated token ==\n", budget_mb, cap, (n_layer - dense_lead) * n_expert);
    printf("%-40s %-14s %8s %9s %8s %10s\n", "setup", "prompt", "misses", "prefetched", "extra", "MB/token");
    for (const char * warm : {"first", "freq"}) {
        for (auto cfg : std::vector<std::pair<int, int>>{{0, 0}, {1, K}, {1, K + 2}, {2, K}, {2, K + 2}}) {
            char label[64];
            snprintf(label, sizeof label, "warm=%s %s", warm,
                     cfg.first == 0 ? "no prefetch" : ("prefetch +" + std::to_string(cfg.first) + " top " + std::to_string(cfg.second)).c_str());
            for (auto & r : runs) {
                auto s = simulate(r, cap, warm, cfg.first, cfg.second);
                double t = std::max(1.0, s.tokens);
                printf("%-40s %-14.14s %8.1f %9.1f %8.1f %10.0f\n", label, r.name.c_str(), s.misses / t, s.covered / t, s.extra / t,
                       (s.misses + s.covered + s.extra) / t * expert_bytes / 1e6);
            }
        }
    }
    llama_model_free(model);
    return 0;
}
