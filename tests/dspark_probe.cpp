// DSpark (block draft) through llama_rn_context's completion path, the one the
// app drives. Runs the same chat three ways: plain, every turn drafted, and
// alternating (plain narration, drafted record), and prints each turn's reply,
// speed, acceptance and how many prompt tokens it re-read. Greedy sampling, so
// the drafted replies should match the plain ones (verification can still
// differ in the last bits of floating point and so, rarely, in a token).
// Usage:
//   dspark_probe <target.gguf> <dspark-draft.gguf> [n_max] [threads]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "common.h"
#include "rn-completion.h"
#include "rn-llama.h"

using namespace rnllama;

namespace {

struct Turn {
    std::string reply;
    double seconds = 0;
    size_t tokens = 0;
    size_t drafted = 0;
    size_t accepted = 0;
    size_t reprocessed = 0;
    size_t reused = 0;
};

const char * SYSTEM =
    "<|im_start|>system\nYou are the narrator of a dark-fantasy text adventure. "
    "Write vivid second-person prose, 2 short paragraphs, and end by asking what the player does.<|im_end|>\n";

std::string user(const std::string & s) { return "<|im_start|>user\n" + s + "<|im_end|>\n"; }
std::string assistant_open() { return "<|im_start|>assistant\n<think>\n\n</think>\n\n"; }

bool load(llama_rn_context & ctx, const char * target, const char * draft, int n_max, int threads) {
    common_params params;
    params.model.path = target;
    params.n_ctx = 4096;
    params.n_parallel = 1;
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.cpuparams.n_threads = threads;
    params.n_gpu_layers = 0;
    params.ctx_shift = false;
    params.sampling.temp = 0.0f;
    params.sampling.top_k = 1;
    if (draft != nullptr) {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
        params.speculative.draft.n_max = n_max;
        params.speculative.draft.mparams.path = draft;
        params.speculative.draft.n_gpu_layers = 0;
    }
    if (!ctx.loadModel(params)) return false;
    // As the JSI layer does after every load.
    ctx.attachThreadpoolsIfAvailable();
    if (ctx.completion == nullptr) ctx.completion = new llama_rn_context_completion(&ctx);
    return true;
}

Turn run(llama_rn_context & ctx, const std::string & prompt, bool drafted, int n_max, int max_new) {
    Turn t;
    ctx.params.prompt = prompt;
    ctx.params.n_predict = max_new;
    // Per call, the way JSIParams applies `spec_type`.
    if (ctx.blockDraftContext() != nullptr) {
        ctx.params.speculative.types = drafted
            ? std::vector<common_speculative_type>{ COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK }
            : std::vector<common_speculative_type>{ COMMON_SPECULATIVE_TYPE_NONE };
        ctx.params.speculative.draft.n_max = n_max;
    }
    auto * c = ctx.completion;
    const auto t0 = std::chrono::steady_clock::now();
    c->rewind();
    c->initSampling();
    c->loadPrompt({});
    t.reused = (size_t) c->n_past;
    if (std::getenv("PROBE_DEBUG")) fprintf(stderr, "DEBUG loadPrompt: reused=%d of %zu\n", (int) c->n_past, c->embd.size());
    c->beginCompletion();
    while (c->has_next_token && (int) t.tokens < max_new) {
        completion_token_output out = c->nextToken();
        // As the JSI loop does: drain on has_next_token. The speculative path
        // marks EOS while tokens of the accepted block are still pending.
        if (out.tok == -1) break;
        if (c->is_interrupted) break;
        t.reply += out.text;
        t.tokens++;
        if (c->stopped_limit) break;
    }
    c->endCompletion();
    if (std::getenv("PROBE_DEBUG")) {
        fprintf(stderr, "DEBUG after turn: embd=%zu n_past=%d pos_max=%d pos_min=%d\n", c->embd.size(), (int) c->n_past,
            (int) llama_memory_seq_pos_max(llama_get_memory(ctx.ctx), 0), (int) llama_memory_seq_pos_min(llama_get_memory(ctx.ctx), 0));
    }
    t.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    t.drafted = c->num_draft_tokens;
    t.accepted = c->num_draft_tokens_accepted;
    t.reprocessed = c->mtp_prompt_reprocessed;
    return t;
}

const std::vector<std::string> USER_TURNS = {
    "I push open the rusted gate of the abandoned monastery and step into the courtyard, torch raised.",
    "I walk to the fountain and look into the water.",
    "I call out: is anyone here?",
    "I head for the chapel doors.",
};

// mode: 0 plain, 1 every turn drafted, 2 drafted on odd turns only.
std::vector<Turn> chat(llama_rn_context & ctx, int mode, int n_max, int max_new) {
    std::vector<Turn> out;
    std::string convo = SYSTEM;
    for (size_t i = 0; i < USER_TURNS.size(); ++i) {
        convo += user(USER_TURNS[i]) + assistant_open();
        const bool drafted = mode == 1 || (mode == 2 && i % 2 == 1);
        Turn t = run(ctx, convo, drafted, n_max, max_new);
        convo += t.reply + "<|im_end|>\n";
        out.push_back(t);
    }
    return out;
}

void print(const char * label, const std::vector<Turn> & turns) {
    printf("\n== %s\n", label);
    for (size_t i = 0; i < turns.size(); ++i) {
        const auto & t = turns[i];
        printf("turn %zu: %zu tokens in %.2f s (%.1f tok/s overall), reused %zu, draft re-read %zu, drafted %zu accepted %zu\n",
            i + 1, t.tokens, t.seconds, t.tokens / t.seconds, t.reused, t.reprocessed, t.drafted, t.accepted);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <target.gguf> <dspark-draft.gguf> [n_max] [threads]\n", argv[0]);
        return 2;
    }
    const int n_max = argc > 3 ? atoi(argv[3]) : 7;
    const int threads = argc > 4 ? atoi(argv[4]) : 4;
    const int max_new = 120;

    std::vector<Turn> plain, drafted, mixed;
    // Plain turns only: with LLAMA_EXPERT_CACHE_* set, a quick streaming check.
    if (std::getenv("PROBE_PLAIN_ONLY")) {
        llama_rn_context ctx;
        if (!load(ctx, argv[1], nullptr, n_max, threads)) return 1;
        plain = chat(ctx, 0, n_max, max_new);
        print("plain", plain);
        return 0;
    }
    {
        llama_rn_context ctx;
        if (!load(ctx, argv[1], nullptr, n_max, threads)) { fprintf(stderr, "plain load failed\n"); return 1; }
        plain = chat(ctx, 0, n_max, max_new);
    }
    {
        llama_rn_context ctx;
        if (!load(ctx, argv[1], argv[2], n_max, threads)) { fprintf(stderr, "draft load failed\n"); return 1; }
        drafted = chat(ctx, 1, n_max, max_new);
        mixed = chat(ctx, 2, n_max, max_new);
    }
    print("plain", plain);
    print("dspark every turn", drafted);
    print("dspark on turns 2 and 4 (same context, after the run above)", mixed);

    int same = 0, fails = 0;
    for (size_t i = 0; i < plain.size(); ++i) {
        const bool a = plain[i].reply == drafted[i].reply;
        same += a;
        if (drafted[i].drafted == 0) {
            printf("FAIL: turn %zu drafted nothing\n", i + 1);
            fails++;
        }
        if (!a) printf("turn %zu differs:\n  plain:  %.160s\n  dspark: %.160s\n", i + 1, plain[i].reply.c_str(), drafted[i].reply.c_str());
    }
    for (size_t i = 0; i < mixed.size(); ++i) {
        const bool want = i % 2 == 1;
        if (want != (mixed[i].drafted > 0)) {
            printf("FAIL: mixed turn %zu drafted %zu, wanted %s\n", i + 1, mixed[i].drafted, want ? "some" : "none");
            fails++;
        }
    }
    printf("\nreplies identical to plain: %d of %zu\n", same, plain.size());
    printf("first reply:\n%s\n", drafted[0].reply.c_str());
    return fails == 0 ? 0 : 1;
}
