// dflash_server — native C++ HTTP server for dflash27b.
//
// Replaces the Python server.py for production use. Owns the ModelBackend
// directly (no subprocess, no pipe protocol), enabling:
//   - Immediate client-disconnect cancellation (via send() failure)
//   - Lower latency (no IPC overhead)
//   - Single binary deployment
//
// Usage:
//   dflash_server <model.gguf> [--draft <draft.gguf>] [--port 8080]
//                              [--host 0.0.0.0] [--max-ctx 131072]
//                              [--max-tokens 4096] [--gpu 0]

#include "http_server.h"
#include "common/backend_factory.h"
#include "common/gguf_inspect.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

using namespace dflash27b;

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "Usage: %s <model.gguf> [options]\n"
        "\n"
        "Options:\n"
        "  --draft <path>       Draft model for speculative decode (qwen35 only)\n"
        "  --port <N>           Listen port (default: 8080)\n"
        "  --host <addr>        Bind address (default: 0.0.0.0)\n"
        "  --max-ctx <N>        Max context length (default: 131072)\n"
        "  --max-tokens <N>     Default max output tokens (default: 4096)\n"
        "  --gpu <N>            Target GPU device (default: 0)\n"
        "  --draft-gpu <N>      Draft GPU device (default: 0)\n"
        "  --chunk <N>          Chunked-prefill chunk size (default: 512)\n"
        "  --model-name <name>  Model name for /v1/models (default: dflash)\n"
        "  --ddtree             Enable DDTree speculative decode\n"
        "  --ddtree-budget <N>  DDTree budget (default: 64)\n"
        "  --no-cors            Disable CORS headers\n"
        "\n", prog);
}

int main(int argc, char ** argv) {
    if (argc < 2 || argv[1][0] == '-') {
        print_usage(argv[0]);
        return 2;
    }

    // Parse arguments.
    BackendArgs bargs;
    ServerConfig sconfig;
    bargs.model_path = argv[1];

    for (int i = 2; i < argc; i++) {
        if (std::strcmp(argv[i], "--draft") == 0 && i + 1 < argc) {
            bargs.draft_path = argv[++i];
        } else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            sconfig.port = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            sconfig.host = argv[++i];
        } else if (std::strcmp(argv[i], "--max-ctx") == 0 && i + 1 < argc) {
            int v = std::atoi(argv[++i]);
            sconfig.max_ctx = v;
            bargs.device.max_ctx = v;
        } else if (std::strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            sconfig.max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            bargs.device.gpu = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--draft-gpu") == 0 && i + 1 < argc) {
            bargs.draft_gpu = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) {
            bargs.chunk = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--model-name") == 0 && i + 1 < argc) {
            sconfig.model_name = argv[++i];
        } else if (std::strcmp(argv[i], "--ddtree") == 0) {
            bargs.ddtree_mode = true;
            bargs.fast_rollback = true;
        } else if (std::strcmp(argv[i], "--ddtree-budget") == 0 && i + 1 < argc) {
            bargs.ddtree_budget = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-cors") == 0) {
            sconfig.enable_cors = false;
        } else {
            std::fprintf(stderr, "[server] unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }

    // Load tokenizer.
    std::fprintf(stderr, "[server] loading tokenizer from %s\n", bargs.model_path);
    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(bargs.model_path)) {
        std::fprintf(stderr, "[server] tokenizer load failed\n");
        return 1;
    }

    // Create backend.
    std::fprintf(stderr, "[server] creating backend...\n");
    auto backend = create_backend(bargs);
    if (!backend) {
        std::fprintf(stderr, "[server] backend creation failed\n");
        return 1;
    }

    // Start HTTP server.
    std::fprintf(stderr, "[server] starting HTTP server on %s:%d\n",
                 sconfig.host.c_str(), sconfig.port);
    HttpServer server(*backend, tokenizer, sconfig);
    int ret = server.run();

    // Cleanup.
    backend->shutdown();
    return ret;
}
