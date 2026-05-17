// HTTP server infrastructure for dflash27b native server.
//
// Ported from ds4_server.c's socket/threading/HTTP layer, converted to C++.
// Architecture:
//   - Main thread: listen + accept
//   - Per-client thread: parse HTTP request, enqueue job, wait for completion
//   - Single worker thread: dequeue jobs, call ModelBackend::generate()
//
// Client disconnect detection: the worker writes SSE chunks via send().
// If send() fails (EPIPE/ECONNRESET), generation aborts immediately.

#pragma once

#include "common/model_backend.h"
#include "tokenizer.h"
#include "chat_template.h"
#include "third_party/nlohmann/json.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dflash27b {

using json = nlohmann::json;

// ─── Forward declarations ───────────────────────────────────────────────
struct ServerJob;

// ─── Server configuration ───────────────────────────────────────────────
struct ServerConfig {
    std::string host        = "0.0.0.0";
    int         port        = 8080;
    int         max_tokens  = 4096;     // default max output tokens
    int         max_ctx     = 131072;   // model context window
    bool        enable_cors = true;
    std::string model_name  = "dflash";
};

// ─── Parsed request ─────────────────────────────────────────────────────
enum class ApiFormat { OPENAI_CHAT, ANTHROPIC, RESPONSES, COMPLETIONS };

struct ParsedRequest {
    ApiFormat                  format;
    std::vector<int32_t>      prompt_tokens;  // tokenized prompt
    int                       max_output   = 4096;
    bool                      stream       = true;
    SamplerCfg                sampler;
    std::string               model;
    // Tool definitions (stored as JSON for response formatting)
    json                      tools;
    // Original messages (for response formatting)
    json                      messages;
    // Response ID
    std::string               response_id;
};

// ─── HTTP server ────────────────────────────────────────────────────────
class HttpServer {
public:
    HttpServer(ModelBackend & backend,
               Tokenizer & tokenizer,
               const ServerConfig & config);
    ~HttpServer();

    HttpServer(const HttpServer &) = delete;
    HttpServer & operator=(const HttpServer &) = delete;

    // Start listening. Blocks until shutdown() is called.
    int run();

    // Signal the server to stop accepting new connections and drain.
    void shutdown();

private:
    // Client thread: read HTTP request, parse, enqueue job, wait.
    void handle_client(int fd);

    // Worker thread: process jobs sequentially.
    void worker_loop();

    // Parse HTTP request from socket.
    struct HttpRequest {
        std::string method;
        std::string path;
        std::string body;
    };
    bool read_http_request(int fd, HttpRequest & out);

    // Route request to appropriate parser.
    bool route_request(int fd, const HttpRequest & hr);

    // Send HTTP response helpers.
    bool send_response(int fd, int status, const std::string & content_type,
                       const std::string & body);
    bool send_error(int fd, int status, const std::string & message);
    bool send_sse_headers(int fd);
    bool send_sse_chunk(int fd, const std::string & data);
    bool send_sse_done(int fd);

    // Send raw bytes with stall detection.
    bool send_all(int fd, const void * data, size_t len);

    // Job queue.
    void enqueue(ServerJob * job);
    ServerJob * dequeue();

    // Members.
    ModelBackend &   backend_;
    Tokenizer &      tokenizer_;
    ServerConfig     config_;
    ChatFormat       chat_format_;

    // Worker thread.
    std::thread                     worker_thread_;
    std::mutex                      queue_mu_;
    std::condition_variable         queue_cv_;
    ServerJob *                     queue_head_ = nullptr;
    ServerJob *                     queue_tail_ = nullptr;
    std::atomic<bool>               stopping_{false};

    // Listen socket.
    int listen_fd_ = -1;
};

// ─── Job (stack-owned by client thread) ─────────────────────────────────
struct ServerJob {
    int           fd = -1;
    ParsedRequest req;
    bool          done = false;
    std::mutex    mu;
    std::condition_variable cv;
    ServerJob *   next = nullptr;
};

}  // namespace dflash27b
