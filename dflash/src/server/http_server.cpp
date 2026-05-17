// HTTP server implementation.
//
// Core infrastructure: socket listen/accept, client threads, HTTP parsing,
// job queue, worker thread with SSE streaming and disconnect detection.

#include "http_server.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

namespace dflash27b {

// ─── Utilities ──────────────────────────────────────────────────────────

static std::string generate_id(const char * prefix) {
    static std::atomic<uint64_t> counter{0};
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s_%016llx",
                  prefix, (unsigned long long)counter.fetch_add(1));
    return buf;
}

// ─── HttpServer ─────────────────────────────────────────────────────────

HttpServer::HttpServer(ModelBackend & backend,
                       Tokenizer & tokenizer,
                       const ServerConfig & config)
    : backend_(backend)
    , tokenizer_(tokenizer)
    , config_(config)
    , chat_format_(ChatFormat::QWEN3)  // default, overridden by arch
{
}

HttpServer::~HttpServer() {
    shutdown();
}

void HttpServer::shutdown() {
    stopping_.store(true);
    queue_cv_.notify_all();
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

int HttpServer::run() {
    // Ignore SIGPIPE so send() returns EPIPE instead of killing the process.
    signal(SIGPIPE, SIG_IGN);

    // Create listen socket.
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        std::fprintf(stderr, "[server] socket() failed: %s\n", strerror(errno));
        return 1;
    }

    int yes = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)config_.port);
    inet_pton(AF_INET, config_.host.c_str(), &sa.sin_addr);

    if (bind(listen_fd_, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        std::fprintf(stderr, "[server] bind(%s:%d) failed: %s\n",
                     config_.host.c_str(), config_.port, strerror(errno));
        ::close(listen_fd_);
        listen_fd_ = -1;
        return 1;
    }

    if (listen(listen_fd_, 128) < 0) {
        std::fprintf(stderr, "[server] listen() failed: %s\n", strerror(errno));
        ::close(listen_fd_);
        listen_fd_ = -1;
        return 1;
    }

    std::fprintf(stderr, "[server] listening on http://%s:%d\n",
                 config_.host.c_str(), config_.port);

    // Start worker thread.
    worker_thread_ = std::thread([this]() { worker_loop(); });

    // Accept loop.
    while (!stopping_.load()) {
        struct sockaddr_in client_sa{};
        socklen_t client_len = sizeof(client_sa);
        int client_fd = accept(listen_fd_, (struct sockaddr *)&client_sa, &client_len);
        if (client_fd < 0) {
            if (stopping_.load()) break;
            if (errno == EINTR) continue;
            std::fprintf(stderr, "[server] accept() error: %s\n", strerror(errno));
            continue;
        }

        // Disable Nagle for low-latency SSE streaming.
        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        // Spawn client thread (detached — client_main owns the fd).
        std::thread([this, client_fd]() {
            handle_client(client_fd);
        }).detach();
    }

    // Wait for worker to finish.
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    return 0;
}

// ─── Client thread ──────────────────────────────────────────────────────

void HttpServer::handle_client(int fd) {
    HttpRequest hr;
    if (!read_http_request(fd, hr)) {
        send_error(fd, 400, "bad HTTP request");
        ::close(fd);
        return;
    }

    // CORS preflight.
    if (hr.method == "OPTIONS") {
        send_response(fd, 204, "", "");
        ::close(fd);
        return;
    }

    // Health check.
    if (hr.method == "GET" && (hr.path == "/health" || hr.path == "/")) {
        send_response(fd, 200, "application/json", "{\"status\":\"ok\"}\n");
        ::close(fd);
        return;
    }

    // Models endpoint.
    if (hr.method == "GET" && hr.path == "/v1/models") {
        json models = {
            {"object", "list"},
            {"data", json::array({
                {{"id", config_.model_name},
                 {"object", "model"},
                 {"owned_by", "dflash"}}
            })}
        };
        send_response(fd, 200, "application/json", models.dump() + "\n");
        ::close(fd);
        return;
    }

    // Route POST endpoints.
    if (!route_request(fd, hr)) {
        send_error(fd, 404, "unknown endpoint");
    }
    ::close(fd);
}

bool HttpServer::route_request(int fd, const HttpRequest & hr) {
    if (hr.method != "POST") return false;

    ParsedRequest req;
    std::string err;

    try {
        json body = json::parse(hr.body);

        // Common fields.
        req.stream = body.value("stream", true);
        req.model = body.value("model", config_.model_name);
        req.max_output = body.value("max_tokens",
                         body.value("max_output_tokens",
                         body.value("max_completion_tokens", config_.max_tokens)));

        // Sampler parameters.
        req.sampler.temp = body.value("temperature", 0.0f);
        req.sampler.top_p = body.value("top_p", 1.0f);
        req.sampler.top_k = body.value("top_k", 0);
        if (body.contains("seed")) {
            req.sampler.seed = body["seed"].get<uint64_t>();
        }

        // Tools.
        if (body.contains("tools")) {
            req.tools = body["tools"];
        }

        if (hr.path == "/v1/chat/completions") {
            req.format = ApiFormat::OPENAI_CHAT;
            req.messages = body["messages"];
        } else if (hr.path == "/v1/messages") {
            req.format = ApiFormat::ANTHROPIC;
            req.messages = body["messages"];
            if (body.contains("system")) {
                // Anthropic puts system as a top-level field.
                json sys_msg = {{"role", "system"}, {"content", body["system"]}};
                req.messages.insert(req.messages.begin(), sys_msg);
            }
        } else if (hr.path == "/v1/responses") {
            req.format = ApiFormat::RESPONSES;
            req.response_id = generate_id("resp");
            // Responses API uses "input" instead of "messages".
            if (body.contains("input")) {
                req.messages = body["input"];
            }
            if (body.contains("instructions")) {
                json sys_msg = {{"role", "system"}, {"content", body["instructions"]}};
                if (req.messages.is_array()) {
                    req.messages.insert(req.messages.begin(), sys_msg);
                } else {
                    req.messages = json::array({sys_msg, {{"role", "user"}, {"content", body["input"]}}});
                }
            }
        } else {
            return false;
        }

        // Render messages to text and tokenize.
        std::vector<ChatMessage> chat_msgs;
        if (req.messages.is_array()) {
            for (const auto & m : req.messages) {
                ChatMessage cm;
                cm.role = m.value("role", "user");
                if (m["content"].is_string()) {
                    cm.content = m["content"].get<std::string>();
                } else if (m["content"].is_array()) {
                    // Multi-part content (text parts only for now).
                    for (const auto & part : m["content"]) {
                        if (part.value("type", "") == "text") {
                            cm.content += part.value("text", "");
                        }
                    }
                }
                chat_msgs.push_back(std::move(cm));
            }
        } else if (req.messages.is_string()) {
            // Simple string input (Responses API shorthand).
            chat_msgs.push_back({"user", req.messages.get<std::string>()});
        }

        std::string rendered = render_chat_template(chat_msgs, chat_format_, true);
        req.prompt_tokens = tokenizer_.encode(rendered);

    } catch (const std::exception & e) {
        send_error(fd, 400, std::string("JSON parse error: ") + e.what());
        return true;  // handled (with error)
    }

    // Check context length.
    if ((int)req.prompt_tokens.size() + req.max_output > config_.max_ctx) {
        send_error(fd, 400, "prompt + max_tokens exceeds context window");
        return true;
    }

    // Set socket non-blocking for send() stall detection during streaming.
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    // Enqueue job and wait for worker.
    ServerJob job;
    job.fd = fd;
    job.req = std::move(req);

    enqueue(&job);

    // Wait for the worker to signal completion.
    {
        std::unique_lock<std::mutex> lk(job.mu);
        job.cv.wait(lk, [&]() { return job.done; });
    }

    return true;
}

// ─── Worker thread ──────────────────────────────────────────────────────

void HttpServer::worker_loop() {
    while (true) {
        ServerJob * job = dequeue();
        if (!job) break;  // stopping

        int fd = job->fd;
        const auto & req = job->req;

        // Send SSE headers.
        if (req.stream) {
            if (!send_sse_headers(fd)) {
                // Client already disconnected before we started.
                std::lock_guard<std::mutex> lk(job->mu);
                job->done = true;
                job->cv.notify_one();
                continue;
            }
        }

        // Build generate request.
        GenerateRequest gen_req;
        gen_req.prompt = req.prompt_tokens;
        gen_req.n_gen = req.max_output;
        gen_req.sampler = req.sampler;
        gen_req.do_sample = req.sampler.temp > 0.0f;
        gen_req.stream = false;  // we handle streaming via on_token callback

        // Set up DaemonIO with on_token callback for streaming + disconnect.
        DaemonIO io;
        io.stream_fd = -1;  // no pipe — we write SSE directly

        std::string accumulated_text;
        bool client_disconnected = false;

        io.on_token = [&](int32_t token) -> bool {
            if (client_disconnected) return false;

            std::string text = tokenizer_.token_text(token);
            accumulated_text += text;

            if (req.stream && !text.empty()) {
                // Format SSE chunk based on API format.
                std::string chunk;
                switch (req.format) {
                case ApiFormat::OPENAI_CHAT: {
                    json delta = {{"choices", json::array({{
                        {"index", 0},
                        {"delta", {{"content", text}}}
                    }})}};
                    chunk = delta.dump();
                    break;
                }
                case ApiFormat::ANTHROPIC: {
                    json event = {{"type", "content_block_delta"},
                                  {"index", 0},
                                  {"delta", {{"type", "text_delta"}, {"text", text}}}};
                    chunk = event.dump();
                    break;
                }
                case ApiFormat::RESPONSES: {
                    json event = {{"type", "response.output_text.delta"},
                                  {"delta", text}};
                    chunk = event.dump();
                    break;
                }
                default:
                    chunk = text;
                }

                if (!send_sse_chunk(fd, chunk)) {
                    client_disconnected = true;
                    return false;  // abort generation
                }
            }
            return true;
        };

        // Run generation.
        GenerateResult result = backend_.generate(gen_req, io);

        // Send final SSE event.
        if (req.stream && !client_disconnected) {
            send_sse_chunk(fd, "[DONE]");
        } else if (!req.stream && !client_disconnected) {
            // Non-streaming: send complete JSON response.
            json resp;
            switch (req.format) {
            case ApiFormat::OPENAI_CHAT:
                resp = {{"choices", json::array({{
                    {"index", 0},
                    {"message", {{"role", "assistant"}, {"content", accumulated_text}}},
                    {"finish_reason", result.ok ? "stop" : "error"}
                }})}};
                break;
            case ApiFormat::ANTHROPIC:
                resp = {{"content", json::array({{
                    {"type", "text"},
                    {"text", accumulated_text}
                }})}};
                break;
            case ApiFormat::RESPONSES:
                resp = {{"id", req.response_id},
                        {"output", json::array({{
                            {"type", "message"},
                            {"content", json::array({{
                                {"type", "output_text"},
                                {"text", accumulated_text}
                            }})}
                        }})}};
                break;
            default:
                resp = {{"text", accumulated_text}};
            }
            // Set socket back to blocking for the final send.
            int flags = fcntl(fd, F_GETFL, 0);
            if (flags >= 0) fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
            send_response(fd, 200, "application/json", resp.dump() + "\n");
        }

        if (client_disconnected) {
            std::fprintf(stderr, "[server] client disconnected — generation aborted "
                         "(prompt=%zu out=%zu)\n",
                         req.prompt_tokens.size(), result.tokens.size());
        }

        // Signal client thread that we're done.
        {
            std::lock_guard<std::mutex> lk(job->mu);
            job->done = true;
            job->cv.notify_one();
        }
    }
}

// ─── Job queue ──────────────────────────────────────────────────────────

void HttpServer::enqueue(ServerJob * job) {
    std::lock_guard<std::mutex> lk(queue_mu_);
    job->next = nullptr;
    if (queue_tail_) queue_tail_->next = job;
    else queue_head_ = job;
    queue_tail_ = job;
    queue_cv_.notify_one();
}

ServerJob * HttpServer::dequeue() {
    std::unique_lock<std::mutex> lk(queue_mu_);
    queue_cv_.wait(lk, [this]() { return queue_head_ != nullptr || stopping_.load(); });
    if (!queue_head_) return nullptr;
    ServerJob * j = queue_head_;
    queue_head_ = j->next;
    if (!queue_head_) queue_tail_ = nullptr;
    j->next = nullptr;
    return j;
}

// ─── HTTP I/O ───────────────────────────────────────────────────────────

bool HttpServer::read_http_request(int fd, HttpRequest & out) {
    std::string buf;
    buf.reserve(8192);
    char tmp[4096];

    // Read until we find the header/body boundary (\r\n\r\n or \n\n).
    ssize_t hend = -1;
    while (hend < 0 && buf.size() < 65536) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        buf.append(tmp, n);

        // Look for end of headers.
        for (size_t i = 3; i < buf.size(); i++) {
            if (buf[i-3] == '\r' && buf[i-2] == '\n' &&
                buf[i-1] == '\r' && buf[i] == '\n') {
                hend = i + 1;
                break;
            }
        }
        if (hend < 0) {
            for (size_t i = 1; i < buf.size(); i++) {
                if (buf[i-1] == '\n' && buf[i] == '\n') {
                    hend = i + 1;
                    break;
                }
            }
        }
    }
    if (hend < 0) return false;

    // Parse request line.
    size_t line_end = buf.find('\n');
    if (line_end == std::string::npos) return false;
    std::string line = buf.substr(0, line_end);
    if (!line.empty() && line.back() == '\r') line.pop_back();

    // "METHOD /path HTTP/1.1"
    size_t sp1 = line.find(' ');
    size_t sp2 = line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    out.method = line.substr(0, sp1);
    out.path = line.substr(sp1 + 1, sp2 - sp1 - 1);

    // Strip query string from path.
    size_t q = out.path.find('?');
    if (q != std::string::npos) out.path = out.path.substr(0, q);

    // Find Content-Length.
    long content_length = 0;
    {
        std::string headers = buf.substr(0, hend);
        std::string lower_headers = headers;
        std::transform(lower_headers.begin(), lower_headers.end(),
                       lower_headers.begin(), ::tolower);
        size_t cl_pos = lower_headers.find("content-length:");
        if (cl_pos != std::string::npos) {
            size_t val_start = cl_pos + 15;
            while (val_start < lower_headers.size() &&
                   lower_headers[val_start] == ' ') val_start++;
            content_length = std::strtol(headers.c_str() + val_start, nullptr, 10);
        }
    }

    if (content_length < 0 || content_length > 64 * 1024 * 1024) return false;

    // Read body.
    while ((ssize_t)buf.size() < hend + content_length) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        buf.append(tmp, n);
    }

    out.body = buf.substr(hend, content_length);
    return true;
}

bool HttpServer::send_all(int fd, const void * data, size_t len) {
    const char * p = (const char *)data;
    size_t sent = 0;
    while (sent < len) {
        // Use poll() with timeout to detect stalled connections.
        struct pollfd pfd = {fd, POLLOUT, 0};
        int ret = poll(&pfd, 1, 30000);  // 30s timeout
        if (ret <= 0) return false;       // timeout or error
        if (pfd.revents & (POLLERR | POLLHUP)) return false;

        ssize_t n = send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
            return false;  // EPIPE, ECONNRESET, etc.
        }
        sent += n;
    }
    return true;
}

bool HttpServer::send_response(int fd, int status, const std::string & content_type,
                               const std::string & body) {
    std::string header = "HTTP/1.1 " + std::to_string(status) + " OK\r\n";
    if (config_.enable_cors) {
        header += "Access-Control-Allow-Origin: *\r\n"
                  "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                  "Access-Control-Allow-Headers: *\r\n";
    }
    if (!content_type.empty()) {
        header += "Content-Type: " + content_type + "\r\n";
    }
    header += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    header += "Connection: close\r\n\r\n";
    header += body;
    return send_all(fd, header.data(), header.size());
}

bool HttpServer::send_error(int fd, int status, const std::string & message) {
    json err = {{"error", {{"message", message}, {"type", "invalid_request_error"}}}};
    return send_response(fd, status, "application/json", err.dump() + "\n");
}

bool HttpServer::send_sse_headers(int fd) {
    std::string header = "HTTP/1.1 200 OK\r\n";
    if (config_.enable_cors) {
        header += "Access-Control-Allow-Origin: *\r\n";
    }
    header += "Content-Type: text/event-stream\r\n"
              "Cache-Control: no-cache\r\n"
              "Connection: keep-alive\r\n\r\n";
    return send_all(fd, header.data(), header.size());
}

bool HttpServer::send_sse_chunk(int fd, const std::string & data) {
    std::string chunk = "data: " + data + "\n\n";
    return send_all(fd, chunk.data(), chunk.size());
}

bool HttpServer::send_sse_done(int fd) {
    return send_sse_chunk(fd, "[DONE]");
}

}  // namespace dflash27b
