#include "spark_corpus.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace dflash::common {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

constexpr std::size_t kPerSession = 4;  // chunks sampled per session

std::string env_or(const char * name, const fs::path & fallback) {
    if (const char * v = std::getenv(name)) return std::string(v);
    return fallback.string();
}

void append(std::string & out, const std::string & s) {
    if (!s.empty()) { out += s; out += '\n'; }
}

// Claude Code message.content (string or array of typed blocks) -> text.
void claude_blocks(const json & content, std::string & out) {
    if (content.is_string()) { append(out, content.get<std::string>()); return; }
    if (!content.is_array()) return;
    for (const auto & b : content) {
        if (!b.is_object()) continue;
        const std::string t = b.value("type", "");
        if (t == "text" && b.contains("text")) {
            append(out, b["text"].get<std::string>());
        } else if (t == "thinking" && b.contains("thinking")) {
            append(out, b["thinking"].get<std::string>());
        } else if (t == "tool_use" && b.contains("input")) {
            append(out, b["input"].dump().substr(0, 4000));
        } else if (t == "tool_result" && b.contains("content")) {
            const auto & cc = b["content"];
            if (cc.is_string()) {
                append(out, cc.get<std::string>());
            } else if (cc.is_array()) {
                for (const auto & x : cc)
                    if (x.is_object() && x.value("type", "") == "text")
                        append(out, x.value("text", ""));
            }
        }
    }
}

std::string claude_session_text(const fs::path & p) {
    std::ifstream f(p);
    std::string line, out;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        json o;
        try { o = json::parse(line); } catch (...) { continue; }
        const std::string ty = o.value("type", "");
        if ((ty == "user" || ty == "assistant") && o.contains("message") &&
            o["message"].is_object() && o["message"].contains("content"))
            claude_blocks(o["message"]["content"], out);
    }
    return out;
}

// Codex rollout response_item (user + assistant; skips developer boilerplate).
std::string codex_session_text(const fs::path & p) {
    std::ifstream f(p);
    std::string line, out;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        json o;
        try { o = json::parse(line); } catch (...) { continue; }
        if (o.value("type", "") != "response_item") continue;
        if (!o.contains("payload") || !o["payload"].is_object()) continue;
        const auto & pl = o["payload"];
        const std::string role = pl.value("role", "");
        if (role != "user" && role != "assistant") continue;
        if (!pl.contains("content")) continue;
        const auto & c = pl["content"];
        if (c.is_string()) {
            append(out, c.get<std::string>());
        } else if (c.is_array()) {
            for (const auto & b : c) {
                if (!b.is_object()) continue;
                const std::string t = b.value("type", "");
                if ((t == "input_text" || t == "output_text" || t == "text") && b.contains("text"))
                    append(out, b["text"].get<std::string>());
            }
        }
    }
    return out;
}

void chunk_and_sample(const std::string & text, std::size_t chunk_chars, std::size_t min_chars,
                      std::vector<std::string> & out) {
    std::vector<std::string> chunks;
    for (std::size_t i = 0; i < text.size(); i += chunk_chars) {
        std::string c = text.substr(i, chunk_chars);
        if (c.size() >= min_chars) chunks.push_back(std::move(c));
    }
    if (chunks.empty()) return;
    const std::size_t step = chunks.size() / kPerSession ? chunks.size() / kPerSession : 1;
    for (std::size_t i = 0, taken = 0; i < chunks.size() && taken < kPerSession; i += step, ++taken)
        out.push_back(chunks[i]);
}

// Walk a directory tree, feeding each matching .jsonl through `extract`.
template <typename Accept, typename Extract>
void scan_dir(const fs::path & dir, std::size_t max_chunks, std::size_t chunk_chars,
              std::size_t min_chars, Accept accept, Extract extract,
              std::vector<std::string> & out) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return;
    for (auto it = fs::recursive_directory_iterator(dir, ec);
         it != fs::recursive_directory_iterator() && out.size() < max_chunks; it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file(ec)) continue;
        if (it->path().extension() != ".jsonl") continue;
        if (!accept(it->path())) continue;
        chunk_and_sample(extract(it->path()), chunk_chars, min_chars, out);
    }
}

}  // namespace

std::vector<std::string> spark_scrape_corpus(std::size_t max_chunks, std::size_t chunk_chars,
                                             std::size_t min_chars) {
    std::vector<std::string> out;
    const char * home = std::getenv("HOME");
    const fs::path home_dir = home ? fs::path(home) : fs::path();

    const fs::path claude_dir = env_or("DFLASH_SPARK_CLAUDE_DIR", home_dir / ".claude" / "projects");
    const fs::path codex_dir  = env_or("DFLASH_SPARK_CODEX_DIR",  home_dir / ".codex" / "sessions");

    scan_dir(claude_dir, max_chunks, chunk_chars, min_chars,
             [](const fs::path &) { return true; }, claude_session_text, out);
    scan_dir(codex_dir, max_chunks, chunk_chars, min_chars,
             [](const fs::path & p) { return p.filename().string().rfind("rollout-", 0) == 0; },
             codex_session_text, out);

    if (out.size() > max_chunks) out.resize(max_chunks);
    return out;
}

}  // namespace dflash::common
