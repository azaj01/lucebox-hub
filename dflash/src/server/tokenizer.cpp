// BPE tokenizer implementation.
//
// GGUF loading uses ggml's gguf_init_from_file API (already vendored).
// Pre-tokenization is a hand-coded state machine matching the Qwen3/3.5
// regex pattern without pulling in a regex library.

#include "tokenizer.h"

#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

namespace dflash27b {

// ─── Unicode helpers ────────────────────────────────────────────────────

static int utf8_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;  // invalid — advance one byte
}

static uint32_t utf8_decode(const char * s, int * len) {
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80) { *len = 1; return c; }
    if ((c & 0xE0) == 0xC0) {
        *len = 2;
        return ((uint32_t)(c & 0x1F) << 6) |
               ((uint32_t)((uint8_t)s[1]) & 0x3F);
    }
    if ((c & 0xF0) == 0xE0) {
        *len = 3;
        return ((uint32_t)(c & 0x0F) << 12) |
               (((uint32_t)((uint8_t)s[1]) & 0x3F) << 6) |
               ((uint32_t)((uint8_t)s[2]) & 0x3F);
    }
    if ((c & 0xF8) == 0xF0) {
        *len = 4;
        return ((uint32_t)(c & 0x07) << 18) |
               (((uint32_t)((uint8_t)s[1]) & 0x3F) << 12) |
               (((uint32_t)((uint8_t)s[2]) & 0x3F) << 6) |
               ((uint32_t)((uint8_t)s[3]) & 0x3F);
    }
    *len = 1;
    return 0xFFFD;
}

// Unicode character property tests (simplified — covers the ranges needed
// for Qwen3/3.5 tokenizer pre-split).

static bool is_letter(uint32_t cp) {
    // ASCII letters
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    // Common Latin-1 Supplement letters
    if (cp >= 0xC0 && cp <= 0xFF && cp != 0xD7 && cp != 0xF7) return true;
    // Latin Extended-A/B
    if (cp >= 0x100 && cp <= 0x24F) return true;
    // Greek and Coptic
    if (cp >= 0x370 && cp <= 0x3FF) return true;
    // Cyrillic
    if (cp >= 0x400 && cp <= 0x4FF) return true;
    // Arabic
    if (cp >= 0x600 && cp <= 0x6FF) return true;
    // Devanagari
    if (cp >= 0x900 && cp <= 0x97F) return true;
    // CJK Unified Ideographs
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    // CJK Extension A
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    // Hangul Syllables
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
    // Hiragana
    if (cp >= 0x3040 && cp <= 0x309F) return true;
    // Katakana
    if (cp >= 0x30A0 && cp <= 0x30FF) return true;
    // CJK Compatibility Ideographs
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    // Fullwidth Latin
    if (cp >= 0xFF21 && cp <= 0xFF3A) return true;
    if (cp >= 0xFF41 && cp <= 0xFF5A) return true;
    // Thai
    if (cp >= 0x0E01 && cp <= 0x0E3A) return true;
    // Hebrew
    if (cp >= 0x05D0 && cp <= 0x05EA) return true;
    // General: Other Letter (Lo) ranges for common scripts
    if (cp >= 0x1100 && cp <= 0x11FF) return true;  // Hangul Jamo
    if (cp >= 0x2E80 && cp <= 0x2EFF) return true;  // CJK Radicals
    if (cp >= 0x3000 && cp <= 0x303F) return true;  // CJK Symbols
    // Catch-all: treat non-ASCII, non-digit, non-whitespace, non-punct as letter
    // if it's in certain high ranges. This is approximate but sufficient.
    if (cp >= 0x1000 && cp <= 0xFFFD &&
        !is_letter(cp)) {
        // Avoid infinite recursion — this branch only catches remaining ranges
        return false;
    }
    return false;
}

static bool is_digit(uint32_t cp) {
    return cp >= '0' && cp <= '9';
}

static bool is_mark(uint32_t cp) {
    // Unicode Mark category (Mn, Mc, Me) — combining marks.
    // Common ranges for diacritics used in many languages.
    if (cp >= 0x0300 && cp <= 0x036F) return true;   // Combining Diacritical Marks
    if (cp >= 0x0591 && cp <= 0x05BD) return true;   // Hebrew accents
    if (cp >= 0x0610 && cp <= 0x061A) return true;   // Arabic
    if (cp >= 0x064B && cp <= 0x065F) return true;   // Arabic
    if (cp >= 0x0900 && cp <= 0x0903) return true;   // Devanagari
    if (cp >= 0x093A && cp <= 0x094F) return true;   // Devanagari
    if (cp >= 0x0E31 && cp == 0x0E31) return true;   // Thai
    if (cp >= 0x0E34 && cp <= 0x0E3A) return true;   // Thai
    if (cp >= 0xFE20 && cp <= 0xFE2F) return true;   // Combining Half Marks
    if (cp >= 0x20D0 && cp <= 0x20FF) return true;   // Combining for Symbols
    return false;
}

static bool is_whitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == '\f' || cp == '\v' ||
           cp == 0x00A0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

static bool is_newline(uint32_t cp) {
    return cp == '\n' || cp == '\r';
}

// ─── Pre-tokenizer ─────────────────────────────────────────────────────
// Matches the Qwen3.5 pattern:
//   (?:'[sStTmMdD]|...) |
//   [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ |
//   \p{N} |
//   ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]* |
//   \s*[\r\n]+ |
//   \s+(?!\S) |
//   \s+

std::vector<std::string> Tokenizer::pre_tokenize(const std::string & text) const {
    std::vector<std::string> pieces;
    const char * s = text.c_str();
    const size_t len = text.size();
    size_t pos = 0;

    auto peek_cp = [&](size_t p, int * cplen) -> uint32_t {
        if (p >= len) { *cplen = 0; return 0; }
        return utf8_decode(s + p, cplen);
    };

    while (pos < len) {
        size_t start = pos;
        int cplen = 0;
        uint32_t cp = peek_cp(pos, &cplen);

        // Pattern 1: English contractions 's 't 're 've 'm 'll 'd
        if (cp == '\'') {
            size_t save = pos;
            pos++;
            bool matched = false;
            if (pos < len) {
                char c = s[pos] | 0x20;  // lowercase
                if (c == 's' || c == 't' || c == 'm' || c == 'd') {
                    pos++;
                    matched = true;
                } else if (c == 'r' && pos + 1 < len && (s[pos+1] | 0x20) == 'e') {
                    pos += 2;
                    matched = true;
                } else if (c == 'v' && pos + 1 < len && (s[pos+1] | 0x20) == 'e') {
                    pos += 2;
                    matched = true;
                } else if (c == 'l' && pos + 1 < len && (s[pos+1] | 0x20) == 'l') {
                    pos += 2;
                    matched = true;
                }
            }
            if (matched) {
                pieces.push_back(text.substr(start, pos - start));
                continue;
            }
            pos = save;  // reset, try other patterns
        }

        // Pattern 2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        {
            size_t p = pos;
            int cl = 0;
            uint32_t c = peek_cp(p, &cl);
            // Optional leading non-letter, non-digit, non-newline char
            if (cl > 0 && !is_newline(c) && !is_letter(c) && !is_digit(c)) {
                p += cl;
                c = peek_cp(p, &cl);
            }
            // One or more letter/mark chars
            if (cl > 0 && (is_letter(c) || is_mark(c))) {
                while (cl > 0 && (is_letter(c) || is_mark(c))) {
                    p += cl;
                    c = peek_cp(p, &cl);
                }
                pieces.push_back(text.substr(pos, p - pos));
                pos = p;
                continue;
            }
        }

        // Pattern 3: \p{N}  (single digit)
        if (is_digit(cp)) {
            pos += cplen;
            pieces.push_back(text.substr(start, pos - start));
            continue;
        }

        // Pattern 4: ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        {
            size_t p = pos;
            int cl = 0;
            uint32_t c = peek_cp(p, &cl);
            // Optional leading space
            if (c == ' ') {
                p += cl;
                c = peek_cp(p, &cl);
            }
            // One or more non-whitespace, non-letter, non-mark, non-digit
            size_t punc_start = p;
            while (cl > 0 && !is_whitespace(c) && !is_letter(c) &&
                   !is_mark(c) && !is_digit(c)) {
                p += cl;
                c = peek_cp(p, &cl);
            }
            if (p > punc_start) {
                // Trailing newlines
                while (cl > 0 && is_newline(c)) {
                    p += cl;
                    c = peek_cp(p, &cl);
                }
                pieces.push_back(text.substr(pos, p - pos));
                pos = p;
                continue;
            }
        }

        // Pattern 5: \s*[\r\n]+
        if (is_whitespace(cp)) {
            size_t p = pos;
            int cl = 0;
            uint32_t c = peek_cp(p, &cl);
            // Consume leading whitespace
            while (cl > 0 && is_whitespace(c) && !is_newline(c)) {
                p += cl;
                c = peek_cp(p, &cl);
            }
            if (cl > 0 && is_newline(c)) {
                while (cl > 0 && is_newline(c)) {
                    p += cl;
                    c = peek_cp(p, &cl);
                }
                pieces.push_back(text.substr(pos, p - pos));
                pos = p;
                continue;
            }
            // Pattern 6: \s+(?!\S) — trailing whitespace not followed by non-ws
            // Pattern 7: \s+ — general whitespace
            p = pos;
            c = peek_cp(p, &cl);
            while (cl > 0 && is_whitespace(c)) {
                p += cl;
                c = peek_cp(p, &cl);
            }
            pieces.push_back(text.substr(pos, p - pos));
            pos = p;
            continue;
        }

        // Fallback: single character (shouldn't normally hit this)
        pos += cplen > 0 ? cplen : 1;
        pieces.push_back(text.substr(start, pos - start));
    }

    return pieces;
}

// ─── BPE encoding ──────────────────────────────────────────────────────

// Encode a single pre-tokenized piece using BPE merges.
std::vector<int32_t> Tokenizer::bpe_encode_piece(const std::string & piece) const {
    if (piece.empty()) return {};

    // Start with individual bytes/chars as initial symbols.
    // Each symbol is a string that we look up in the vocab.
    std::vector<std::string> symbols;

    // Try to find the piece as a single token first (common for special tokens).
    auto it = token_to_id_.find(piece);
    if (it != token_to_id_.end()) {
        return { it->second };
    }

    // Split into individual UTF-8 bytes as initial BPE symbols.
    // Qwen models use byte-level BPE where each byte maps to a token.
    for (size_t i = 0; i < piece.size(); ) {
        int clen = utf8_len((uint8_t)piece[i]);
        std::string sym = piece.substr(i, clen);
        // Check if this char exists as a token
        auto sit = token_to_id_.find(sym);
        if (sit != token_to_id_.end()) {
            symbols.push_back(sym);
        } else {
            // Byte-fallback: try individual bytes
            for (int b = 0; b < clen; b++) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>",
                              (unsigned)(uint8_t)piece[i + b]);
                symbols.push_back(buf);
            }
        }
        i += clen;
    }

    if (symbols.size() <= 1) {
        if (symbols.empty()) return {};
        auto sit = token_to_id_.find(symbols[0]);
        if (sit != token_to_id_.end()) return { sit->second };
        return {};  // unknown token
    }

    // Iteratively merge the highest-priority pair until no more merges apply.
    while (symbols.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_pos = SIZE_MAX;

        for (size_t i = 0; i + 1 < symbols.size(); i++) {
            std::string pair = symbols[i] + " " + symbols[i + 1];
            auto mit = merge_rank_.find(pair);
            if (mit != merge_rank_.end() && mit->second < best_rank) {
                best_rank = mit->second;
                best_pos = i;
            }
        }

        if (best_pos == SIZE_MAX) break;  // no more merges

        // Merge the best pair.
        symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
        symbols.erase(symbols.begin() + best_pos + 1);
    }

    // Convert merged symbols to token IDs.
    std::vector<int32_t> ids;
    ids.reserve(symbols.size());
    for (const auto & sym : symbols) {
        auto sit = token_to_id_.find(sym);
        if (sit != token_to_id_.end()) {
            ids.push_back(sit->second);
        } else {
            // Unknown symbol — emit byte-fallback tokens if available.
            for (size_t i = 0; i < sym.size(); i++) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>",
                              (unsigned)(uint8_t)sym[i]);
                auto bit = token_to_id_.find(buf);
                if (bit != token_to_id_.end()) {
                    ids.push_back(bit->second);
                }
                // else: truly unknown byte — skip (shouldn't happen)
            }
        }
    }

    return ids;
}

// ─── Public API ─────────────────────────────────────────────────────────

bool Tokenizer::load_from_gguf(const char * model_path) {
    struct gguf_init_params params = { /*.no_alloc=*/ true, /*.ctx=*/ nullptr };
    struct gguf_context * gctx = gguf_init_from_file(model_path, params);
    if (!gctx) {
        std::fprintf(stderr, "[tokenizer] failed to open GGUF: %s\n", model_path);
        return false;
    }

    // Load token strings.
    int tokens_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (tokens_key < 0) {
        std::fprintf(stderr, "[tokenizer] missing tokenizer.ggml.tokens in %s\n",
                     model_path);
        gguf_free(gctx);
        return false;
    }

    const int n_vocab = gguf_get_arr_n(gctx, tokens_key);
    id_to_token_.resize(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        const char * tok = gguf_get_arr_str(gctx, tokens_key, i);
        id_to_token_[i] = tok ? tok : "";
        token_to_id_[id_to_token_[i]] = i;
    }

    // Load merge table.
    int merges_key = gguf_find_key(gctx, "tokenizer.ggml.merges");
    if (merges_key >= 0) {
        const int n_merges = gguf_get_arr_n(gctx, merges_key);
        for (int i = 0; i < n_merges; i++) {
            const char * merge = gguf_get_arr_str(gctx, merges_key, i);
            if (merge) {
                merge_rank_[merge] = i;
            }
        }
    }

    // Detect pre-tokenizer type.
    int pre_key = gguf_find_key(gctx, "tokenizer.ggml.pre");
    if (pre_key >= 0) {
        const char * pre = gguf_get_val_str(gctx, pre_key);
        if (pre && std::strcmp(pre, "qwen35") == 0) {
            pre_type_ = PreTokenizer::QWEN35;
        } else {
            pre_type_ = PreTokenizer::QWEN2;
        }
    }

    // Load special token IDs.
    auto get_i32 = [&](const char * key) -> int32_t {
        int k = gguf_find_key(gctx, key);
        if (k < 0) return -1;
        return (int32_t)gguf_get_val_u32(gctx, k);
    };

    bos_id_ = get_i32("tokenizer.ggml.bos_token_id");
    eos_id_ = get_i32("tokenizer.ggml.eos_token_id");
    eos_chat_id_ = get_i32("tokenizer.ggml.eot_token_id");
    if (eos_chat_id_ < 0) {
        // Qwen3 uses <|im_end|> as EOT.
        auto eot = token_to_id_.find("<|im_end|>");
        if (eot != token_to_id_.end()) eos_chat_id_ = eot->second;
    }

    gguf_free(gctx);

    std::fprintf(stderr, "[tokenizer] loaded vocab=%d merges=%zu bos=%d eos=%d eot=%d pre=%s\n",
                 n_vocab, merge_rank_.size(), bos_id_, eos_id_, eos_chat_id_,
                 pre_type_ == PreTokenizer::QWEN35 ? "qwen35" : "qwen2");
    return true;
}

std::vector<int32_t> Tokenizer::encode(const std::string & text) const {
    std::vector<std::string> pieces = pre_tokenize(text);
    std::vector<int32_t> ids;
    for (const auto & piece : pieces) {
        auto piece_ids = bpe_encode_piece(piece);
        ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
    }
    return ids;
}

std::string Tokenizer::token_text(int32_t id) const {
    if (id < 0 || id >= (int32_t)id_to_token_.size()) return "";
    const std::string & tok = id_to_token_[id];

    // Handle byte-fallback tokens like <0xNN>.
    if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' &&
        tok[2] == 'x' && tok[5] == '>') {
        unsigned val = 0;
        if (std::sscanf(tok.c_str(), "<0x%02X>", &val) == 1) {
            return std::string(1, (char)(uint8_t)val);
        }
    }

    return tok;
}

std::string Tokenizer::decode(const std::vector<int32_t> & ids) const {
    std::string result;
    for (int32_t id : ids) {
        result += token_text(id);
    }
    return result;
}

int32_t Tokenizer::token_to_id(const std::string & token) const {
    auto it = token_to_id_.find(token);
    return it != token_to_id_.end() ? it->second : -1;
}

}  // namespace dflash27b
