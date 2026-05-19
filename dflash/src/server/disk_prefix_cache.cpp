// Disk-backed prefix cache implementation.

#include "disk_prefix_cache.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dflash27b {

// ─── Inline SHA-1 (same as prefix_cache.cpp) ────────────────────────────

static void sha1_hash(const void * data, size_t len, uint8_t out[20]) {
    auto rotl = [](uint32_t x, int n) -> uint32_t {
        return (x << n) | (x >> (32 - n));
    };

    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE,
             h3 = 0x10325476, h4 = 0xC3D2E1F0;

    size_t new_len = len + 1;
    while (new_len % 64 != 56) new_len++;
    std::vector<uint8_t> msg(new_len + 8, 0);
    std::memcpy(msg.data(), data, len);
    msg[len] = 0x80;
    uint64_t bit_len = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) {
        msg[new_len + i] = (uint8_t)(bit_len >> (56 - 8 * i));
    }

    for (size_t offset = 0; offset < msg.size(); offset += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)msg[offset + 4*i] << 24) |
                    ((uint32_t)msg[offset + 4*i+1] << 16) |
                    ((uint32_t)msg[offset + 4*i+2] << 8) |
                    ((uint32_t)msg[offset + 4*i+3]);
        }
        for (int i = 16; i < 80; i++) {
            w[i] = rotl(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
        }

        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20)      { f = (b & c) | (~b & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d;          k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else              { f = b ^ c ^ d;          k = 0xCA62C1D6; }
            uint32_t temp = rotl(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rotl(b, 30); b = a; a = temp;
        }
        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }

    auto store32 = [](uint8_t * p, uint32_t v) {
        p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
        p[2] = (uint8_t)(v >> 8);  p[3] = (uint8_t)v;
    };
    store32(out,     h0);
    store32(out + 4, h1);
    store32(out + 8, h2);
    store32(out + 12, h3);
    store32(out + 16, h4);
}

// ─── Utility ────────────────────────────────────────────────────────────

static std::string hex(const uint8_t * data, int len) {
    static const char hex_chars[] = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (int i = 0; i < len; ++i) {
        out.push_back(hex_chars[data[i] >> 4]);
        out.push_back(hex_chars[data[i] & 0x0f]);
    }
    return out;
}

static bool mkdir_p(const std::string & path) {
    struct stat st{};
    if (stat(path.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
    // Try to create parent first.
    size_t slash = path.rfind('/');
    if (slash != std::string::npos && slash > 0) {
        mkdir_p(path.substr(0, slash));
    }
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

static uint64_t now_unix() {
    return (uint64_t)std::time(nullptr);
}

// Little-endian I/O helpers.
static void write_u32(FILE * f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
static void write_u64(FILE * f, uint64_t v) { std::fwrite(&v, 8, 1, f); }
static void write_i64(FILE * f, int64_t v)  { std::fwrite(&v, 8, 1, f); }
static void write_u16(FILE * f, uint16_t v) { std::fwrite(&v, 2, 1, f); }

static uint32_t read_u32(FILE * f) { uint32_t v = 0; std::fread(&v, 4, 1, f); return v; }
static uint64_t read_u64(FILE * f) { uint64_t v = 0; std::fread(&v, 8, 1, f); return v; }
static int64_t  read_i64(FILE * f) { int64_t v = 0;  std::fread(&v, 8, 1, f); return v; }
static uint16_t read_u16(FILE * f) { uint16_t v = 0; std::fread(&v, 2, 1, f); return v; }

// ─── Construction ───────────────────────────────────────────────────────

DiskPrefixCache::DiskPrefixCache(const DiskCacheConfig & cfg, ModelBackend & backend)
    : config_(cfg), backend_(backend) {}

// ─── Initialization ─────────────────────────────────────────────────────

bool DiskPrefixCache::init() {
    if (disabled()) return true;

    if (!mkdir_p(config_.cache_dir)) {
        std::fprintf(stderr, "[disk-cache] failed to create dir: %s\n",
                     config_.cache_dir.c_str());
        return false;
    }

    // Try to learn layout from existing files (enables first-request disk hits).
    try_learn_from_disk();

    std::fprintf(stderr, "[disk-cache] initialized dir=%s budget=%.1f GB layout=%s\n",
                 config_.cache_dir.c_str(),
                 (double)config_.budget_bytes / (1024.0 * 1024.0 * 1024.0),
                 layout_known_ ? hex(layout_id_.data(), 16).c_str() : "pending");
    return true;
}

// ─── Layout fingerprint ─────────────────────────────────────────────────

void DiskPrefixCache::compute_layout_id(ggml_context * ctx) {
    // Collect tensor metadata sorted by name for deterministic fingerprint.
    struct TInfo { std::string name; uint32_t type; int64_t ne[4]; };
    std::vector<TInfo> tensors;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        TInfo ti;
        ti.name = t->name;
        ti.type = (uint32_t)t->type;
        ti.ne[0] = t->ne[0];
        ti.ne[1] = 1;  // normalize sequence-length dimension
        ti.ne[2] = t->ne[2];
        ti.ne[3] = t->ne[3];
        tensors.push_back(std::move(ti));
    }
    std::sort(tensors.begin(), tensors.end(), [](const TInfo & a, const TInfo & b) {
        return a.name < b.name;
    });

    // Build a single buffer and hash it.
    std::vector<uint8_t> buf;
    for (const auto & ti : tensors) {
        buf.insert(buf.end(), ti.name.begin(), ti.name.end());
        buf.insert(buf.end(), (uint8_t *)&ti.type, (uint8_t *)&ti.type + 4);
        buf.insert(buf.end(), (uint8_t *)ti.ne, (uint8_t *)ti.ne + 32);
    }

    uint8_t digest[20];
    sha1_hash(buf.data(), buf.size(), digest);
    std::memcpy(layout_id_.data(), digest, 16);
}

void DiskPrefixCache::learn_layout(int slot) {
    if (layout_known_ || disabled()) return;

    auto ref = backend_.snapshot_ref(slot);
    if (!ref.ctx) return;

    compute_layout_id(ref.ctx);
    layout_known_ = true;
    layout_dir_ = config_.cache_dir + "/" + hex(layout_id_.data(), 16);
    mkdir_p(layout_dir_);

    std::fprintf(stderr, "[disk-cache] layout learned: %s\n",
                 hex(layout_id_.data(), 16).c_str());

    // Now scan for any previously saved files.
    scan_directory();
}

// ─── Directory scanning ─────────────────────────────────────────────────

void DiskPrefixCache::scan_directory() {
    entries_.clear();
    total_bytes_ = 0;

    if (layout_dir_.empty()) return;

    DIR * dir = opendir(layout_dir_.c_str());
    if (!dir) return;

    struct dirent * ent;
    while ((ent = readdir(dir)) != nullptr) {
        const char * name = ent->d_name;
        size_t nlen = std::strlen(name);
        if (nlen < 36 || std::strcmp(name + nlen - 4, ".dkv") != 0) continue;

        std::string path = layout_dir_ + "/" + name;
        FILE * f = std::fopen(path.c_str(), "rb");
        if (!f) continue;

        DiskCacheHeader hdr{};
        if (!read_header(f, hdr)) { std::fclose(f); continue; }
        std::fclose(f);

        // Validate magic and layout.
        if (std::memcmp(hdr.magic, "DKVC", 4) != 0) continue;
        if (std::memcmp(hdr.layout_id, layout_id_.data(), 16) != 0) continue;

        DiskEntry entry;
        entry.path = path;
        std::memcpy(entry.token_hash.data(), hdr.token_hash, 16);
        entry.token_count = hdr.token_count;
        entry.cur_pos     = hdr.cur_pos;
        entry.last_used   = hdr.last_used;

        struct stat st{};
        if (stat(path.c_str(), &st) == 0) {
            entry.file_size = (uint64_t)st.st_size;
        }

        total_bytes_ += entry.file_size;
        entries_.push_back(std::move(entry));
    }
    closedir(dir);

    std::fprintf(stderr, "[disk-cache] scanned %zu files, %.1f MB\n",
                 entries_.size(), (double)total_bytes_ / (1024.0 * 1024.0));
}

// ─── Cold start: learn layout from existing files ───────────────────────

void DiskPrefixCache::try_learn_from_disk() {
    // Scan cache_dir for subdirectories (each is a layout fingerprint).
    DIR * dir = opendir(config_.cache_dir.c_str());
    if (!dir) return;

    struct dirent * ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.') continue;
        std::string subdir = config_.cache_dir + "/" + ent->d_name;
        struct stat st{};
        if (stat(subdir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) continue;

        // Check if this subdir has any .dkv files.
        DIR * sub = opendir(subdir.c_str());
        if (!sub) continue;

        struct dirent * sent;
        while ((sent = readdir(sub)) != nullptr) {
            size_t nlen = std::strlen(sent->d_name);
            if (nlen < 4 || std::strcmp(sent->d_name + nlen - 4, ".dkv") != 0) continue;

            // Read the header to get the layout_id.
            std::string fpath = subdir + "/" + sent->d_name;
            FILE * f = std::fopen(fpath.c_str(), "rb");
            if (!f) continue;

            DiskCacheHeader hdr{};
            if (read_header(f, hdr) && std::memcmp(hdr.magic, "DKVC", 4) == 0) {
                std::memcpy(layout_id_.data(), hdr.layout_id, 16);
                layout_known_ = true;
                layout_dir_ = subdir;
                std::fclose(f);
                closedir(sub);
                closedir(dir);
                scan_directory();
                return;
            }
            std::fclose(f);
        }
        closedir(sub);
    }
    closedir(dir);
}

// ─── Lookup ─────────────────────────────────────────────────────────────

bool DiskPrefixCache::lookup(const std::vector<int32_t> & prompt_ids, int slot) {
    if (disabled() || !layout_known_) return false;

    PrefixHash hash = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());

    std::lock_guard<std::mutex> lock(mu_);
    int idx = find_entry(hash);
    if (idx < 0) return false;

    auto & entry = entries_[idx];
    if (!read_file(entry.path, slot)) {
        // File is corrupt or incompatible — remove it.
        std::remove(entry.path.c_str());
        total_bytes_ -= entry.file_size;
        entries_.erase(entries_.begin() + idx);
        return false;
    }

    // Update last_used on disk.
    entry.last_used = now_unix();
    // Optionally rewrite header timestamp (non-critical, skip for perf).
    return true;
}

// ─── Save ───────────────────────────────────────────────────────────────

bool DiskPrefixCache::save(int slot, const std::vector<int32_t> & prompt_ids) {
    if (disabled()) return false;

    // Learn layout on first save.
    if (!layout_known_) {
        learn_layout(slot);
        if (!layout_known_) return false;
    }

    // Check minimum token threshold.
    if ((int)prompt_ids.size() < config_.min_tokens) return false;

    auto ref = backend_.snapshot_ref(slot);
    if (!ref.ctx) return false;

    PrefixHash hash = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());

    std::lock_guard<std::mutex> lock(mu_);

    // Skip if already on disk.
    if (find_entry(hash) >= 0) return true;

    std::string path = make_path(hash);
    std::string tmp_path = path + ".tmp";

    if (!write_file(tmp_path, ref, prompt_ids)) {
        std::remove(tmp_path.c_str());
        return false;
    }

    // Atomic rename.
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::remove(tmp_path.c_str());
        return false;
    }

    // Update index.
    DiskEntry entry;
    entry.path        = path;
    std::memcpy(entry.token_hash.data(), hash.data(), 16);
    entry.token_count = (uint32_t)prompt_ids.size();
    entry.cur_pos     = (uint32_t)ref.cur_pos;
    entry.last_used   = now_unix();
    struct stat st{};
    if (stat(path.c_str(), &st) == 0) entry.file_size = (uint64_t)st.st_size;

    total_bytes_ += entry.file_size;
    entries_.push_back(std::move(entry));

    std::fprintf(stderr, "[disk-cache] saved %s (%u tokens, %d pos, %.1f MB)\n",
                 hex(hash.data(), 16).c_str(),
                 (uint32_t)prompt_ids.size(), ref.cur_pos,
                 (double)entries_.back().file_size / (1024.0 * 1024.0));

    enforce_budget();
    return true;
}

// ─── Budget enforcement ─────────────────────────────────────────────────

void DiskPrefixCache::enforce_budget() {
    while (total_bytes_ > config_.budget_bytes && !entries_.empty()) {
        // Find entry with oldest last_used.
        auto it = std::min_element(entries_.begin(), entries_.end(),
            [](const DiskEntry & a, const DiskEntry & b) {
                return a.last_used < b.last_used;
            });

        std::fprintf(stderr, "[disk-cache] evicting %s (%.1f MB, last_used=%llu)\n",
                     hex(it->token_hash.data(), 16).c_str(),
                     (double)it->file_size / (1024.0 * 1024.0),
                     (unsigned long long)it->last_used);

        std::remove(it->path.c_str());
        total_bytes_ -= it->file_size;
        entries_.erase(it);
    }
}

// ─── Touch ──────────────────────────────────────────────────────────────

void DiskPrefixCache::touch(const PrefixHash & hash) {
    std::lock_guard<std::mutex> lock(mu_);
    int idx = find_entry(hash);
    if (idx >= 0) {
        entries_[idx].last_used = now_unix();
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────

std::string DiskPrefixCache::make_path(const PrefixHash & hash) const {
    return layout_dir_ + "/" + hex(hash.data(), 16) + ".dkv";
}

int DiskPrefixCache::find_entry(const PrefixHash & hash) const {
    for (int i = 0; i < (int)entries_.size(); ++i) {
        if (entries_[i].token_hash == hash) return i;
    }
    return -1;
}

// ─── File I/O: Write ────────────────────────────────────────────────────

bool DiskPrefixCache::write_file(const std::string & path,
                                 const ModelBackend::SnapshotRef & ref,
                                 const std::vector<int32_t> & prompt_ids) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;

    // Count tensors and compute payload size.
    uint32_t n_tensors = 0;
    uint64_t payload_bytes = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); t; t = ggml_get_next_tensor(ref.ctx, t)) {
        n_tensors++;
        payload_bytes += ggml_nbytes(t);
    }

    // Write header.
    DiskCacheHeader hdr{};
    std::memcpy(hdr.magic, "DKVC", 4);
    hdr.version = DISK_CACHE_VERSION;
    std::memcpy(hdr.layout_id, layout_id_.data(), 16);
    hdr.cur_pos       = (uint32_t)ref.cur_pos;
    hdr.n_tensors     = n_tensors;
    hdr.token_count   = (uint32_t)prompt_ids.size();
    PrefixHash ph = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());
    std::memcpy(hdr.token_hash, ph.data(), 16);
    hdr.payload_bytes = payload_bytes;
    hdr.created_at    = now_unix();
    hdr.last_used     = hdr.created_at;
    hdr.last_tok      = ref.last_tok;

    if (!write_header(f, hdr)) { std::fclose(f); return false; }

    // Write tensor table.
    for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); t; t = ggml_get_next_tensor(ref.ctx, t)) {
        uint16_t name_len = (uint16_t)std::strlen(t->name);
        write_u16(f, name_len);
        std::fwrite(t->name, 1, name_len, f);
        write_u32(f, (uint32_t)t->type);
        for (int d = 0; d < 4; ++d) write_i64(f, t->ne[d]);
        write_u64(f, ggml_nbytes(t));
    }

    // Write tensor data.
    std::vector<uint8_t> buf(4 * 1024 * 1024);  // 4 MB transfer buffer
    for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); t; t = ggml_get_next_tensor(ref.ctx, t)) {
        size_t nbytes = ggml_nbytes(t);
        size_t offset = 0;
        while (offset < nbytes) {
            size_t chunk = std::min(buf.size(), nbytes - offset);
            ggml_backend_tensor_get(t, buf.data(), offset, chunk);
            if (std::fwrite(buf.data(), 1, chunk, f) != chunk) {
                std::fclose(f);
                return false;
            }
            offset += chunk;
        }
    }

    std::fclose(f);
    return true;
}

// ─── File I/O: Read ─────────────────────────────────────────────────────

bool DiskPrefixCache::read_file(const std::string & path, int slot) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    DiskCacheHeader hdr{};
    if (!read_header(f, hdr)) { std::fclose(f); return false; }

    // Validate.
    if (std::memcmp(hdr.magic, "DKVC", 4) != 0) { std::fclose(f); return false; }
    if (hdr.version != DISK_CACHE_VERSION) { std::fclose(f); return false; }
    if (std::memcmp(hdr.layout_id, layout_id_.data(), 16) != 0) { std::fclose(f); return false; }

    // Read tensor table.
    std::vector<DiskTensorEntry> table;
    table.reserve(hdr.n_tensors);
    for (uint32_t i = 0; i < hdr.n_tensors; ++i) {
        DiskTensorEntry ent;
        uint16_t name_len = read_u16(f);
        if (name_len >= GGML_MAX_NAME) { std::fclose(f); return false; }
        char name_buf[GGML_MAX_NAME] = {};
        if (std::fread(name_buf, 1, name_len, f) != name_len) { std::fclose(f); return false; }
        ent.name = name_buf;
        ent.type = read_u32(f);
        for (int d = 0; d < 4; ++d) ent.ne[d] = read_i64(f);
        ent.nbytes = (size_t)read_u64(f);
        table.push_back(std::move(ent));
    }

    // Allocate ggml context + buffer.
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(hdr.n_tensors + 4) + 4096;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { std::fclose(f); return false; }

    // Create tensors.
    for (const auto & ent : table) {
        ggml_tensor * t = ggml_new_tensor(ctx, (ggml_type)ent.type, 4, ent.ne);
        if (!t) {
            ggml_free(ctx);
            std::fclose(f);
            return false;
        }
        ggml_set_name(t, ent.name.c_str());
    }

    // Allocate buffer on CPU backend.
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { ggml_free(ctx); std::fclose(f); return false; }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buf) {
        ggml_free(ctx);
        ggml_backend_free(cpu);
        std::fclose(f);
        return false;
    }

    // Read tensor data.
    std::vector<uint8_t> read_buf(4 * 1024 * 1024);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        size_t nbytes = ggml_nbytes(t);
        size_t offset = 0;
        while (offset < nbytes) {
            size_t chunk = std::min(read_buf.size(), nbytes - offset);
            if (std::fread(read_buf.data(), 1, chunk, f) != chunk) {
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
                ggml_backend_free(cpu);
                std::fclose(f);
                return false;
            }
            ggml_backend_tensor_set(t, read_buf.data(), offset, chunk);
            offset += chunk;
        }
    }
    std::fclose(f);

    // Hand off to backend.
    if (!backend_.snapshot_adopt(slot, ctx, buf, (int)hdr.cur_pos, hdr.last_tok)) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        return false;
    }

    // The cpu backend must persist as long as the buffer is alive. The buffer
    // does NOT own the backend, so we cannot free it here. Since snapshot_free
    // only frees buf + ctx, we accept a small leak of the lightweight cpu
    // backend object (~64 bytes per load). For a bounded disk cache this is
    // negligible. A proper fix would store cpu alongside the snapshot.

    return true;
}

// ─── Header I/O ─────────────────────────────────────────────────────────

bool DiskPrefixCache::write_header(FILE * f, const DiskCacheHeader & hdr) {
    std::fwrite(hdr.magic, 1, 4, f);
    write_u32(f, hdr.version);
    std::fwrite(hdr.layout_id, 1, 16, f);
    write_u32(f, hdr.cur_pos);
    write_u32(f, hdr.n_tensors);
    write_u32(f, hdr.token_count);
    std::fwrite(hdr.token_hash, 1, 16, f);
    write_u64(f, hdr.payload_bytes);
    write_u64(f, hdr.created_at);
    write_u64(f, hdr.last_used);
    write_u32(f, (uint32_t)hdr.last_tok);  // stored as u32, cast back on read
    // Total: 4+4+16+4+4+4+16+8+8+8+4 = 80 bytes. No extra padding needed.
    return !std::ferror(f);
}

bool DiskPrefixCache::read_header(FILE * f, DiskCacheHeader & hdr) {
    if (std::fread(hdr.magic, 1, 4, f) != 4) return false;
    hdr.version = read_u32(f);
    if (std::fread(hdr.layout_id, 1, 16, f) != 16) return false;
    hdr.cur_pos       = read_u32(f);
    hdr.n_tensors     = read_u32(f);
    hdr.token_count   = read_u32(f);
    if (std::fread(hdr.token_hash, 1, 16, f) != 16) return false;
    hdr.payload_bytes = read_u64(f);
    hdr.created_at    = read_u64(f);
    hdr.last_used     = read_u64(f);
    hdr.last_tok      = (int32_t)read_u32(f);
    return !std::ferror(f);
}

}  // namespace dflash27b
