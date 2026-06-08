#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace dflash::common {

// Scrape local agent history into text chunks for the Spark calibration
// bootstrap. Reads Claude Code transcripts (~/.claude/projects/<project>/*.jsonl)
// and Codex rollouts (~/.codex/sessions/**/rollout-*.jsonl); the directories are
// overridable via DFLASH_SPARK_CLAUDE_DIR / DFLASH_SPARK_CODEX_DIR. Returns up to
// max_chunks chunks of ~chunk_chars characters each (>= min_chars), sampled
// across sessions. Empty if no history is present.
std::vector<std::string> spark_scrape_corpus(std::size_t max_chunks,
                                             std::size_t chunk_chars,
                                             std::size_t min_chars);

}  // namespace dflash::common
