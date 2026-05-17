// Chat template renderer for dflash27b native server.
//
// Renders chat messages (system/user/assistant/tool) into the model-specific
// token format. Hard-coded for supported architectures:
//   - Qwen3/3.5: <|im_start|>role\ncontent<|im_end|>\n
//   - Laguna: XML-style <|begin_of_sentence|><|User|>...<|Assistant|>

#pragma once

#include <string>
#include <vector>

namespace dflash27b {

// A single message in a chat conversation.
struct ChatMessage {
    std::string role;       // "system", "user", "assistant", "tool"
    std::string content;    // message text
    // Optional tool_call_id for tool result messages.
    std::string tool_call_id;
};

// Chat template format.
enum class ChatFormat {
    QWEN3,     // <|im_start|>role\n...<|im_end|>\n
    LAGUNA,    // <|begin_of_sentence|><|User|>...<|Assistant|>
};

// Render chat messages into the model-specific prompt string.
// The result is plain text ready to be tokenized.
//
// If `add_generation_prompt` is true, appends the assistant turn prefix
// at the end (so the model starts generating as assistant).
std::string render_chat_template(
    const std::vector<ChatMessage> & messages,
    ChatFormat format,
    bool add_generation_prompt = true);

// Detect the appropriate chat format for an architecture.
ChatFormat chat_format_for_arch(const std::string & arch);

}  // namespace dflash27b
