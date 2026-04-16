Running Qwen 3.5 in agentic setups (coding agents, function calling loops)? Here are the 4 bugs that make tool calling   break, which servers have fixed what, and what you still need to do client-side.
                                                                                                                          ---
  The Bugs

  1. XML tool calls leak as plain text. Qwen 3.5 emits tool calls as
  <function=bash><parameter=command>ls</parameter></function>. When the server fails to parse this (especially when text
   precedes the XML, or thinking is enabled), it arrives as raw text with finish_reason: stop. Your agent never executes
   it.

  - llama.cpp: https://github.com/ggml-org/llama.cpp/issues/20260 -- peg-native parser fails when text precedes
  <tool_call>. Open.
  - llama.cpp: https://github.com/ggml-org/llama.cpp/issues/20837 -- tool calls emitted inside thinking block. Open.
  - Ollama: https://github.com/ollama/ollama/issues/14745 -- still sometimes prints tool calls as text (post-fix). Open.
  - vLLM: https://github.com/vllm-project/vllm/issues/35266 -- streaming drops opening { brace.
  https://github.com/vllm-project/vllm/issues/36769 -- ValueError in parser.

  2. <think> tags leak into text and poison context. llama.cpp forces thinking=1 internally regardless of
  enable_thinking: false. Tags accumulate across turns and destroy multi-turn sessions.

  - llama.cpp: https://github.com/ggml-org/llama.cpp/issues/20182 -- still open on b8664.
  https://github.com/ggml-org/llama.cpp/issues/20409 confirms across 27B/9B/2B.
  - Ollama had unclosed </think> bug (https://github.com/ollama/ollama/issues/14493), fixed in v0.17.6.

  3. Wrong finish_reason. Server sends "stop" when tool calls are present. Agent treats it as final answer.

  4. Non-standard finish_reason. Some servers return "eos_token", "", or null. Most frameworks crash on the unknown
  value before checking if tool calls exist.

  ---
  Server Status (April 2026)

  ┌─────────┬─────────────────────────────────────────┬──────────────────────────────────────────────┬─────────────┐
  │         │               XML parsing               │                  Think leak                  │ finish_reas │
  │         │                                         │                                              │     on      │
  ├─────────┼─────────────────────────────────────────┼──────────────────────────────────────────────┼─────────────┤
  │ LM      │ Best local option (fixed in https://lms │                                              │ Usually     │
  │ Studio  │ tudio.ai/changelog/lmstudio-v0.4.7)     │ Improved                                     │ correct     │
  │ 0.4.9   │                                         │                                              │             │
  ├─────────┼─────────────────────────────────────────┼──────────────────────────────────────────────┼─────────────┤
  │ vLLM    │ Works (--tool-call-parser qwen3_coder), │ Fixed                                        │ Usually     │
  │ 0.19.0  │  streaming bugs                         │                                              │ correct     │
  ├─────────┼─────────────────────────────────────────┼──────────────────────────────────────────────┼─────────────┤
  │ Ollama  │ Improved since https://github.com/ollam │ Fixed                                        │ Sometimes   │
  │ 0.20.2  │ a/ollama/issues/14493, still flaky      │                                              │ wrong       │
  ├─────────┼─────────────────────────────────────────┼──────────────────────────────────────────────┼─────────────┤
  │ llama.c │ Parser exists, fails with thinking      │ Broken (https://github.com/ggml-org/llama.cp │ Wrong when  │
  │ pp      │ enabled                                 │ p/issues/20182)                              │ parser      │
  │ b8664   │                                         │                                              │ fails       │
  └─────────┴─────────────────────────────────────────┴──────────────────────────────────────────────┴─────────────┘

  ---
  What To Do

  Use Unsloth GGUFs. Stock Qwen 3.5 Jinja templates have https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/4
  (|items filter fails on tool args). Unsloth ships 21 template fixes.

  Add a client-side safety net. 3 small functions that catch what servers miss:

  import re, json, uuid

  # 1. Parse Qwen XML tool calls from text content
  def parse_qwen_xml_tools(text):
      results = []
      for m in re.finditer(r'<function=([\w.-]+)>([\s\S]*?)</function>', text):
          args = {}
          for p in re.finditer(r'<parameter=([\w.-]+)>([\s\S]*?)</parameter>', m.group(2)):
              k, v = p.group(1).strip(), p.group(2).strip()
              try: v = json.loads(v)
              except: pass
              args[k] = v
          results.append({"id": f"call_{uuid.uuid4().hex[:24]}", "name": m.group(1), "args": args})
      return results

  # 2. Strip leaked think tags
  def strip_think_tags(text):
      return re.sub(r'<think>[\s\S]*?</think>', '', re.sub(r'^</think>\s*', '', text)).strip()

  # 3. Fix finish_reason
  def fix_stop_reason(message):
      has_tools = any(b.get("type") == "tool_call" for b in message.get("content", []))
      if has_tools and message.get("stop_reason") in ("stop", "error", "eos_token", "", None):
          message["stop_reason"] = "tool_use"

  Set compat flags (Pi SDK / OpenAI-compatible clients):
  - thinkingFormat: "qwen" -- sends enable_thinking instead of OpenAI reasoning format
  - maxTokensField: "max_tokens" -- not max_completion_tokens
  - supportsDeveloperRole: false -- use system role, not developer
  - supportsStrictMode: false -- don't send strict: true on tool schemas

  ---
  The model is smart. It's the plumbing that breaks.