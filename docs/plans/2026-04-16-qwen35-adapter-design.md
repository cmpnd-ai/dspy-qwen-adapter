# Qwen35Adapter Design

**Date:** 2026-04-16
**Status:** Approved, pending implementation plan
**Target:** PyPI package `dspy-qwen35-adapter` exposing `Qwen35Adapter`

## Problem

Qwen 3.5 35B A3B is trained to emit tool calls as XML:
`<function=NAME><parameter=K>V</parameter></function>`. DSPy's stock adapters
(`ChatAdapter`, `JSONAdapter`, `XMLAdapter`) either expect a different output
protocol or rely on the inference server's tool-call parsing. Both paths fail
frequently with Qwen 3.5:

- `ChatAdapter` asks for `[[ ## next_tool_name ## ]]` delimited output; Qwen
  ignores the delimiters and emits its trained XML. Parsing fails.
- `JSONAdapter` with native function calling passes `tools=[]` to the server.
  Every major local inference engine (LM Studio, vLLM, Ollama, llama.cpp) has
  documented bugs translating Qwen's XML into OpenAI `tool_calls` — XML leaks
  as text with `finish_reason=stop`, `<think>` tags leak into content,
  non-standard `finish_reason` values. See `qwen-35-toolnotes-from-reddit.md`.

The community response — `ResilientReAct` wrappers that retry within a budget —
treats the symptom. A correctly shaped adapter should eliminate most retries.

## Goal

One drop-in DSPy adapter that makes `dspy.ReAct` reliable with Qwen 3.5 on any
OpenAI-compatible local endpoint, starting with LM Studio.

User-facing DX:

```python
import dspy
from dspy_qwen35_adapter import Qwen35Adapter

dspy.configure(
    lm=dspy.LM("openai/qwen3.5-35b-a3b", api_base="http://localhost:1234/v1"),
    adapter=Qwen35Adapter(),
)

react = dspy.ReAct("question -> answer", tools=[get_weather, search])
react(question="What's the weather in Tokyo?")
```

No other configuration, no prompts/parsers/flags to learn.

## Approach

**Text-native XML.** Do not pass `tools=[]` to the server. Prompt Qwen in its
trained XML format, parse tool calls out of `message.content` ourselves. Strip
`<think>` blocks. Ignore `finish_reason`. Rejected alternatives:

- *Hybrid native-first, text-fallback* — adds complexity without closing the
  full bug set (streaming brace drops, truncated content).
- *Native-only thin adapter* — relies on the server-side parsers the Reddit
  notes document as broken.

Text-native is the only path that is portable across OpenAI-compatible servers
because it depends on nothing server-specific.

## Architecture

One class, `Qwen35Adapter`, subclassing `dspy.adapters.base.Adapter` directly
(not `ChatAdapter`/`XMLAdapter` — those carry formatting conventions we
override wholesale).

Key stances:

- `use_native_function_calling = False`, hardcoded. Tools never pass to the
  server.
- System prompt uses Qwen's trained tool-description convention: a `<tools>`
  block with JSON schemas and a one-shot XML exemplar.
- `parse()` splits `message.content` into `next_thought` + `next_tool_name` +
  `next_tool_args`, matching ReAct's augmented signature.
- Defensive pre-parse: strip `<think>...</think>` and stray `</think>`
  openers; ignore `finish_reason`.
- Pure OpenAI chat-completions I/O. No server-specific flags.

## Package layout

```
dspy_qwen35_adapter/
  __init__.py              # exports Qwen35Adapter
  adapter.py               # the class
  parsing.py               # pure extractors, unit-testable
  prompts.py               # system prompt builder
tests/
  fixtures/traces/         # (raw, expected) pairs
  test_parsing.py
  test_adapter_integration.py
harness/
  scenarios.py             # 1/3/10-tool ReAct scenarios + golden trajectories
  run_eval.py              # CSV-producing live benchmark
  analyze.py               # CSV -> markdown summary
  traces/                  # captured live content (promoted into fixtures)
  results/                 # timestamped CSVs
pyproject.toml
```

`harness/` and `tests/` are excluded from the PyPI build. Users import only
`Qwen35Adapter` from the package root.

## Components

**`prompts.build_system_prompt(signature, tools) -> str`**
Task description, non-tool I/O field descriptions, a `<tools>` block listing
each tool (name, description, JSON parameter schema), a one-shot XML exemplar,
and the invariant: "emit exactly one `<function>...</function>` per turn, or
answer in plain text." Single f-string, no Jinja.

**`parsing.py` — four pure functions:**

- `strip_think(text) -> str` — removes `<think>...</think>`, leading
  `</think>` orphans, and an unclosed `<think>` (strips through end of string).
- `extract_tool_call(text) -> (name, args_dict) | None` — regex
  `<function=(\S+?)>([\s\S]*?)</function>`. Parameters extracted with
  `<parameter=(\S+?)>([\s\S]*?)</parameter>`. Values JSON-decoded with
  `json_repair` fallback to raw string.
- `split_thought_and_call(text) -> (thought, tool_call | None)` — splits at
  the first `<function=` boundary.
- `coerce_args_to_schema(args, tool) -> args` — best-effort string → int /
  float / bool coercion where the tool's JSON schema is unambiguous.

All pure, stateless, no LM dependency.

**`Qwen35Adapter` class — overrides:**

- `__init__` — forces `use_native_function_calling=False`, accepts
  `callbacks`, `native_response_types`, `strict_parse: bool = False`.
- `format_field_description` — delegates to `build_system_prompt` when a
  `list[Tool]` input field is present.
- `format_field_structure` — documents the Qwen XML output contract.
- `format_user_message_content` — standard text.
- `format_assistant_message_content` — replays prior turns as Qwen XML so
  multi-turn history stays in-distribution.
- `parse` — the integration point. If the signature has ReAct's
  `next_tool_name`/`next_tool_args` fields, uses `split_thought_and_call` to
  populate them. Otherwise routes to a plain-text output path.

Not overridden: `_call_preprocess`, `_call_postprocess` (those handle native
tool calling, which we bypass). Callbacks are inherited from the base class —
`__init_subclass__` auto-decorates `format()` and `parse()` with
`with_callbacks` (dspy/adapters/base.py:59-64).

## Data flow (one ReAct turn)

**Outbound:**

1. `ReAct.forward` → `Predict.forward` → `Qwen35Adapter.__call__`.
2. `_call_preprocess` no-op (we don't inject `tools=`).
3. `format()` builds messages: system (task + `<tools>` block + XML exemplar),
   user (question + trajectory).
4. LiteLLM → LM Studio `/v1/chat/completions`. No `tools` in the payload.

**Inbound:**

5. LM Studio returns `message.content` = raw Qwen output (may contain
   `<think>...`, plain text, `<function=...>`). `tool_calls` empty.
   `finish_reason` whatever — ignored.
6. DSPy's `base_lm._process_completion` wraps as `{"text": ..., "tool_calls":
   None}`.
7. `_call_postprocess` forwards `text` into `Qwen35Adapter.parse`.

**Parse:**

8. `strip_think(text)` → clean content.
9. `split_thought_and_call(cleaned)` → (thought, (name, args) | None).
10. Returns `{"next_thought": ..., "next_tool_name": ..., "next_tool_args":
    ...}`. If no tool call: `next_tool_name="finish"`, full cleaned text as
    `next_thought` (graceful-finish default) or `AdapterParseError` if
    `strict_parse=True`.

**Back in ReAct:**

11. Trajectory updated. Tool executed via `self.tools[name](**args)`.
    Observation stored. Loop continues.
12. Next turn's `format_assistant_message_content` replays the previous
    response as Qwen XML, keeping history in the model's trained format.

### Bug-to-defense mapping

| Reddit-documented bug | Defense |
|---|---|
| XML leaks as text with `finish_reason=stop` | We parse `content`; `finish_reason` ignored. |
| `<think>` tags leak and poison context | `strip_think` on every turn; replayed assistant messages use cleaned text. |
| Wrong / non-standard `finish_reason` | Unused. |
| Stock Jinja template bugs | Not invoked — we own the system prompt. |
| Server native tool parser failures (all servers) | Not invoked — we don't pass `tools=`. |

## Error handling

| Failure mode | Response |
|---|---|
| No `<function>` tag | Default: implicit finish (`next_tool_name="finish"`, cleaned text as `next_thought`). `strict_parse=True` raises `AdapterParseError`. |
| Malformed / nested / unclosed XML | Permissive regex; mismatches fall through to the "no tool call" path. No heroic recovery. |
| Unknown tool name | Returned verbatim. ReAct's existing `try/except` around tool execution logs and continues. |
| Args don't match schema | `coerce_args_to_schema` does best-effort casting. Failures propagate to tool execution; the error string becomes the observation; next turn retries. |
| Unclosed `<think>` block | `strip_think` falls back to stripping from `<think>` through end-of-string. |

Explicit non-goals: no in-adapter retries (ReAct's loop already retries); no
JSONAdapter fallback (text protocol *is* our robust path); no `finish_reason`
validation; `ContextWindowExceededError` propagates unchanged.

Logging: one `logger.warning` per parse failure with a 200-char content
snippet. `DSPY_QWEN35_DEBUG=1` raises to debug-level full-content dumps.

## Testing

Three layers.

**Layer 1 — `parsing.py` unit tests (fast, no LM).**
Corpus of ~30 `(raw_content.txt, expected.json)` fixture pairs under
`tests/fixtures/traces/`. Test discovers pairs, runs one test per pair.
Adding a new edge case = adding two files. Initial fixtures cover each
documented Reddit bug; new failures from live harness runs get promoted into
this dir as regression tests.

**Layer 2 — Adapter integration tests (fast, mocked LM).**
`DummyLM` stubs responses. Verifies:
- System prompt contains `<tools>` block when `list[Tool]` input field
  present.
- `format()` replays multi-turn trajectory as Qwen XML in assistant messages.
- `parse()` populates ReAct's three fields correctly given fixture content.
- `strict_parse=True` raises; default tolerates.

**Layer 3 — Live harness (slow, real LM Studio).**
`harness/run_eval.py --adapter {chat,json,qwen35} --scenario {s1,s3,s10}`.

Scenarios:
- **S1** — 1 tool (weather lookup). Golden = 1 tool call.
- **S3** — 3 tools (search + calculator + finish). Golden = 2-3 calls.
- **S10** — 10 tools (mixed research/math/utility). Golden = 3-5 calls.

N=20 runs per cell. Total: 3 × 3 × 20 = 180 runs against local LM Studio.

Per-run CSV columns: `scenario`, `adapter`, `run_idx`, `turns_completed`,
`max_iters_hit`, `parse_failures`, `tool_exec_failures`, `task_succeeded`.
`analyze.py` renders a markdown summary table. Headline metric: parse-failure
rate per adapter per scenario. Secondary: task success, mean turns.

`--capture-traces` flag dumps every raw `message.content` to
`harness/traces/` for later promotion into the fixture set.

## Success criteria (v1)

- `Qwen35Adapter` shows ≤ 1/3 the parse-failure rate of `ChatAdapter` on S3
  and S10.
- Task success rate ≥ `ChatAdapter` and `JSONAdapter` on all three scenarios.
- Parser unit tests pass against the full fixture corpus.
- User-facing API is exactly the quickstart shown in the Goal section.

If v1 meets these, publish to PyPI.

## Out of scope for v1

- Multi-model support (Qwen 3.5 35B A3B only).
- Multi-server certification (LM Studio tested; other OpenAI-compatible
  servers should work but not benchmarked).
- Streaming (documented as the buggiest server path; non-streaming only).
- Async / batched inference.
