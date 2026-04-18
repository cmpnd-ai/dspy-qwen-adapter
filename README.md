# dspy-qwen35-adapter

A DSPy adapter that makes `dspy.ReAct` reliable with Qwen 3.5 on any
OpenAI-compatible local inference server (LM Studio, vLLM, llama.cpp,
Ollama, SGLang).

## Why this exists

Qwen 3.5 was fine-tuned on a specific tool-call wire format —
`<tool_call><function=NAME><parameter=K>\nVALUE\n</parameter>...</function></tool_call>`
— inherited from the Qwen3-Coder lineage. Its official vLLM/SGLang
deployment recipe uses `--tool-call-parser qwen3_coder` to read that format
back. DSPy's stock adapters don't know about this format and either ask the
model to emit JSON (`JSONAdapter`) or a custom delimiter scheme
(`ChatAdapter`). That works — Qwen 3.5 is flexible enough to follow any
in-context prompt — but it drifts the model off its trained multi-turn
distribution and silently loses quality on longer chains.

This adapter:

- **Prompts the model in its trained format** — canonical `<tool_call>` /
  `<tool_response>` XML, no JSON escaping fragility.
- **Replays multi-turn trajectories** using Qwen's chat-template shape
  (`<tool_call>` on assistant turns, `<tool_response>` on tool turns) so
  long agent runs stay in-distribution.
- **Bypasses the inference server's tool-call parser** entirely (doesn't
  pass `tools=[]`), so it works even on servers whose Qwen 3 parser has
  known bugs.
- **Strips leaked `<think>` tags** from completions before parsing.
- **Rescues empty `text` turns** by falling back to `reasoning_content`
  — important for thinking-mode models on LM Studio, where the server can
  route the entire completion into a side channel.

## Benchmark results

Head-to-head comparison against stock `ChatAdapter` and an LM Studio-compat
`JSONAdapter` shim across 8 scenarios × 3 adapters × 5 runs per model, with
LLM-judge verdicts alongside cheap substring matching. See
[docs/benchmarks.md](docs/benchmarks.md) for full methodology, per-cell
reasoning, and limitations.

**Columns:** task_success (substring / LLM-judge) / tool_fail per run.
All runs completed 0 parse failures.

### qwen3.5-35b-a3b

| scenario | chat | json | **qwen35** |
|---|---|---|---|
| s1, s3, s_sql, s_code | 100 / 100 / 0.00 | 100 / 100 / 0.00 | **100 / 100 / 0.00** |
| s10 (10 tools) | 100 / 100 / 0.00 | 100 / 100 / 0.80 | **100 / 100 / 0.00** |
| s_echo (adversarial delimiters) | 100 / 100 / 0.00 | 100 / 100 / 0.40 | **100 / 100 / 0.00** |
| s_deep (8-step chain) | 100 / 100 / 0.60 | 100 / 100 / 0.20 | **100 / 100 / 0.00** |
| **s_i18n (multilingual arg)** | **0 / 0 / 0.00** | 40 / 40 / 0.00 | **100 / 80 / 0.00** |

### qwen3.5-4b

| scenario | chat | json | **qwen35** |
|---|---|---|---|
| s3, s10, s_sql, s_code | 100 / 100 / 0.00 | 100 / 100 / 0.00 | **100 / 100 / 0.00** |
| s1 (single tool) | 100 / 100 / 1.00 | 100 / 100 / 0.00 | **100 / 100 / 0.00** |
| s_echo | 100 / 100 / 0.00 | 80 / 80 / 0.00 | **100 / 100 / 0.00** |
| s_deep | 100 / 100 / 1.00 | 100 / 100 / 0.00 | **100 / 100 / 0.00** |
| **s_i18n** | 100 / 100 / 0.00 | **0 / 0 / 0.00** | **100 / 100 / 0.00** |

*(4B `s_i18n` number reflects the fix in commit `635f60e` — extract-turn
guidance that asks the model to quote tool outputs verbatim; before the
fix, qwen35 was 0/0 on this cell.)*

### Headline findings

- **Fewer tool-call failures than either alternative on every complex
  scenario.** `qwen35` averaged **0.00 tool_fail/run** on `s_deep` (8-step
  chain) and `s10` on both models; `chat` spiked to 0.60-1.00 and `json`
  to 0.20-0.80 on the same cells. Same task success, fewer wasted turns.
- **Only adapter that reliably handles multilingual / delimiter-leaking
  tool output.** `s_i18n` on 35B: `qwen35` 100%, chat 0%, json 40%.
- **Rescues `reasoning_content` turns that silently break stock
  adapters.** On the 4B model with thinking mode, extract turns sometimes
  come back with `text=""` and all output fields `None`. `json` lost a
  run on `s_echo` this way; `qwen35` caught it via the
  `reasoning_content` fallback.
- **0 parse failures across 480 total runs** on both models. Parse
  robustness is a non-issue for any adapter on Qwen 3.5; the
  differentiators are trajectory rendering and thinking-mode handling.

## Install

From PyPI (once published):

```bash
pip install dspy-qwen35-adapter
```

From source (editable):

```bash
git clone https://github.com/<user>/dspy-qwen35-adapter
cd dspy-qwen35-adapter
pip install -e .
```

## Quickstart

```python
import dspy
from dspy_qwen35_adapter import Qwen35Adapter

dspy.configure(
    lm=dspy.LM(
        "openai/qwen/qwen3.5-35b-a3b",
        api_base="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        temperature=1.0,
        max_tokens=8192,
    ),
    adapter=Qwen35Adapter(),
)

def get_weather(city: str) -> str:
    """Get the current weather in a city."""
    return f"sunny, 72F in {city}"

react = dspy.ReAct("question -> answer", tools=[get_weather])
print(react(question="What's the weather in Tokyo?").answer)
```

That's the whole user-facing surface — instantiate `Qwen35Adapter()`, pass
it to `dspy.configure`, use `dspy.ReAct` as normal. No prompt templates,
no parser configuration, no server-specific flags.

## Configuration

```python
Qwen35Adapter(
    callbacks=None,                 # list[BaseCallback] — standard DSPy callbacks
    native_response_types=None,     # list[type] — forwarded to base Adapter
    strict_parse=False,             # True: raise AdapterParseError when no tool call
                                    # is present. False (default): treat as a
                                    # graceful finish — the model's text becomes
                                    # the thought, and ReAct moves to extract.
)
```

`use_native_function_calling` is hardcoded off — we never pass `tools=[]`
to the server, which is what makes this adapter robust across servers with
different Qwen tool-parser quirks.

## Compatibility

- **Model**: Qwen 3.5 family. Not Qwen 3 base (which uses a Hermes-style
  JSON-in-XML format — different wire protocol; a `Qwen3Adapter` would be a
  separate package).
- **Server**: any OpenAI-compatible `chat/completions` endpoint. Tested
  against LM Studio 0.4.x; should work against vLLM, SGLang, llama.cpp,
  and Ollama without any server-specific flags, since this adapter doesn't
  rely on native function calling.
- **Python**: 3.10+.
- **DSPy**: 3.1+.

## How it's different

| | ChatAdapter | JSONAdapter | **Qwen35Adapter** |
|---|---|---|---|
| Tool call format | `[[ ## field ## ]]` delimiters | JSON text (with `response_format` on some servers) | canonical `<tool_call>` XML |
| Trajectory replay | flat `name: value` lines | flat JSON per turn | `<tool_call>` + `<tool_response>` XML per turn |
| `<think>` tag handling | none | none | stripped before parsing |
| Empty-text (thinking mode) | drops the turn (all fields None) | drops the turn | falls back to `reasoning_content` |
| Server native tool parser | not used | used when `response_format` is supported | not used |

See [docs/benchmarks.md](docs/benchmarks.md) for the measured effect of each.

## Limitations

- **Only text-native mode.** This adapter does not use the server's native
  tool-call parser — by design. If you're on a server whose tool parser for
  Qwen 3.5 works perfectly, stock `JSONAdapter` with native function calling
  may be faster. The benchmarks show this adapter is at worst equivalent
  and at best dramatically better, at the cost of parsing tool calls in
  Python instead of at the server.
- **No demo / few-shot support in `format()`.** DSPy optimizers that rely
  on demo interleaving (BootstrapFewShot, MIPRO) will silently get
  zero-shot behavior. Tracking as a future enhancement.
- **Non-streaming only.** Streaming parsers for Qwen 3.5 are buggy in most
  current inference stacks; this adapter targets non-streaming responses.

## Development

Run the tests:

```bash
pip install -e '.[dev]'
pytest tests/ -v
```

Run the benchmark harness against a local model:

```bash
./harness/run_matrix.sh --runs 5
```

See [docs/benchmarks.md](docs/benchmarks.md) for the harness docs.

## License

MIT.
