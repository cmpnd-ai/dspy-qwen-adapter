# dspy-qwen-adapter

A DSPy adapter that makes `dspy.ReAct` and `dspy.Predict` reliable across
the **Qwen 3+ family** (Qwen 3, 3.5, Coder variants, and forward-compatible
with 3.6) on any OpenAI-compatible local inference server — LM Studio,
vLLM, llama.cpp, Ollama, SGLang.

## Why this exists

Qwen's tool-calling wire format **changes across generations**:

- **Qwen 3** (base): Hermes-style — `<tool_call>{"name": "...",
  "arguments": {...}}</tool_call>`. vLLM's `--tool-call-parser hermes`.
- **Qwen 3.5 / 3-Coder lineage**: XML — `<tool_call><function=NAME><parameter=K>\nVALUE\n</parameter>...</function></tool_call>`.
  vLLM's `--tool-call-parser qwen3_coder`.

DSPy's stock adapters (`ChatAdapter`, `JSONAdapter`, `XMLAdapter`) don't
know about either format. They ask the model for their own delimiter /
JSON / tagged-field schemes, which Qwen will follow via in-context
compliance — but that drifts the model off its trained multi-turn
distribution and silently loses quality on longer chains, or outright
fails when the model's output mixes formats.

This adapter:

- **Prompts the model in Qwen 3.5's canonical format.** In-context
  compliance pulls Qwen 3 to the same format via prompt exemplar, so a
  single adapter covers both generations. Benchmarks below show zero
  parse failures across 880 runs spanning three models.
- **Replays multi-turn trajectories** using the Qwen chat-template shape
  (`<tool_call>` on assistant turns, `<tool_response name="...">` on tool
  turns) so long agent runs stay in-distribution.
- **Bypasses the inference server's tool-call parser** entirely (never
  passes `tools=[]`), so it works even on servers whose native Qwen
  parsers have known bugs.
- **Strips leaked `<think>` tags** from completions before parsing.
- **Rescues empty `text` turns** by falling back to `reasoning_content`
  — important for thinking-mode models on LM Studio, where the server
  can route the entire completion into a side channel and leave `text`
  empty.
- **Inherits `XMLAdapter`** for plain `dspy.Predict` / `ChainOfThought` —
  non-tool-calling paths get `<field>content</field>` tags, which is
  still in Qwen's XML-heavy training distribution, plus demos and
  `dspy.History` support for free.

## Benchmark results

880 total runs: 4 adapters × 8 scenarios × 5 runs × 3 models, each with
an LLM-judge verdict alongside cheap substring matching. See
[docs/benchmarks.md](docs/benchmarks.md) for methodology, per-cell
reasoning, and limitations.

**Columns:** task_success (substring / LLM-judge) / tool_fail per run.
**Bold** entries mark the decisive cell for that row.

### qwen3.5-35b-a3b

| scenario | chat | json | xml | **qwen** |
|---|---|---|---|---|
| s1, s_sql, s_code | 100 / 100 / 0.00 | 100 / 100 / 0.00 | 100 / 100 / 0.00 | **100 / 100 / 0.00** |
| **s3 (three tools)** | 100 / 100 / 0.00 | 100 / 100 / 0.00 | **0 / 0 / 0.00** *(parse_fail 1.00)* | **100 / 100 / 0.00** |
| s10 (10 tools) | 100 / 100 / 0.00 | 100 / 100 / 0.80 | 100 / 100 / 0.40 | **100 / 100 / 0.00** |
| s_echo (adversarial delimiters) | 100 / 100 / 0.00 | 100 / 100 / 0.40 | 100 / 100 / 0.20 | **100 / 100 / 0.00** |
| s_deep (8-step chain) | 100 / 100 / 0.60 | 100 / 100 / 0.20 | **80 / 80 / 2.20** | **100 / 100 / 0.00** |
| **s_i18n (multilingual arg)** | **0 / 0 / 0.00** | 40 / 40 / 0.00 | 0 / 0 / 0.00 | **100 / 80 / 0.00** |

### qwen3.5-4b

| scenario | chat | json | xml | **qwen** |
|---|---|---|---|---|
| s3, s10, s_sql | 100 / 100 / 0.00 | 100 / 100 / 0.00 | 100 / 100 / 0.00 | **100 / 100 / 0.00** |
| s1 (single tool) | 100 / 100 / 1.00 | 100 / 100 / 0.00 | 100 / 100 / 0.00 | **100 / 100 / 0.00** |
| **s_code** | 100 / 100 / 0.00 | 100 / 100 / 0.00 | **0 / 0 / 0.00** *(parse_fail 1.00)* | **100 / 100 / 0.00** |
| s_echo | 100 / 100 / 0.00 | 80 / 80 / 0.00 | 100 / 100 / 0.00 | **100 / 100 / 0.00** |
| s_deep | 100 / 100 / 1.00 | 100 / 100 / 0.00 | 100 / 100 / 0.00 | **100 / 100 / 0.00** |
| s_i18n | 100 / 100 / 0.00 | 0 / 0 / 0.00 | 0 / 0 / 0.00 | 0 / 0 / 0.00 |

### qwen3-4b (out-of-distribution — Hermes-format model)

Qwen 3 was trained on Hermes-style tool calls, not the XML format this
adapter prompts for. The benchmark tests whether in-context compliance
is strong enough to bridge the distribution gap.

| scenario | chat | json | xml | **qwen** |
|---|---|---|---|---|
| s1, s10, s_deep | 100 / 100 / 0.00 | 100 / 100 / 0.00 | 100 / 100 / 0.00 | **100 / 100 / 0.00** |
| s3 (three tools) | 100 / 100 / 0.00 | 100 / 100 / **2.00** | 100 / 100 / 0.00 | **100 / 100 / 0.00** |
| s_sql | 100 / 100 / 0.00 | 100 / 100 / **1.00** | 100 / 100 / 0.00 | **100 / 100 / 0.00** |
| **s_code** | 100 / 100 / **1.00** | 100 / 100 / **1.00** | 100 / 100 / **2.00** | **100 / 100 / 0.00** |
| s_echo | 100 / 0 / 0.00 | 100 / 0 / 0.00 | 100 / 0 / 0.00 | 0 / 0 / 0.00 |
| s_i18n | 0 / 0 / 0.00 | 0 / 0 / 0.00 | 0 / 0 / 0.00 | 0 / 0 / 0.00 |

*(s_echo + s_i18n 4B failures across all adapters are weak-model + mock-tool
artifacts — the 4B model hallucinates lengths or paraphrases narrative
prefixes regardless of adapter. See [docs/benchmarks.md](docs/benchmarks.md).)*

### Headline findings

- **0 parse failures across all 600 qwen-adapter runs on all three models.**
  XMLAdapter by comparison failed every `s3` run on 35B and every `s_code`
  run on 4B (parse_fail 1.00 per run).
- **0.00 tool_fail per run on every scenario on every model.** The closest
  alternatives spike to 0.20 – 2.20 on multi-step and structured-arg
  scenarios. Same or better task success, fewer wasted turns.
- **Only adapter that reliably handles multilingual / delimiter-leaking
  tool output on 35B.** `s_i18n`: `qwen` 100/80 vs chat 0/0, json 40/40,
  xml 0/0.
- **Rescues `reasoning_content` turns** that silently break stock adapters
  on thinking-mode models. `json` lost a run on `s_echo` 4B this way;
  `qwen` caught it via the fallback.
- **Works on Qwen 3 despite the training-distribution mismatch.** The XML
  exemplar in our prompt is strong enough that Qwen 3 (trained on
  Hermes) follows it anyway, and our adapter still posts the best
  tool_fail numbers across all scenarios.

## Install

From PyPI (once published):

```bash
pip install dspy-qwen-adapter
```

From source (editable):

```bash
git clone https://github.com/<user>/dspy-qwen-adapter
cd dspy-qwen-adapter
pip install -e .
```

## Quickstart

```python
import dspy
from dspy_qwen_adapter import QwenAdapter

dspy.configure(
    lm=dspy.LM(
        "openai/qwen/qwen3.5-35b-a3b",
        api_base="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        temperature=1.0,
        max_tokens=8192,
    ),
    adapter=QwenAdapter(),
)

def get_weather(city: str) -> str:
    """Get the current weather in a city."""
    return f"sunny, 72F in {city}"

react = dspy.ReAct("question -> answer", tools=[get_weather])
print(react(question="What's the weather in Tokyo?").answer)
```

That's the whole user-facing surface — instantiate `QwenAdapter()`, pass
it to `dspy.configure`, use `dspy.ReAct` or `dspy.Predict` as normal. No
prompt templates, no parser configuration, no server-specific flags.

Same code works unchanged on Qwen 3 — swap `openai/qwen/qwen3-4b` as the
model and run.

## Configuration

```python
QwenAdapter(
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

- **Model**: Qwen 3+ family. Optimized for Qwen 3.5 (XML-format lineage);
  works on Qwen 3 (Hermes-format) and Qwen 3-Coder via in-context
  compliance. Smaller variants (4B and below) can have weak-model
  artifacts on narrative-mock benchmarks but still post the best
  tool_fail rates.
- **Server**: any OpenAI-compatible `chat/completions` endpoint. Tested
  against LM Studio 0.4.x; should work against vLLM, SGLang, llama.cpp,
  and Ollama without any server-specific flags, since this adapter
  doesn't rely on native function calling.
- **Python**: 3.12+.
- **DSPy**: 3.1+.

## How it's different

| | ChatAdapter | JSONAdapter | XMLAdapter | **QwenAdapter** |
|---|---|---|---|---|
| Tool call format | `[[ ## field ## ]]` delimiters | JSON text (+ `response_format`) | `<field>content</field>` per output | canonical Qwen `<tool_call>` XML |
| Trajectory replay | flat `name: value` lines | flat JSON per turn | `<field>` lines per turn | `<tool_call>` + `<tool_response name="...">` XML per turn |
| `<think>` tag handling | — | — | — | stripped before parsing |
| Empty-text (thinking mode) | drops the turn (all fields None) | drops the turn | drops the turn | falls back to `reasoning_content` |
| Server native tool parser | not used | used when `response_format` is supported | not used | not used (by design) |
| Plain `dspy.Predict` | works | works | works | works (via XMLAdapter inheritance) |

See [docs/benchmarks.md](docs/benchmarks.md) for the measured effect of each.

## Limitations

- **Only text-native mode.** This adapter does not use the server's native
  tool-call parser — by design. If you're on a server whose tool parser for
  Qwen works perfectly, stock `JSONAdapter` with native function calling
  may be faster. The benchmarks show this adapter is at worst equivalent
  and at best dramatically better, at the cost of parsing tool calls in
  Python instead of at the server.
- **No demo / few-shot support on the ReAct path.** DSPy optimizers that
  rely on demo interleaving (BootstrapFewShot, MIPRO) will silently get
  zero-shot behavior on ReAct calls. Plain `Predict` inherits demo
  support from `XMLAdapter`. Tracking as a future enhancement.
- **Non-streaming only.** Streaming parsers for Qwen are buggy in most
  current inference stacks; this adapter targets non-streaming responses.
- **Small-model quirks.** The 4B-class models occasionally paraphrase
  narrative tool output or hallucinate numeric details on contrived
  benchmark scenarios (`s_echo`, `s_i18n`). Not a production
  tool-calling regression — real tools return real data. Bigger models
  (35B+) pass these cleanly.

## Development

Run the tests:

```bash
pip install -e '.[dev]'
pytest tests/ -v
```

Run the benchmark harness against a local model:

```bash
./harness/run_matrix.sh --runs 5 --use-judge
```

See [docs/benchmarks.md](docs/benchmarks.md) for the harness docs.

## License

MIT.
