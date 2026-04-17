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

Head-to-head benchmarks against stock `ChatAdapter` and `JSONAdapter` on
Qwen 3.5 35B-A3B and Qwen 3.5 4B are in [docs/benchmarks.md](docs/benchmarks.md).
The headline: on multi-turn scenarios where tool output fidelity matters
(multilingual args, adversarial delimiters in tool output), this adapter
wins decisively. On simple scenarios, all three adapters pass — Qwen 3.5
is smart enough that parse robustness isn't the bottleneck, but trajectory
rendering is.

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
