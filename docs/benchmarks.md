# Benchmarks

Head-to-head comparison of `Qwen35Adapter` against DSPy's stock `ChatAdapter`
and `JSONAdapter` (with an LM Studio-compat shim) on a suite of ReAct
scenarios. Two Qwen 3.5 models, three adapters, eight scenarios, five runs
per cell — 240 runs total.

## TL;DR

- **Parsing is not the bottleneck** for any of the three adapters on Qwen
  3.5. Across 240 runs, the adapter-level parse-failure rate was 0.00.
  Models are smart enough to follow whichever wire format they're prompted
  for.
- **The real differentiators are trajectory rendering and thinking-mode
  handling**, not tool-call format. Qwen35Adapter's canonical
  `<tool_call>` / `<tool_response>` replay keeps the model in-distribution
  across turns; its `reasoning_content` fallback rescues turns where the
  server splits all output into a side channel.
- **Task-success numbers look deceptively close on easy scenarios.** Almost
  every adapter gets 100% on 1-tool, 3-tool, 10-tool, SQL-escaping, Python
  write-and-run, and adversarial-delimiter cases. The differences show up on
  multilingual args, long chains, and thinking-mode turns.

## Setup

| Field | Value |
|---|---|
| Inference server | LM Studio 0.4.x (OpenAI-compatible) at `http://127.0.0.1:1234/v1` |
| Models | `qwen3.5-35b-a3b` (Unsloth GGUF), `qwen3.5-4b` |
| Context window | 262,144 tokens (both models) |
| Sampling | `temperature=0.0` (deterministic where possible), `max_tokens=8192` |
| ReAct `max_iters` | 10 |
| Adapters | `ChatAdapter` (stock), `JSONAdapter` w/ LM Studio shim (skips `response_format` since LM Studio rejects `json_object`), `Qwen35Adapter` (this package) |
| Runs per cell | 5 |
| Harness | `harness/run_matrix.sh` |

Raw CSVs + captured LM traces for every run are archived under
`harness/results/qwen35-35b-a3b/` and `harness/results/qwen3.5-4b/`.

## Results

### qwen3.5-35b-a3b

| scenario | chat (succ / tool_fail) | json (succ / tool_fail) | **qwen35** (succ / tool_fail) |
|---|---|---|---|
| s1 — single tool | 100% / 0.00 | 100% / 0.00 | **100% / 0.00** |
| s3 — three tools | 100% / 0.00 | 100% / 0.00 | **100% / 0.00** |
| s10 — ten tools | 100% / 0.00 | 100% / 0.20 | **100% / 0.00** |
| s_sql — quoted SQL strings | 100% / 0.00 | 100% / 0.00 | **100% / 0.00** |
| s_code — Python write + run | 100% / 0.40 | 100% / 0.00 | **100% / 0.00** |
| s_echo — adversarial delimiters | 100% / 0.00 | 100% / 0.00 | **100% / 0.00** |
| s_deep — 8-step chain | 100% / 1.00 | 100% / 1.00 | **100% / 1.00** |
| **s_i18n — multilingual arg** | **20% / 0.00** | **0% / 0.00** | **100% / 0.00** |

### qwen3.5-4b

| scenario | chat (succ / tool_fail) | json (succ / tool_fail) | **qwen35** (succ / tool_fail) |
|---|---|---|---|
| s1 — single tool | 100% / 1.00 | 100% / 0.00 | **100% / 0.00** |
| s3 — three tools | 100% / 0.00 | 100% / 0.00 | **100% / 0.00** |
| s10 — ten tools | 100% / 0.00 | 100% / 0.00 | **100% / 0.00** |
| s_sql — quoted SQL strings | 100% / 0.00 | 100% / 0.00 | **100% / 0.00** |
| s_code — Python write + run | 100% / 0.00 | 100% / 0.00 | **100% / 0.00** |
| s_echo — adversarial delimiters | 100% / 0.00 | **80% / 0.00** | **100% / 0.00** |
| s_deep — 8-step chain | 100% / 1.00 | 100% / 0.00 | **100% / 0.00** |
| s_i18n — multilingual arg | 100% / 0.00 | 0% / 0.00 | 0% / 0.00 |

## Scenario catalog

Each scenario is a `(signature, tools, golden_answer_substring)` tuple
defined in `harness/scenarios.py`. The `golden_answer_substring` is a
case-insensitive substring match against `pred.answer`.

| key | scope | tools | what it stresses |
|---|---|---|---|
| `s1` | 1 tool, ~2 turns | weather lookup | baseline parse round-trip |
| `s3` | 3 tools, ~2-3 turns | search, calculator, word_count | tool selection + numeric args |
| `s10` | 10 tools, ~3 turns | mixed utilities | tool-list bloat, name disambiguation |
| `s_sql` | 1 tool | `execute_sql` on a users table | apostrophe-quoted SQL string args |
| `s_code` | 2 tools | `write_python`, `run_python` | multi-line source with quotes/braces; real `exec()` in sandbox |
| `s_echo` | 2 tools | `format_template`, `inspect_text` | tool output contains literal `[[ ## answer ## ]]` tokens that get replayed into the next turn's trajectory |
| `s_deep` | 10 tools, 5+ turns | same as s10 | trajectory replay accuracy over many turns |
| `s_i18n` | 1 tool | `translate` (mock) | multilingual paragraph with em-dashes, curly quotes, `café` |

## Cross-model findings

### 1. Parse robustness is a non-issue

0 parse failures across 240 runs and 3 adapters. Qwen 3.5 in-context-follows
whatever output format the prompt shows — XML if prompted XML, JSON if
prompted JSON, delimited if prompted delimited. The "Qwen emits native XML
that servers mangle" Reddit narrative that motivated the XML parser only
applies when using the server's **native function calling** path
(`tools=[]`), which none of these adapters use in text-native mode.

### 2. Tool-execution retries show hidden quality differences

Even when every adapter shows 100% task success, the `tool_fail / run`
column tells a quieter story:

- **35B `s_code`**: chat 0.40, json 0.00, qwen35 0.00. ChatAdapter produced 2
  bad tool calls across 5 runs that ReAct had to recover from. Same task
  success, but more wasted turns.
- **35B `s10`**: json 0.20 vs others 0.00. One bad call out of five runs.
- **4B `s1`**: chat 1.00 vs others 0.00. ChatAdapter averaged a bad call
  per run on the *simplest* scenario — 4B gets confused by delimiter
  parsing more than by Qwen XML.

### 3. Trajectory rendering matters more than parsing

`s_i18n` on **35B** is the clearest single-scenario win:

| adapter | 35B s_i18n | why |
|---|---|---|
| chat | 20% | flat trajectory rendering; model echoes tool output verbatim but sometimes drops the `[translated to SPANISH]` prefix |
| json | 0% | same plus some extract turns come back with empty `text` (see finding #4) |
| **qwen35** | **100%** | canonical `<tool_response>...</tool_response>` wrapping in trajectory keeps the model aware of what the tool returned; answer consistently references the tool's output |

The win isn't that our parser is more forgiving — it's that Qwen's trained
multi-turn format (with `<tool_call>` on assistant turns and
`<tool_response>` on tool turns) keeps the model in-distribution across a
sequence. When that structure is absent, the model loses track of what came
from where.

### 4. Thinking-mode splits `text` and breaks downstream extraction

On `qwen3.5-4b`, LM Studio's reasoning parser routes `<think>...</think>`
content into a separate `reasoning_content` field in the OpenAI response.
On turns where the model thinks *a lot* (40+ KB of reasoning observed),
`text` comes back empty and DSPy's base `_call_postprocess` skips
`parse()` entirely (see `dspy/adapters/base.py:136`) — returning `None`
for every output field.

This manifested in the 4B run as:
- `json` adapter 0% on `s_i18n` and 80% on `s_echo`.
- Any adapter can hit this on hard extract turns.

`Qwen35Adapter` handles this defensively: when `text` is empty and
`reasoning_content` is non-empty, it promotes the reasoning into `text`
before parsing. Our existing `strip_think` + `split_thought_and_call`
pipeline then runs on it and recovers any trailing tool call or answer.

### 5. The `s_i18n` flip between models is a model-capability regression, not a metric artifact

On 35B, qwen35 wins `s_i18n` 100-20-0. On 4B, chat wins 100-0-0 with qwen35
at 0%. An earlier version of this doc attributed the flip to the
loose substring metric; a follow-up run with the LLM-judge metric (see
§6) confirmed the judge **agrees with substring on 4B**: the qwen35 4B
answer genuinely is weaker than the chat 4B answer, not just paraphrased.

What's actually happening:

- **35B chat** sometimes strips the mock `[translated to SPANISH]` prefix
  when reporting (20% pass); qwen35's trajectory keeps it visible (100%).
- **4B chat** literally echoes the tool's observation including the
  prefix — a simpler-model "just copy the trajectory" behavior that
  happens to mention "Spanish."
- **4B qwen35** paraphrases the observation away, claims the tool's
  English output IS the Spanish translation, and loses the context that
  a translation was attempted.

The takeaway: the adapter's cleaner trajectory rendering is a win on the
35B model (keeps important context salient) and a loss on the 4B model
(the paraphrase is too aggressive for a weak model to reconstruct the
right answer). Adapter benefits are capability-dependent.

### 6. LLM-judge validation confirms the substring metric is directionally correct

We re-ran a narrow slice (4B, `s_i18n`, 3 runs per adapter) with an
LLM-judge metric against per-scenario rubrics (`Scenario.judge_criterion`).
Results aligned perfectly with the substring match — chat 100/100, json
0/0, qwen35 0/0 — but the judge's one-sentence reasons are more
informative:

- json's "answer is empty" reason pinpoints the thinking-mode empty-text
  failure mode (finding #4).
- qwen35's "repeats the original English text instead of reporting a
  Spanish translation" reason pinpoints the paraphrasing regression
  described above.

Full cross-model rerun with the judge is future work; see `--use-judge`
in the harness docs.

## Limitations and threats to validity

- **N=5 per cell.** Adequate for order-of-magnitude differences (e.g. 0%
  vs 100% on `s_i18n` is reliable); too noisy for smaller effects (e.g. a
  20% → 40% improvement would need N=30+).
- **Loose substring success metric.** Depends on the golden substring
  appearing literally in `pred.answer`. Fails to credit correct answers
  phrased differently; passes incorrect answers that contain the substring
  by accident. See finding #5.
- **Mock tools are deterministic and narrow.** Real production scenarios
  have API flakiness, variable output shapes, and richer error modes.
- **`temperature=0.0`.** Reduces run-to-run variance (good for N=5) but
  also hides cases where the parser has to handle sampling-induced
  variability in output format.
- **Single inference server.** Results on vLLM, llama.cpp, or Ollama may
  differ because each server's reasoning parser and tool-call parser
  behave differently with Qwen 3.5 GGUFs.
- **Non-ReAct usage not tested.** The adapter has a non-ReAct fallback
  (`Predict` over a signature with a single output field); no scenario
  here exercises it.

## Reproducing

From the repo root with `.venv` activated and LM Studio serving a Qwen 3.5
model at `http://127.0.0.1:1234/v1`:

```bash
# Full matrix — all adapters, all scenarios
./harness/run_matrix.sh --runs 5

# Only the scenarios that matter for a specific question
./harness/run_matrix.sh --runs 20 --scenarios s_i18n,s_deep

# Skip adapters that are already known to pass
./harness/run_matrix.sh --runs 5 --adapters json,qwen35

# Different model
QWEN_MODEL=openai/qwen3.5-4b ./harness/run_matrix.sh --runs 5
```

Results land in `harness/results/` as per-cell CSVs plus a `summary.md`.
Raw LM completions are captured under `harness/traces/`.
