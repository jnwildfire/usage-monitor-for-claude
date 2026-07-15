# Energy Estimate

The popup shows an estimated electricity figure for the current week and current month, next to the usage bars. This page explains what it is, where the numbers come from, and its limits.

## What it measures

Anthropic's OAuth usage API (`/api/oauth/usage`, see [API Reference](api-reference.md)) only reports quota **utilization percentages** for rolling time windows - it does not expose token counts, so it cannot be used to estimate energy use.

Instead, the energy estimate reads the token usage that Claude Code already logs locally, per turn, in `~/.claude/projects/**/*.jsonl` (or `$CLAUDE_CONFIG_DIR/projects/` if set). Each assistant turn in these transcripts includes a `message.usage` object with `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, and `cache_read_input_tokens`. The app sums these for the current calendar week (Monday 00:00 local time) and the current calendar month (the 1st, 00:00 local time), deduplicating turns that appear more than once across files (e.g. from resumed or compacted sessions).

## How tokens become watt-hours

There is no Anthropic-published per-token energy figure for Claude - this is **not a measurement**, it is an order-of-magnitude estimate using three configurable rates:

| Rate | Default | Applies to | Why it's different |
|------|---------|------------|---------------------|
| `energy_wh_per_1k_output_tokens` | 0.4 Wh / 1K tokens | `output_tokens` | Generated one token at a time (autoregressive decode) - the most expensive token type per token generated |
| `energy_wh_per_1k_input_tokens` | 0.05 Wh / 1K tokens | `input_tokens` + `cache_creation_input_tokens` | Processed in a single parallelizable prefill pass, roughly an order of magnitude cheaper per token than decode |
| `energy_wh_per_1k_cache_read_tokens` | 0.005 Wh / 1K tokens | `cache_read_input_tokens` | Reuses an existing KV cache instead of reprocessing the tokens, so it costs very little extra compute |

These defaults are round numbers picked to sit within the range of public estimates for large-model inference (for example Google's 2025 disclosure of a ~0.24 Wh median Gemini prompt, and Epoch AI's and Luccioni et al.'s published estimates for GPT-scale models) after accounting for a typical prompt's mix of input/output/cached tokens. Treat the resulting figure as **directionally useful, not precise** - actual energy use depends on hardware generation, batching, data center PUE, and Anthropic's own infrastructure, none of which is public information.

If you have better numbers (e.g. from a specific published disclosure you trust more), override the three rates in `usage-monitor-settings.json` - see [Configuration](configuration.md#energy-estimate).

## Disabling it

Set `"energy_enabled": false` in `usage-monitor-settings.json` if you'd rather not scan local transcripts or see the section at all.
