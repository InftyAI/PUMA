# Configuration

PUMA's inference engine exposes a set of tunable parameters. Both `puma run`
and `puma serve` accept the same engine flags — all are optional and fall back
to the defaults below.

## Engine flags

| Flag | Default | Description |
|------|---------|-------------|
| `--memory-pool-bytes` | `104857600` (100 MB) | Total KV-cache memory pool, in bytes |
| `--block-size-bytes` | `512` | Size of a single KV block, in bytes |
| `--tokens-per-block` | `16` | Number of tokens whose KV state fits in one block |
| `--max-batch-size` | `32` | Maximum number of sequences batched together per step |
| `--default-max-tokens` | `100` | Completion-token budget when a request omits `max_tokens` |

## Examples

```bash
# Larger KV-cache pool and batch size, bigger default generation budget
puma serve qwen/qwen2.5-0.5b \
  --memory-pool-bytes 209715200 \
  --max-batch-size 64 \
  --default-max-tokens 256

# The same flags work for interactive run
puma run qwen/qwen2.5-0.5b --tokens-per-block 32
```

## Notes

- **`--tokens-per-block`** must be greater than `0`.
- **`--default-max-tokens`** only applies when a request omits `max_tokens`; an
  explicit per-request `max_tokens` always takes precedence.
- The flag defaults are sourced from `EngineConfig` in
  `src/engine/mod.rs`, so the CLI and the library stay in sync.
