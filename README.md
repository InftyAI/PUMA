<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/images/logo-light.svg">
  <img alt="PUMA Logo" src="https://github.com/InftyAI/PUMA" width="240">
</picture>

**A lightweight, high-performance inference engine for local AI**

[![Stability: Active](https://img.shields.io/badge/stability-active-brightgreen.svg)](https://github.com/InftyAI/PUMA)
[![Latest Release](https://img.shields.io/github/v/release/InftyAI/PUMA)](https://github.com/InftyAI/PUMA/releases)

</div>

## ✨ Features

🔧 **Model Management** - Download, cache, and organize AI models from Hugging Face

🔍 **Advanced Filtering** - Search models with regex patterns and SQL-style queries

💻 **System Detection** - Automatic GPU detection and resource reporting

## Installation

### Install with Cargo

```bash
cargo install puma
```

### Build from Source

```bash
# Clone the repository
git clone https://github.com/InftyAI/PUMA.git
cd PUMA

# Build the binary
make build

# The binary will be available at ./puma
./puma version
```

## Quick Start

```bash
# Download a model
puma pull inftyai/tiny-random-gpt2

# List all models
puma ls

# Inspect model details
puma inspect inftyai/tiny-random-gpt2

# Check system info
puma info

# Remove a model
puma rm inftyai/tiny-random-gpt2
```

## Commands

| Command | Status | Description |
|---------|--------|-------------|
| `pull <model>` | ✅ | Download model from provider |
| `ls` | ✅ | List models (supports regex, label filters) |
| `inspect <model>` | ✅ | Show detailed model information |
| `rm <model>` | ✅ | Remove model and cache |
| `info` | ✅ | Display system information |
| `version` | ✅ | Show PUMA version |
| `ps` | 🚧 | List running models |
| `run` | 🚧 | Start model inference |
| `stop` | 🚧 | Stop running model |

## Advanced Usage

### Pattern Matching

```bash
# Substring match
puma ls qwen

# Prefix match
puma ls "^inftyai/"

# Alternation
puma ls "llama-(2|3)"
```

### Label Filtering

```bash
# Single filter
puma ls -l author=inftyai

# Multiple filters (AND condition)
puma ls -l author=inftyai,license=mit

# Combine pattern + filter
puma ls llama -l author=meta
```

**Available filters:** `author`, `task`, `license`, `provider`, `model_series`

### Inspect Output

```bash
$ puma inspect inftyai/tiny-random-gpt2

name: inftyai/tiny-random-gpt2
kind: Model
spec:
  author:         inftyai
  task:           text-generation
  license:        MIT
  model_series:   gpt2
  context_window: 2.05K
  safetensors:
    total:        7.00B
    parameters:
      f32:        7.00B
  artifact:
    provider:     huggingface
    revision:     abc123de
    size:         1.24 GB
    cache_path:   ~/.puma/cache/...
status:
  created:      2 hours ago
  updated:      2 hours ago
```

## Model Management

- **Database:** `~/.puma/models.db` (SQLite)
- **Cache:** `~/.puma/cache/` (model files)

Models are stored with lowercase names for case-insensitive matching.

## Development

```bash
# Build
make build

# Run tests (67 unit + 22 integration)
make test
```

### Project Structure

```
puma/
├── src/
│   ├── cli/          # Command implementations (ls, rm, inspect)
│   ├── downloader/   # HuggingFace download logic
│   ├── registry/     # Model registry & metadata
│   ├── storage/      # SQLite storage backend
│   ├── system/       # System info detection
│   └── utils/        # Formatting & helpers
├── tests/            # Integration tests
├── Cargo.toml        # Rust dependencies
└── Makefile          # Build commands
```

## License

Apache-2.0

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=inftyai/puma&type=Date)](https://www.star-history.com/#inftyai/puma&Date)
