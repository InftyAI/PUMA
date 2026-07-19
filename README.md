<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/InftyAI/PUMA/main/site/images/logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/InftyAI/PUMA/main/site/images/logo-light.svg">
  <img alt="PUMA Logo" src="https://raw.githubusercontent.com/InftyAI/PUMA/main/site/images/logo-light.svg" width="240">
</picture>

**A lightweight, high-performance model engine for local AI**

[![Stability: Active](https://img.shields.io/badge/stability-active-brightgreen.svg)](https://github.com/InftyAI/PUMA)
[![Latest Release](https://img.shields.io/github/v/release/InftyAI/PUMA)](https://github.com/InftyAI/PUMA/releases)

</div>

## ✨ Features

🔧 **Model Management** - Download, cache, and organize AI models from Hugging Face

🔍 **Advanced Filtering** - Search models with regex patterns and SQL-style queries

💻 **System Detection** - Automatic GPU detection and resource reporting

🚀 **OpenAI-Compatible API** - RESTful API with streaming support

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

### CLI Usage

```bash
# Download a model
puma pull qwen/qwen2.5-0.5b

# Run a model in an interactive chat
puma run qwen/qwen2.5-0.5b

# List all models
puma ls

# Inspect model details
puma inspect qwen/qwen2.5-0.5b

# Check system info
puma info

# Remove a model
puma rm qwen/qwen2.5-0.5b
```

### API Server

```bash
# Start the inference server with a model
puma serve qwen/qwen2.5-0.5b

# Server will start on http://0.0.0.0:8000
# API endpoints:
#   POST /v1/chat/completions
#   POST /v1/completions
#   GET  /v1/models
#   GET  /v1/models/:model
#   GET  /health
```

**Test the API:**

```bash
# Health check
curl http://localhost:8000/health

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Or use the test script
./hack/scripts/test_api.sh
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
| `run <model>` | ✅ | Run a model in an interactive chat |
| `serve <model>` | ✅ | Start OpenAI-compatible API server with a model |
| `ps` | 🚧 | List running models |
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

### Engine Tuning

Both `run` and `serve` accept flags to tune the inference engine (KV-cache pool,
block size, batch size, default token budget):

```bash
puma run qwen/qwen2.5-0.5b --max-batch-size 64 --default-max-tokens 256
```

See [docs/configuration.md](docs/configuration.md) for the full list of flags and defaults.

## API Server

PUMA provides an OpenAI-compatible API server for model inference.

### Starting the Server

```bash
# Start server with a model (default: 0.0.0.0:8000)
puma serve qwen/qwen2.5-0.5b

# Custom host and port
puma serve qwen/qwen2.5-0.5b --host 127.0.0.1 --port 3000

# Model must be pulled first
puma pull qwen/qwen2.5-0.5b
```

Engine parameters (KV-cache, batch size, etc.) can be tuned with additional flags — see [docs/configuration.md](docs/configuration.md).

### API Endpoints

#### Chat Completions (Recommended)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen2.5-0.5b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

#### Streaming (Server-Sent Events)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

#### List Models
```bash
# Returns the currently loaded model
curl http://localhost:8000/v1/models
```

#### Health Check
```bash
curl http://localhost:8000/health
# Returns: {"status":"ok"}
```

### OpenAI Python Client

PUMA is compatible with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not required
)

response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### Inspect Output

```bash
$ puma inspect qwen/qwen2.5-0.5b

name: qwen/qwen2.5-0.5b
kind: Model
spec:
  author:         qwen
  model_series:   qwen2
  task:           text-generation
  license:        APACHE-2.0
  context_window: 32.77K
  safetensors:
    total:        494.03M
    parameters:
      bf16:        494.03M
  provider:       huggingface
  cache:
    revision:       060db6499f32faf8b98477b0a26969ef7d8b9987
    size:           988.10 MB
    path:           ~/.puma/cache/huggingface/models--qwen--qwen2.5-0.5b
status:
  created:        2 months ago
  updated:        2 months ago
```

## Model Management

- **Database:** `~/.puma/models.db` (SQLite)
- **Cache:** `~/.puma/cache/` (model files)

Models are stored with lowercase names for case-insensitive matching.

## Development

```bash
# Build
make build

# Run all tests
make test

# Test API manually
./hack/scripts/test_api.sh
```

### Project Structure

```
puma/
├── src/
│   ├── api/          # OpenAI-compatible API
│   ├── backend/      # Inference backends (Mock, MLX)
│   ├── cli/          # Command implementations
│   ├── downloader/   # HuggingFace download logic
│   ├── registry/     # Model registry & metadata
│   ├── storage/      # SQLite storage backend
│   ├── system/       # System info detection
│   └── utils/        # Formatting & helpers
├── tests/            # Integration tests
├── hack/             # Development scripts
├── Cargo.toml        # Rust dependencies
└── Makefile          # Build commands
```

## License

Apache-2.0

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=inftyai/puma&type=Date)](https://www.star-history.com/#inftyai/puma&Date)
