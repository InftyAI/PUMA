# PUMA

A lightweight, high-performance inference engine for local AI. *Play for fun.*

## Features

- **Model Management** - Download and manage AI models from model providers like Hugging Face
- **System Detection** - Automatic GPU detection and system information reporting
- **Local Caching** - Efficient model storage with custom cache directories
- **Multiple Providers** - Support for Hugging Face with ModelScope coming soon

## Installation

### From Source

```bash
make build
```

The binary will be available as `./puma`.

## Quick Start

### 1. Download a Model

```bash
# From Hugging Face (default)
puma pull InftyAI/tiny-random-gpt2
```

### 2. List Downloaded Models

```bash
puma ls
```

### 3. Check System Information

```bash
puma info
```

Example output:
```
System Information:
  Operating System:   Darwin
  Architecture:       arm64
  CPU Cores:          14
  Total Memory:       36.00 GiB
  GPU:                Apple M4 Max (Metal) - 32 GPU cores

PUMA Information:
  PUMA Version:       0.0.1
  Cache Directory:    ~/.puma/cache
  Cache Size:         799.88 MiB
  Models:             1
  Running Models:     0
```

## Commands

| Command | Status | Description | Example |
|---------|--------|-------------|---------|
| `pull` | ✅ | Download a model from a provider | `puma pull InftyAI/tiny-random-gpt2` |
| `ls` | ✅ | List local models | `puma ls` |
| `ps` | 🚧 | List running models | `puma ps` |
| `run` | 🚧 | Create and run a model | `puma run InftyAI/tiny-random-gpt2` |
| `stop` | 🚧 | Stop a running model | `puma stop <model-id>` |
| `rm` | ✅ | Remove a model | `puma rm InftyAI/tiny-random-gpt2` |
| `info` | ✅ | Display system-wide information | `puma info` |
| `inspect` | ✅ | Return detailed information about a model or service | `puma inspect InftyAI/tiny-random-gpt2` |
| `version` | ✅ | Show PUMA version | `puma version` |
| `help` | ✅ | Show help information | `puma help` |

## Configuration

PUMA stores models in `~/.puma/cache` by default. This location is used for all downloaded models and metadata.

## Supported Providers

- **Hugging Face** - Full support with custom cache directories

## Development

### Build

```bash
make build
```

### Test

```bash
make test
```

### Project Structure

```
puma/
├── src/
│   ├── cli/         # Command-line interface
│   ├── downloader/  # Model download logic
│   ├── registry/    # Model registry management
│   ├── system/      # System detection (CPU, GPU, memory)
│   └── utils/       # Utility functions
├── Cargo.toml       # Rust dependencies
└── Makefile         # Build commands
```

## License

Apache-2.0

## Contributing

[![Star History Chart](https://api.star-history.com/svg?repos=inftyai/puma&type=Date)](https://www.star-history.com/#inftyai/puma&Date)
