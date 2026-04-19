# PUMA

**PUMA** aims to be a lightweight, high-performance inference engine for local AI. *Play for fun.*

## Features

- 🚀 **Model Management** - Download and manage AI models from multiple providers

## Quick Start

### Install from source

```bash
make build
```

## Commands

| Command | Description |
|---------|-------------|
| `pull` | Download a model from a provider |
| `ls` | List local models |
| `ps` | List running models |
| `run` | Create and run a model |
| `stop` | Stop a running model |
| `rm` | Remove a model |
| `info` | Display system-wide information |
| `inspect` | Return detailed information about a model |
| `version` | Show PUMA version |
| `help` | Show help information |

## Development

### Build

```bash
make build
```

### Test

```bash
make test
```

### Supported Providers

- ✅ **Hugging Face** - Full support with custom cache directories
- 🚧 **ModelScope** - Coming soon
