# Hack Directory

Development and testing utilities for PUMA.

## Structure

```
hack/
└── scripts/          # Test and utility scripts
    └── test_api.sh
```

## Scripts

### `scripts/test_api.sh`

Tests all PUMA API endpoints manually.

**Usage:**
```bash
# Start PUMA server first
./puma serve

# In another terminal
./hack/scripts/test_api.sh
```

**Tests:**
- Health check
- List models
- Chat completion (non-streaming)
- Chat completion (streaming)
- Text completion

**Requirements:**
- Running PUMA server
- `curl` and `jq` installed

---


## Adding New Scripts

Place development and testing scripts in `hack/scripts/`:

```bash
# Create new script
cat > hack/scripts/my_script.sh << 'EOF'
#!/bin/bash
# Your script here
EOF

# Make executable
chmod +x hack/scripts/my_script.sh
```

---

## Why "hack"?

The `hack/` directory is a convention from Kubernetes and other projects for:
- Development utilities
- Test scripts
- Build helpers
- CI/CD scripts
- One-off tools

It keeps the root directory clean while providing a place for development tools.
