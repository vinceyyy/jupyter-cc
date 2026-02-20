# jupyter-cc-magic

<!-- Brief one-line description of the project -->

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd jupyter-cc-magic

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --dev --upgrade

# Install pre-commit hooks
uv run pre-commit install
```


## Development

### Running Tests

```bash
uv run pytest
```

### Linting & Formatting

```bash
# Check for issues
uv run ruff check .
uv run ruff format --check .

# Auto-fix issues
uv run ruff check --fix .
uv run ruff format .
```

### Pre-commit Hooks

Pre-commit hooks run automatically on commit. To run manually:

```bash
uv run pre-commit run --all-files
```


## Project Structure

```
jupyter-cc-magic/
├── src/jupyter_cc_magic/   # Main package code
├── tests/                    # Test files
├── docs/                     # Documentation
│   ├── architecture.md       # Technical architecture
│   ├── product-design.md     # Product requirements
│   └── plans/                # Implementation plans
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Documentation

- [Product Design](docs/product-design.md) - Product vision and requirements
- [Architecture](docs/architecture.md) - Technical design decisions

## License

MIT
