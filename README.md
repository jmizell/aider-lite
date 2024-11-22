# Aider-Lite - Stripped Down ChatGPT Coding Assistant

**Note: This is a stripped down fork of [aider](https://github.com/paul-gauthier/aider)**

This fork focuses on core functionality by maintaining only essential features. Key changes include:

- Removed voice, vision, web scraping, docs and analytics capabilities
- Removed support for non-OpenAI compatible API endpoints
- Stripped down to core code editing and git integration features
- Reduced complexity of command-line arguments

Aider-Lite is a command-line tool that helps you edit code in your local git repository using natural language conversations with AI language models. It's designed to be focused on core pair programming functionality.

## Installation

```bash
pip install .
```

## Quick Start
```bash
# Start aider_lite with specific files
aider_lite-lite file1.py file2.py

# Start aider_lite with a git repository
aider_lite-lite path/to/repo
```

## Supported Models
Aider-lite works with the following models out of the box:

* `anthropic/claude-3.5-sonnet` (default)
* `meta-llama/llama-3.2-3b-instruct`
* `meta-llama/llama-3.1-70b-instruct`
* `meta-llama/llama-3.1-405b-instruct`

## Command Line Options

### Main Options
* `--files`: Files to edit with the AI (optional)
* `--model MODEL`: Specify the model to use (default: `anthropic/claude-3.5-sonnet`)

### Model Settings
* `--list-models MODEL`: List known models matching the (partial) MODEL name
* `--model-settings-file FILE`: Specify model settings file (default: `.aider.model.settings.yml`)
* `--model-metadata-file FILE`: Specify model metadata file (default: `.aider.model.metadata.json`)
* `--edit-format FORMAT`: Specify edit format for the LLM
* `--weak-model MODEL`: Model for commit messages and history summarization
* `--editor-model MODEL`: Model for editor tasks
* `--max-chat-history-tokens N`: Token limit for chat history before summarization

### Cache Settings
* `--cache-prompts`: Enable prompt caching (default: `False`)
* `--cache-keepalive-pings N`: Number of keepalive pings for cache (default: `0`)

### Git Integration
* `--git`: Enable/disable git repo detection (default: `True`)
* `--gitignore`: Add `.aider*` to `.gitignore` (default: `False`)
* `--auto-commits`: Enable/disable auto-commit of changes (default: `False`)
* `--dirty-commits`: Enable/disable commits for dirty repo (default: `True`)
* `--commit`: Commit changes and exit
* `--commit-prompt PROMPT`: Custom prompt for commit messages

### Code Quality
* `--lint`: Lint and fix provided/dirty files
* `--lint-cmd CMD`: Specify lint commands (e.g., `python: flake8 --select=...`)
* `--auto-lint`: Enable/disable automatic linting (default: `True`)
* `--test-cmd CMD`: Specify test command
* `--auto-test`: Enable/disable automatic testing (default: `False`)
* `--test`: Run tests and fix problems

### Output Settings
* `--dark-mode`: Use colors for dark terminal background
* `--light-mode`: Use colors for light terminal background
* `--pretty`: Enable/disable colorized output (default: `True`)
* `--stream`: Enable/disable streaming responses (default: `False`)

### Other Settings
* `--vim`: Use VI editing mode in terminal (default: `False`)
* `--encoding ENC`: Specify encoding (default: `utf-8`)
* `--message MSG`: Send single message and exit
* `--message-file FILE`: Send message from file and exit
* `--yes-always`: Always confirm all prompts
* `-v`, `--verbose`: Enable verbose output
* `--version`: Show version number and exit

## Environment Variables
### Essential environment variables:

* `OPENAI_API_KEY`: Your API key
* `OPENAI_API_BASE`: API base URL (default: `https://api.openai.com`)
Additional options can be configured using environment variables with the prefix `AIDER-LITE_`.

## Configuration Files
Aider looks for configuration in the following locations (in order):

* `.aider-lite.conf.yml` in the current directory
* `.aider-lite.conf.yml` in the git repository root
* `.aider-lite.conf.yml` in your home directory

### Examples
```
# Use with specific model
aider-lite --model anthropic/claude-3.5-sonnet file.py

# Enable dark mode with auto-commits
aider-lite --dark-mode --auto-commits project/

# Run with custom lint command
aider-lite --lint-cmd "python: flake8 --select=E9" src/

# Send single message and exit
aider-lite -m "Fix the bug in main.py" main.py
```
