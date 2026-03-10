# auto-model

A **GitHub webhook router** that automatically selects the most cost-effective AI model for each GitHub task — using Claude for code reviews, GPT-4o-mini for documentation and issue triage, and Claude Haiku for quick comments.

## How it works

```
GitHub event  ──►  model_router.py  ──►  Best AI model  ──►  Response posted back
  (PR / issue /                          (by cost +
   push / …)                              capability)
```

The router inspects the GitHub event type (and payload details such as labels or file extensions) and maps it to the most cost-effective model:

| GitHub Event | Task | Selected Model | Why |
|---|---|---|---|
| `pull_request` opened/updated | `code_review` | **claude-sonnet** | Best code reasoning |
| `pull_request` (docs files only) | `documentation` | **gpt-4o-mini** | Cheap, great text |
| `issues` with `bug` label | `bug_fix` | **claude-sonnet** | Deep analysis |
| `issues` opened/edited | `issue_triage` | **gpt-4o-mini** | Low cost, sufficient |
| `issue_comment` / `pull_request_review_comment` | `quick_comment` | **claude-haiku** | Fastest & cheapest |
| `release` / `push` to main | `release_notes` | **gpt-4o-mini** | Good summarisation |

## Quick Start

### Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.9 |
| [GitHub CLI (`gh`)](https://cli.github.com/) | ≥ 2.0 |

No third-party Python packages are required — the router uses only the standard library.

### 1 — Clone the repo

```bash
gh repo clone squassina/auto-model
cd auto-model
```

### 2 — Run the router locally

```bash
# Show the best model for a pull_request event
python model_router.py --event pull_request

# Show the best model for an explicit task
python model_router.py --task documentation

# List every task→model mapping with costs
python model_router.py --list

# Route a real GitHub event payload file
python model_router.py --event pull_request --event-file /path/to/event.json

# Route and call the model API (requires API key in env)
export ANTHROPIC_API_KEY=sk-ant-...
python model_router.py --event pull_request \
    --prompt "Review this change for correctness" \
    --call-api \
    --output result.json
```

### 3 — Trigger via GitHub Actions

```bash
# Show routing decision for a simulated PR event
gh workflow run model-router.yml \
    -f event_override=pull_request

# Simulate issue triage with a prompt (requires API key secret)
gh workflow run model-router.yml \
    -f event_override=issues \
    -f task_override=issue_triage \
    -f prompt="Classify and suggest labels for this issue"

# Watch the run
gh run watch
```

The workflow also fires **automatically** on:
- `pull_request` (opened, synchronize, reopened)
- `issues` (opened, edited)
- `issue_comment` (created)
- `release` (published)
- `push` to `main`

## CLI Reference

| Flag | Description |
|------|-------------|
| `--event <name>` | GitHub event name to route (`pull_request`, `issues`, `push`, `release`, `issue_comment`) |
| `--task <name>` | Override task directly (bypasses event inference) |
| `--event-file <path>` | Path to a GitHub event JSON payload for detailed routing |
| `--prompt <text>` | Prompt to send to the selected model |
| `--call-api` | Actually invoke the model API (requires `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) |
| `--output <path>` | Write routing decision (and optional response) to a JSON file |
| `--list` | Print all task→model mappings with costs |
| `--config <path>` | Path to a custom `models.json` config (default: `config/models.json`) |

## Example Output

```
$ python model_router.py --list

Task                    Model             Provider      In $/1k      Out $/1k
----------------------------------------------------------------------------
  code_review           claude-sonnet     anthropic     $0.00300     $0.01500
  security_audit        claude-sonnet     anthropic     $0.00300     $0.01500
  bug_fix               claude-sonnet     anthropic     $0.00300     $0.01500
  issue_triage          gpt-4o-mini       openai        $0.00015     $0.00060
  documentation         gpt-4o-mini       openai        $0.00015     $0.00060
  release_notes         gpt-4o-mini       openai        $0.00015     $0.00060
  quick_comment         claude-haiku      anthropic     $0.00025     $0.00125
```

```
$ python model_router.py --event pull_request

GitHub AI Model Router
==========================================
Task         : code_review
Description  : Review code changes in a pull request for correctness, style and best practices
Selected     : claude-sonnet  (claude-3-5-sonnet-20241022)
Provider     : anthropic
Cost (input) : $0.00300 / 1k tokens
Cost (output): $0.01500 / 1k tokens
```

## Setting up API Keys

Add the following secrets to your repository (**Settings → Secrets and variables → Actions**):

| Secret | Required for |
|--------|--------------|
| `ANTHROPIC_API_KEY` | `claude-sonnet`, `claude-haiku` |
| `OPENAI_API_KEY` | `gpt-4o-mini`, `gpt-4o` |

The workflow will skip the API call step if the relevant secret is not set, and will still post the routing decision as a PR/issue comment.

## Customising the model routing

Edit [`config/models.json`](config/models.json) to:
- Add or remove models
- Change cost values as pricing evolves
- Reassign tasks to different models
- Add new task types

## Project Structure

```
auto-model/
├── README.md                         ← this file
├── requirements.txt                  ← no runtime deps required
├── model_router.py                   ← CLI + routing library
├── config/
│   └── models.json                   ← model catalogue and task routing config
└── .github/
    └── workflows/
        └── model-router.yml          ← auto-triggered GitHub Actions workflow
```

## License

[MIT](LICENSE)

