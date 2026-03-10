#!/usr/bin/env python3
"""
model_router.py
---------------
GitHub AI Model Router — selects the most cost-effective AI model for each
GitHub task (code review, issue triage, documentation, etc.) and optionally
invokes the model via its API.

Usage examples
--------------
# Determine the best model for a GitHub event type
python model_router.py --event pull_request

# Determine the best model for an explicit task
python model_router.py --task documentation

# Read a real GitHub event payload from a file and call the AI
python model_router.py --event-file $GITHUB_EVENT_PATH --prompt "Review this PR" --call-api

# List all task→model mappings and costs
python model_router.py --list
"""

from typing import Optional
import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = Path(__file__).parent / "config" / "models.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------
def infer_task(event_name: str, event_payload: dict, config: dict) -> str:
    """
    Refine a GitHub event into a specific task using payload details.
    Falls back to the simple event→task map when details are unavailable.
    """
    routing = config["task_routing"]
    default_task = config["event_to_task_map"].get(event_name, "issue_triage")

    if event_name == "pull_request":
        # Detect documentation-only PRs by changed file extensions
        files_changed = [
            f.get("filename", "")
            for f in event_payload.get("pull_request", {})
            .get("changed_files_detail", [])
        ]
        doc_exts = {".md", ".rst", ".txt", ".adoc"}
        if files_changed and all(
            Path(f).suffix.lower() in doc_exts for f in files_changed
        ):
            return "documentation"
        return "code_review"

    if event_name == "issues":
        labels = [
            lbl.get("name", "").lower()
            for lbl in event_payload.get("issue", {}).get("labels", [])
        ]
        if "bug" in labels:
            return "bug_fix"
        if "security" in labels:
            return "security_audit"
        return "issue_triage"

    if event_name == "release":
        return "release_notes"

    return default_task


def select_model(task: str, config: dict) -> tuple[dict, dict]:
    """
    Return the model dict for the preferred model for *task*, falling back
    to the configured fallback model.
    """
    routing = config["task_routing"].get(task)
    if not routing:
        raise ValueError(f"Unknown task '{task}'. "
                         f"Valid tasks: {', '.join(config['task_routing'])}")

    model_index = {m["alias"]: m for m in config["models"]}
    alias = routing["preferred_model"]
    model = model_index.get(alias)
    if not model:
        alias = routing["fallback_model"]
        model = model_index.get(alias)
    if not model:
        raise RuntimeError(f"Neither preferred nor fallback model found for task '{task}'.")

    return model, routing


def estimate_cost(prompt: str, model: dict) -> dict:
    """Very rough cost estimate (assumes output ≈ 20 % of input tokens)."""
    approx_input_tokens = len(prompt.split()) * 1.3
    approx_output_tokens = approx_input_tokens * 0.2
    cost = (
        approx_input_tokens / 1000 * model["cost_per_1k_input_tokens"]
        + approx_output_tokens / 1000 * model["cost_per_1k_output_tokens"]
    )
    return {
        "approx_input_tokens": int(approx_input_tokens),
        "approx_output_tokens": int(approx_output_tokens),
        "estimated_cost_usd": round(cost, 6),
    }


# ---------------------------------------------------------------------------
# API call helpers
# ---------------------------------------------------------------------------
def call_anthropic(model_name: str, prompt: str, api_key: str) -> str:
    import urllib.request

    payload = json.dumps({
        "model": model_name,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


def call_openai(model_name: str, prompt: str, api_key: str) -> str:
    import urllib.request

    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def call_model(model: dict, prompt: str) -> str:
    """Call the selected model's API and return the response text."""
    provider = model["provider"]
    model_name = model["name"]

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            sys.exit("[error] ANTHROPIC_API_KEY environment variable is not set.")
        return call_anthropic(model_name, prompt, api_key)

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            sys.exit("[error] OPENAI_API_KEY environment variable is not set.")
        return call_openai(model_name, prompt, api_key)

    sys.exit(f"[error] Unsupported provider '{provider}'.")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_selection(task: str, model: dict, routing: dict, prompt: Optional[str] = None):
    print(f"\nGitHub AI Model Router")
    print("=" * 42)
    print(f"Task         : {task}")
    print(f"Description  : {routing['description']}")
    print(f"Selected     : {model['alias']}  ({model['name']})")
    print(f"Provider     : {model['provider']}")
    print(f"Cost (input) : ${model['cost_per_1k_input_tokens']:.5f} / 1k tokens")
    print(f"Cost (output): ${model['cost_per_1k_output_tokens']:.5f} / 1k tokens")
    if prompt:
        cost_info = estimate_cost(prompt, model)
        print(f"Est. cost    : ~${cost_info['estimated_cost_usd']:.6f}  "
              f"({cost_info['approx_input_tokens']} in / "
              f"{cost_info['approx_output_tokens']} out tokens)")
    print()


def print_all_mappings(config: dict):
    print(f"\n{'Task':<22}  {'Model':<16}  {'Provider':<12}  "
          f"{'In $/1k':>10}  {'Out $/1k':>10}")
    print("-" * 76)
    model_index = {m["alias"]: m for m in config["models"]}
    for task, routing in config["task_routing"].items():
        model = model_index.get(routing["preferred_model"], {})
        print(
            f"  {task:<20}  {routing['preferred_model']:<16}  "
            f"{model.get('provider','?'):<12}  "
            f"${model.get('cost_per_1k_input_tokens', 0):.5f}     "
            f"${model.get('cost_per_1k_output_tokens', 0):.5f}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Route GitHub tasks to the best cost-effective AI model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--event",
        help="GitHub event name (e.g. pull_request, issues, push, release, "
             "issue_comment).",
    )
    group.add_argument(
        "--task",
        help="Explicit task name (code_review, security_audit, bug_fix, "
             "issue_triage, documentation, release_notes, quick_comment).",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List all task→model mappings and exit.",
    )
    parser.add_argument(
        "--event-file",
        help="Path to a JSON file containing the GitHub event payload "
             "(e.g. $GITHUB_EVENT_PATH).",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt to send to the selected model (required for --call-api).",
    )
    parser.add_argument(
        "--call-api",
        action="store_true",
        help="Actually invoke the selected model's API with --prompt.",
    )
    parser.add_argument(
        "--output",
        help="Write the routing decision + response to a JSON file.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the models.json config file.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(Path(args.config))

    if args.list:
        print_all_mappings(config)
        return

    # --- Determine task ---
    event_payload = {}
    if args.event_file:
        with open(args.event_file) as f:
            event_payload = json.load(f)

    if args.task:
        task = args.task
    elif args.event:
        task = infer_task(args.event, event_payload, config)
    else:
        parser_err = ("Specify one of --event, --task, or --list.\n"
                      "Run with --help for usage.")
        sys.exit(parser_err)

    # --- Select model ---
    model, routing = select_model(task, config)

    # --- Print selection ---
    print_selection(task, model, routing, prompt=args.prompt)

    result = {
        "task": task,
        "selected_model": model["alias"],
        "model_name": model["name"],
        "provider": model["provider"],
        "description": routing["description"],
        "cost_per_1k_input_tokens": model["cost_per_1k_input_tokens"],
        "cost_per_1k_output_tokens": model["cost_per_1k_output_tokens"],
    }

    # --- Optionally call the API ---
    if args.call_api:
        if not args.prompt:
            sys.exit("[error] --prompt is required when using --call-api.")
        print("Calling API...")
        response = call_model(model, args.prompt)
        print("\n--- Model Response ---")
        print(response)
        print("---------------------\n")
        result["response"] = response

    # --- Optional JSON output ---
    if args.output:
        if args.prompt:
            result["cost_estimate"] = estimate_cost(args.prompt, model)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Routing decision written to {args.output}")

    return result


if __name__ == "__main__":
    main()
