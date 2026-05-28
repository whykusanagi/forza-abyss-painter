"""`fap-validate` — check an fd6.shapes v1 JSON against docs/JSON_SPEC.md.

## Why this exists

Before `fap-validate`, the only way to find out a JSON had a problem
was to try injecting it into FH6 and watch the inject silently skip
shapes (or worse, hang the editor). This CLI surfaces every issue the
spec defines, with file location pinpoints (`shapes[42].rx`), in one
pass — usable in CI, as a pre-inject sanity check, or as a 'is this
random JSON I got from Discord going to work?' triage tool.

## Usage

    fap-validate path/to/shapes.json
    fap-validate shapes.json --strict       # warnings → errors → exit 1
    fap-validate shapes.json --quiet        # exit code only
    fap-validate shapes.json --json         # machine-readable output

## Exit codes

  0  document validates clean (info-level findings allowed)
  1  warnings present (in strict mode) OR catastrophic load failure
  2  one or more ERROR-severity findings — injector will misbehave

`--strict` flips warnings into a non-zero exit; default is 'warnings
let you load but inject will be lossy', which matches GUI behavior.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forza_abyss_painter.io.validator import (
    Severity, validate_document,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fap-validate",
        description=(
            "Validate an fd6.shapes v1 JSON document against the "
            "spec at docs/JSON_SPEC.md. Reports each issue with "
            "severity, code, and JSON path. Exit 0 on clean, 1 on "
            "load failure / strict-mode warnings, 2 on errors."
        ),
    )
    parser.add_argument("path", type=Path,
                        help="path to the JSON file to validate")
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as errors (exit 1 if any)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="suppress per-issue output; just set exit code")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="emit findings as a JSON object on stdout "
                             "(suitable for piping into CI)")
    return parser


# Severity-ordered ANSI prefixes. Terminals without color support
# strip these out fine. Order chosen to read at-a-glance in a log.
_PREFIX = {
    Severity.ERROR:   "\033[1;31mERROR\033[0m",
    Severity.WARNING: "\033[1;33mWARN \033[0m",
    Severity.INFO:    "\033[1;36mINFO \033[0m",
}


def main(argv: "list[str] | None" = None) -> int:
    args = _build_parser().parse_args(argv)

    # Load + parse. Catastrophic failures (file missing, malformed JSON)
    # exit 1 — they're not 'validation errors', they're 'can't even start'.
    try:
        raw = args.path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"fap-validate: file not found: {args.path}",
              file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"fap-validate: cannot read {args.path}: {exc}",
              file=sys.stderr)
        return 1

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"fap-validate: {args.path}: JSON parse error: {exc}",
              file=sys.stderr)
        return 1

    issues = validate_document(data)

    # Machine-readable mode: one JSON object on stdout, ignore --quiet,
    # caller decides what to do.
    if args.json_output:
        out = {
            "path": str(args.path),
            "issues": [
                {
                    "severity": i.severity.value,
                    "code": i.code,
                    "message": i.message,
                    "path": i.path,
                }
                for i in issues
            ],
            "summary": _summarize(issues),
        }
        print(json.dumps(out, indent=2))
        return _exit_code(issues, strict=args.strict)

    # Human-readable mode: pretty-print each issue, then a one-line
    # summary. --quiet suppresses both, just sets the exit code.
    if not args.quiet:
        for issue in issues:
            prefix = _PREFIX.get(issue.severity, str(issue.severity))
            loc = f" at {issue.path}" if issue.path else ""
            print(f"  {prefix}  [{issue.code}]{loc}: {issue.message}")
        if not issues:
            print(f"  \033[1;32mOK\033[0m  {args.path} — no findings")
        else:
            print(f"\n  {_summary_line(issues)}")

    return _exit_code(issues, strict=args.strict)


def _summarize(issues: list) -> dict:
    """Counts by severity for the --json output."""
    return {
        "errors":   sum(1 for i in issues if i.severity is Severity.ERROR),
        "warnings": sum(1 for i in issues if i.severity is Severity.WARNING),
        "info":     sum(1 for i in issues if i.severity is Severity.INFO),
        "total":    len(issues),
    }


def _summary_line(issues: list) -> str:
    s = _summarize(issues)
    parts = []
    if s["errors"]:
        parts.append(f"\033[1;31m{s['errors']} error(s)\033[0m")
    if s["warnings"]:
        parts.append(f"\033[1;33m{s['warnings']} warning(s)\033[0m")
    if s["info"]:
        parts.append(f"\033[1;36m{s['info']} info\033[0m")
    return ", ".join(parts) if parts else "no findings"


def _exit_code(issues: list, *, strict: bool) -> int:
    """0 = clean (or info-only), 1 = strict-mode warnings, 2 = errors."""
    if any(i.severity is Severity.ERROR for i in issues):
        return 2
    if strict and any(i.severity is Severity.WARNING for i in issues):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
