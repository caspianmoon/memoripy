from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

from .client import MemoryClient
from .evals import load_contracts, run_contracts
from .service import serve_http


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memoripy",
        description="Evidence-first memory tooling for AI agents.",
    )
    parser.add_argument("--version", action="version", version="memoripy 0.4.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Inspect a memory store for pollution, conflicts, and provenance gaps.")
    audit.add_argument("path", help="Memoripy store directory or state JSON file.")
    audit.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    audit.add_argument("--html", dest="html_path", help="Write a shareable local HTML report.")
    audit.add_argument("--fail-on", choices=("critical", "high", "medium", "low"), help="Return exit code 2 at or above this severity.")

    inspect = subparsers.add_parser("inspect", help="Inspect memories, history, evidence, and write decisions.")
    inspect.add_argument("path")
    inspect.add_argument("--memory-id")
    inspect.add_argument("--json", action="store_true")

    evaluate = subparsers.add_parser("eval", help="Run memory contracts against the v4 runtime.")
    evaluate.add_argument("contract_file", nargs="?", help="Optional JSON contract file.")
    evaluate.add_argument("--json", action="store_true")

    recover = subparsers.add_parser("recover", help="Restore the last known-good file-store backup.")
    recover.add_argument("path")
    recover.add_argument("--json", action="store_true")

    serve = subparsers.add_parser("serve", help="Run the local-development HTTP service.")
    serve.add_argument("path")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--api-key", default=os.environ.get("MEMORIPY_API_KEY"))

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        return _audit(args)
    if args.command == "inspect":
        return _inspect(args)
    if args.command == "eval":
        return _eval(args)
    if args.command == "recover":
        return _recover(args)
    if args.command == "serve":
        return _serve(args)
    return 1


def _audit(args: argparse.Namespace) -> int:
    client = MemoryClient.from_path(args.path)
    report = client.audit().to_dict()
    if args.html_path:
        Path(args.html_path).write_text(_audit_html(report), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"Memoripy memory audit: {args.path}")
        print(
            f"memories={report['memory_count']} evidence={report['evidence_count']} "
            f"findings={report['finding_count']} citation_coverage={report['metrics']['citation_coverage']:.1%}"
        )
        if not report["findings"]:
            print("PASS No memory-quality findings detected.")
        for finding in report["findings"]:
            ids = ", ".join(finding["memory_ids"][:3])
            suffix = f" [{ids}]" if ids else ""
            print(f"{finding['severity'].upper():8} {finding['code']}: {finding['message']}{suffix}")
            if finding.get("suggested_action"):
                print(f"         fix: {finding['suggested_action']}")
        if args.html_path:
            print(f"HTML report: {args.html_path}")
    if args.fail_on:
        threshold = {"critical": 0, "high": 1, "medium": 2, "low": 3}[args.fail_on]
        severity = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        if any(severity.get(item["severity"], 9) <= threshold for item in report["findings"]):
            return 2
    return 0


def _inspect(args: argparse.Namespace) -> int:
    client = MemoryClient.from_path(args.path)
    if args.memory_id:
        payload = client.explain(memory_id=args.memory_id)
    else:
        payload = client.get_all()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.memory_id:
        memory = payload["memory"]
        print(f"{memory['record_id']} {memory['kind']} {memory['state']}")
        print(f"value: {memory['value']}")
        print(f"trust: {memory['trust_level']} durability: {memory['durability']}")
        print(f"valid: {memory['valid_from']} -> {memory['valid_to'] or 'current'}")
        print(f"evidence: {', '.join(memory['citation_evidence_ids']) or 'none'}")
        print("history:")
        for version in payload["history"]:
            print(
                f"- {version['created_at']} {version['action']} {version['value']} "
                f"[{version['state']}]"
            )
    else:
        for item in payload["results"]:
            memory = item["memory"]
            print(
                f"{memory['record_id']} {memory['kind']:20} {memory['state']:12} "
                f"{memory['summary']}"
            )
    return 0


def _eval(args: argparse.Namespace) -> int:
    summary = run_contracts(load_contracts(args.contract_file))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"memory contracts: {summary['passed_count']}/{summary['contract_count']} passed "
            f"({summary['score_ratio']:.1%})"
        )
        for result in summary["results"]:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{status} {result['name']}: {result['description']}")
            for failure in result["failures"]:
                print(f"  {failure}")
    return 0 if summary["failed_count"] == 0 else 1


def _recover(args: argparse.Namespace) -> int:
    result = MemoryClient.from_path(args.path).recover()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"Recovered {result['state_path']} from {result['backup_path']}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    client = MemoryClient.from_path(args.path)
    server = serve_http(
        host=args.host,
        port=args.port,
        client=client,
        api_key=args.api_key,
    )
    auth = "enabled" if args.api_key else "disabled"
    print(f"Memoripy v4 listening on http://{args.host}:{args.port} (API key {auth})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _audit_html(report: dict[str, Any]) -> str:
    finding_rows = []
    for finding in report["findings"]:
        finding_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(finding['severity'].upper())}</strong></td>"
            f"<td><code>{html.escape(finding['code'])}</code></td>"
            f"<td>{html.escape(finding['message'])}</td>"
            f"<td>{html.escape(', '.join(finding['memory_ids']))}</td>"
            f"<td>{html.escape(finding.get('suggested_action') or '')}</td>"
            "</tr>"
        )
    rows = "\n".join(finding_rows) or '<tr><td colspan="5">No findings</td></tr>'
    metrics = "".join(
        f"<li><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</li>"
        for key, value in report["metrics"].items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memoripy Memory Audit</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; color: #111; }}
h1 {{ margin-bottom: 0; }}
.sub {{ color: #555; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 24px; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top; }}
th {{ background: #f5f5f5; }}
code {{ background: #f4f4f4; padding: 2px 4px; }}
</style>
</head>
<body>
<h1>Memoripy Memory Audit</h1>
<p class="sub">Generated {html.escape(report['generated_at'])}</p>
<p><strong>{report['memory_count']}</strong> memories, <strong>{report['evidence_count']}</strong> evidence items, <strong>{report['finding_count']}</strong> findings.</p>
<h2>Metrics</h2><ul>{metrics}</ul>
<h2>Findings</h2>
<table><thead><tr><th>Severity</th><th>Code</th><th>Finding</th><th>Memories</th><th>Suggested action</th></tr></thead><tbody>{rows}</tbody></table>
</body>
</html>"""


if __name__ == "__main__":
    raise SystemExit(main())
