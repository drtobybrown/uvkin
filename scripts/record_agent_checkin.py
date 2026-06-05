#!/usr/bin/env python3
"""Append a structured check-in entry for lead alignment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", default="KILOGAS066", help="e.g. KILOGAS066")
    p.add_argument("--worker", required=True, help="Worker role or agent name")
    p.add_argument(
        "--leads",
        required=True,
        help="Comma-separated: science, dev, ops",
    )
    p.add_argument("--summary", required=True, help="One-line status")
    p.add_argument(
        "--results-base",
        type=Path,
        default=Path("/arc/projects/KILOGAS/analysis/toby_sandbox/results"),
    )
    p.add_argument("--science-status", type=Path, default=None)
    p.add_argument("--decision", default="", help="Lead decisions / approval notes")
    p.add_argument("--next-action", default="", help="Planned next step")
    args = p.parse_args(argv)

    checkin_dir = Path(args.results_base) / args.campaign / "agent_checkins"
    checkin_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkin_dir / "CHECKIN_LOG.md"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    leads = [x.strip() for x in args.leads.split(",") if x.strip()]

    status_block = ""
    if args.science_status and Path(args.science_status).is_file():
        st = json.loads(Path(args.science_status).read_text(encoding="utf-8"))
        status_block = (
            f"- overall: `{st.get('overall')}`\n"
            f"- next_action: {st.get('next_action', 'n/a')}\n"
            f"- failed tiers: "
            f"t1={st.get('tier1_pipeline', {}).get('failed', [])} "
            f"t2={st.get('tier2_dataset_similarity', {}).get('failed', [])} "
            f"t3={st.get('tier3_science', {}).get('failed', [])}\n"
        )

    entry = f"""## Check-in {ts}

| Field | Value |
|-------|-------|
| Worker | `{args.worker}` |
| Leads | {', '.join(leads)} |
| Summary | {args.summary} |

{status_block}"""
    if args.decision:
        entry += f"\n**Lead decisions:** {args.decision}\n"
    if args.next_action:
        entry += f"\n**Next action:** {args.next_action}\n"
    entry += "\n---\n\n"

    if not log_path.is_file():
        header = (
            f"# Agent check-in log — {args.campaign}\n\n"
            "See uvkin `docs/agents/lead-checkins.md` for cadence and gates.\n\n"
            "---\n\n"
        )
        log_path.write_text(header + entry, encoding="utf-8")
    else:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(entry)

    meta = {
        "timestamp_utc": ts,
        "worker": args.worker,
        "leads": leads,
        "summary": args.summary,
        "science_status": str(args.science_status) if args.science_status else None,
    }
    (checkin_dir / "latest_checkin.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Appended to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
