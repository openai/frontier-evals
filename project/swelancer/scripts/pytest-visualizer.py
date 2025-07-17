#!/usr/bin/env python3
"""
This script:

- Recursively finds every `run.log` under the supplied *run-group* path.
- Loads each line as JSON, keeping only records whose `event` field begins
    with the canonical *pytest output* prefix (default: `'pytest output'`).
- Launches a lightweight Flask viewer to render the records in a table.

Usage
-----
    pip install flask
    python pytest_viewer.py /path/to/2025-06-09T17-21-53-UTC_run-group_dummy
    # → Navigate to http://127.0.0.1:5000

If you need a non-default prefix or port:

    python pytest_viewer.py /path/to/run-group --prefix "pytest out" --port 5050
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path
from typing import Any, NoReturn

from flask import Flask, Response, abort, jsonify, render_template_string

DEFAULT_PREFIX = "pytest output"
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pytest Log Viewer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 1em; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    tr:nth-child(even) { background: #f8f8f8; }
    tr:hover { background: #e0e0e0; cursor: pointer; }
    .failed { color: #c00; font-weight: bold; }
    .error  { color: #d96f00; font-weight: bold; }
    .passed { color: #090; font-weight: bold; }
    th.sortable { cursor: pointer; }
    th.sortable::after { content: " ⇅"; font-size: 0.8em; color: #555; }
    th.sortable[data-asc="true"]::after  { content: " ↓"; }
    th.sortable[data-asc="false"]::after { content: " ↑"; }
    .filter-btn { margin-left: 0.5em; padding: 2px 6px; border: 1px solid #aaa; background: #f0f0f0; cursor: pointer; }
    .filter-btn.active { background: #ddd; font-weight: bold; }
    #text-filter { margin-left: 0.75em; padding: 3px 6px; width: 260px; }
    #metrics-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5em; }
    #filters { white-space: nowrap; }
    #match-info { flex: 1; text-align: center; font-style: italic; }
    .hl { background: #ffeb3b; padding: 0 1px; }
    tr.detail { display: none; }
    pre { background: #f7f7f7; padding: 1em; overflow: auto; white-space: pre-wrap; margin: 0; }
  </style>
</head>
<body>
  <div id="metrics-bar">
    <div id="metrics">
      <strong>Total:</strong> <span>{{ total }}</span> |
      <strong>Passed:</strong> <span class="passed">{{ passed }}</span> |
      <strong>Failed:</strong> <span class="failed">{{ failed }}</span> |
      <strong>Errored:</strong> <span class="error">{{ error }}</span>
    </div>
    <div id="match-info"><span id="match-count"></span></div>
    <div id="filters">
      Filter :
      <button class="filter-btn active" data-filter="all">All</button>
      <button class="filter-btn" data-filter="passed">Passed</button>
      <button class="filter-btn" data-filter="failed">Failed</button>
      <button class="filter-btn" data-filter="error">Errored</button>
      <input type="text" id="text-filter" placeholder="Type to filter (case insensitive)..." />
    </div>
  </div>
  <table id="log-table">
    <thead><tr>
      <th class="sortable" data-key="issue_id">Issue&nbsp;ID</th>
      <th class="sortable" data-key="status">Status</th>
    </tr></thead>
    <tbody></tbody>
  </table>



<script>
let logData = [];

let currentFilter = 'all';
let textFilter = '';

// Escape HTML entities to keep output safe
function esc(str) {
  return str.replace(/[&<>"']/g, m =>
    ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[m])
  );
}

// Wrap matching substring in <span class="hl">…</span>
function highlightMatch(str) {
  const safe = esc(str);
  if (!textFilter) return safe;

  // Escape any regex‑significant characters in the search term
  const escaped = textFilter.replace(/[\\^$.*+?()[\\]{}|]/g, '\\$&');

  // Build the regexp; use try/catch in case something still slips through
  let re;
  try {
    re = new RegExp(escaped, 'gi');
  } catch (_) {
    return safe; // if invalid, give up highlighting but keep rendering
  }

  return safe.replace(re, m => `<span class="hl">${m}</span>`);
}

function applyFilter(data) {
  // status filter first
  let out = currentFilter === 'all' ? data : data.filter(d => d.status === currentFilter);

  // free‑text filter (case‑insensitive substring on issue_id or output)
  if (textFilter) {
    out = out.filter(d =>
      d.issue_id.toLowerCase().includes(textFilter) ||
      (d.output && d.output.toLowerCase().includes(textFilter))
    );
  }
  return out;
}

function renderTable(data) {
  const tbody = document.querySelector('#log-table tbody');
  tbody.innerHTML = '';
  data.forEach(log => {
    const tr = document.createElement('tr');
    tr.innerHTML =
      `<td>${highlightMatch(log.issue_id)}</td><td class="${log.status}">${log.status}</td>`;
    tr.addEventListener('click', () => toggleDetail(tr, log.id));
    tbody.appendChild(tr);
  });
  // update match count
  const mc = document.getElementById('match-count');
  mc.textContent = textFilter || currentFilter !== 'all'
    ? `Matches: ${data.length}`
    : '';
}

async function fetchLog(id) {
  const res = await fetch('/api/logs/' + id);
  return res.json();
}

async function toggleDetail(rowEl, id) {
  const next = rowEl.nextElementSibling;
  // If a detail row already exists, just toggle its visibility
  if (next && next.classList.contains('detail')) {
    next.style.display = next.style.display === 'none' ? 'table-row' : 'none';
    return;
  }

  // Otherwise fetch detail and insert a new row
  const log = await fetchLog(id);
  const detailTr = document.createElement('tr');
  detailTr.classList.add('detail');

  const td = document.createElement('td');
  td.colSpan = 2;              // match the summary table’s column count
  td.innerHTML = `<pre>${highlightMatch(log.output)}</pre>`;
  detailTr.appendChild(td);

  rowEl.parentNode.insertBefore(detailTr, rowEl.nextSibling);
  detailTr.style.display = 'table-row';

  // Allow double‑click on the expanded output to close it again
  detailTr.addEventListener('dblclick', () => { detailTr.style.display = 'none'; });
}

async function loadSummary() {
  const res = await fetch('/api/logs');
  logData = await res.json();
  renderTable(applyFilter(logData));
}

loadSummary();

// ---- Filter buttons ----
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    // visual toggle
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    currentFilter = btn.dataset.filter;
    renderTable(applyFilter(logData));
  });
});

// ---- Free‑text filter ----
document.getElementById('text-filter').addEventListener('input', e => {
  textFilter = e.target.value.toLowerCase().trim();
  renderTable(applyFilter(logData));
});

document.querySelectorAll('#log-table thead th.sortable').forEach(th => {
  th.addEventListener('click', () => {
    // Clear sort arrows on all headers except the one clicked
    document.querySelectorAll('#log-table thead th.sortable').forEach(th2 => {
      if (th2 !== th) th2.removeAttribute('data-asc');
    });
    const key = th.dataset.key;
    const asc = th.dataset.asc === 'true';
    th.dataset.asc = (!asc).toString();
    const sorted = [...applyFilter(logData)].sort((a, b) => {
      if (a[key] < b[key]) return asc ? 1 : -1;
      if (a[key] > b[key]) return asc ? -1 : 1;
      return 0;
    });
    renderTable(sorted);
  });
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Collection logic
# ---------------------------------------------------------------------------


def get_issue_id(issue_folder_name: str) -> str:
    """
    issue_folder_name consists of:
        <int>  OR  <int>_<int>   followed by '_' + uuid
    Return everything before the final '_' (the uuid separator).
    """
    return issue_folder_name.rsplit("_", 1)[0]


def collect_records(root: Path, prefix: str) -> list[dict[str, Any]]:
    """
    Return pytest-output records from all run.log files under *root*."""

    records: list[dict[str, Any]] = []

    for run_log in root.rglob("run.log"):
        try:
            issue_folder = run_log.relative_to(root).parts[0]
            issue_id = get_issue_id(issue_folder)

            with run_log.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()

                    try:
                        obj = ast.literal_eval(line)
                    except (ValueError, SyntaxError):
                        continue

                    if not isinstance(obj, dict):
                        continue

                    event = obj.get("event", "")

                    if not isinstance(event, str):
                        continue

                    if not event.casefold().startswith(prefix.casefold()):
                        continue

                    obj["issue_id"] = issue_id

                    records.append(obj)
        except (OSError, IOError):
            continue

    return records


# ---------------------------------------------------------------------------
# Viewer logic
# ---------------------------------------------------------------------------


def launch_viewer(records: list[dict[str, Any]], port: int) -> None:
    """Serve the original /api/logs endpoints plus the index UI."""
    app = Flask(__name__)

    # Convert collected records to the simplified structure the UI expects.
    def _to_log(rec: dict[str, Any]) -> dict[str, str]:
        text = rec.get("event", "")

        assert isinstance(text, str), "Expected 'event' to be a string"

        lower = text.casefold()

        if "failed" in lower:
            status = "failed"
        elif "error" in lower:
            status = "error"
        elif "passed" in lower:
            status = "passed"
        else:
            status = "unknown"

        return {
            "issue_id": rec.get("issue_id") or rec.get("id") or rec.get("name") or "",
            "status": status,
            "output": lower,
        }

    LOGS = [_to_log(r) for r in records]

    # -------------------------------------------------------------------
    # Summary metrics
    # -------------------------------------------------------------------
    status_counts = Counter(log["status"] for log in LOGS)
    METRICS = {
        "total": len(LOGS),
        "passed": status_counts.get("passed", 0),
        "failed": status_counts.get("failed", 0),
        "error": status_counts.get("error", 0),
    }

    # ---------------- API ---------------- #
    @app.route("/api/logs")
    def list_logs() -> Response:  # noqa: D401
        return jsonify(
            [
                {
                    "id": idx,
                    "issue_id": log["issue_id"],
                    "status": log["status"],
                    "output": log["output"],
                }
                for idx, log in enumerate(LOGS)
            ]
        )

    @app.route("/api/logs/<int:log_id>")
    def get_log(log_id: int) -> Response | NoReturn:  # noqa: D401
        if 0 <= log_id < len(LOGS):
            return jsonify(LOGS[log_id])

        return abort(404, description="Log not found")

    # --------------- UI ----------------- #
    @app.route("/")
    def index() -> str:  # noqa: D401
        return render_template_string(
            HTML_TEMPLATE,
            total=METRICS["total"],
            passed=METRICS["passed"],
            failed=METRICS["failed"],
            error=METRICS["error"],
        )

    # No auto-reloader or debug – mirror original behaviour.
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


# ---------------------------------------------------------------------------
# CLI glue
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser(description="View pytest outputs from a run-group")
    p.add_argument("path", type=Path, help="Path to <RUN-GROUP> directory")
    p.add_argument("--prefix", default=DEFAULT_PREFIX, help="Event prefix to match")
    p.add_argument("--port", "-p", type=int, default=5000, help="Port for the web UI")
    return p.parse_args()


def main() -> None:  # noqa: D401
    args = parse_args()
    path: Path = args.path.expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"error: {path} is not a directory")

    records = collect_records(path, args.prefix)
    launch_viewer(records, args.port)


if __name__ == "__main__":
    main()
