from __future__ import annotations

import re

import pandas as pd

BEHAVIOR_LABELS = (
    "BENIGN",
    "RECONNAISSANCE",
    "BRUTE_FORCE",
    "C2_BEACONING",
    "EXFILTRATION",
    "EXPLOITATION",
    "DOS",
    "UNKNOWN",
)

_ALIASES = {
    "normal": "BENIGN",
    "attack": "UNKNOWN",
    "scan": "RECONNAISSANCE",
    "recon": "RECONNAISSANCE",
    "bruteforce": "BRUTE_FORCE",
    "brute_force": "BRUTE_FORCE",
    "c2": "C2_BEACONING",
    "command_and_control": "C2_BEACONING",
    "exfil": "EXFILTRATION",
    "exploit": "EXPLOITATION",
    "ddos": "DOS",
}


def normalize_behavior_label(value: object) -> str:
    """Normalize analyst labels while preserving a finite, documented taxonomy."""
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    token = _ALIASES.get(token, token.upper())
    return token if token in BEHAVIOR_LABELS else "UNKNOWN"


def _number(row: pd.Series, *names: str) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            try:
                return float(row[name])
            except (TypeError, ValueError):
                pass
    return 0.0


def guess_behavior(row: pd.Series) -> tuple[str, str]:
    """Return a conservative pseudo-label and the observable rule that produced it."""
    category = row.get("attack_cat")
    if pd.notna(category) and str(category).strip().lower() not in {"", "unknown", "nan"}:
        label = normalize_behavior_label(category)
        if label != "UNKNOWN":
            return label, "mapped from attack_cat"

    packets = _number(row, "packets", "spkts", "src_pkts") + _number(row, "dpkts", "dst_pkts")
    total_len = _number(row, "total_len", "bytes", "sbytes") + _number(row, "dbytes")
    duration = _number(row, "duration", "dur")
    port = int(_number(row, "destination_port", "dst_port", "dport", "port_b"))

    if packets >= 1000 and duration <= 60:
        return "DOS", "high packet rate"
    if total_len >= 100_000_000:
        return "EXFILTRATION", "very large transfer volume"
    if port in {22, 23, 3389, 5900} and packets >= 50:
        return "BRUTE_FORCE", "repeated remote-access traffic"
    return "UNKNOWN", "insufficient observable evidence"


def prepare_behavior_labels(frame: pd.DataFrame, target: str) -> tuple[pd.Series, pd.DataFrame]:
    """Fill absent labels with auditable pseudo-labels marked by a trailing asterisk."""
    supplied = frame[target] if target in frame else pd.Series(pd.NA, index=frame.index)
    labels: list[str] = []
    records: list[dict] = []
    for position, (index, row) in enumerate(frame.iterrows()):
        value = supplied.loc[index]
        if pd.notna(value) and str(value).strip():
            label = normalize_behavior_label(value)
            labels.append(label)
            records.append({"row": position, "behavior_label": label, "pseudo_label": False, "reason": "provided"})
        else:
            label, reason = guess_behavior(row)
            marked = f"{label}*"
            labels.append(label)
            records.append({"row": position, "behavior_label": marked, "pseudo_label": True, "reason": reason})
    return pd.Series(labels, index=frame.index, name=target), pd.DataFrame(records)
