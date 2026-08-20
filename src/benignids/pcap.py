from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_label_manifest(path: str | Path) -> pd.DataFrame:
    """Load pcap_path,label,attack_cat metadata kept outside packet bytes."""
    manifest = pd.read_csv(path)
    required = {"pcap_path", "label"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"PCAP label manifest is missing columns: {sorted(missing)}")
    return manifest


def pcap_to_flows(
    pcap_path: str | Path,
    label: int | None = None,
    attack_cat: str = "unknown",
    session_timeout: float = 300.0,
) -> pd.DataFrame:
    """Aggregate a capture into time-bounded bidirectional session records.

    Labels are required when building training data and omitted for inference captures.
    """
    if session_timeout <= 0:
        raise ValueError("session_timeout must be positive")
    try:
        from scapy.all import IP, TCP, UDP, IPv6, PcapReader
    except ImportError as exc:
        raise RuntimeError("Install PCAP support with: pip install -e '.[pcap]'") from exc

    sessions = defaultdict(list)
    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            network = packet.getlayer(IP) or packet.getlayer(IPv6)
            if network is None:
                continue
            transport = packet.getlayer(TCP) or packet.getlayer(UDP)
            if packet.haslayer(TCP):
                protocol = "tcp"
            elif packet.haslayer(UDP):
                protocol = "udp"
            else:
                protocol = str(network.nh)
            source_port = int(getattr(transport, "sport", 0))
            destination_port = int(getattr(transport, "dport", 0))
            endpoint_a = (str(network.src), source_port)
            endpoint_b = (str(network.dst), destination_port)
            left, right = sorted((endpoint_a, endpoint_b))
            key = (left, right, protocol)
            timestamp = float(packet.time)
            session_list = sessions[key]
            if not session_list or timestamp - session_list[-1]["timestamps"][-1] > session_timeout:
                session_list.append(
                    {
                        "initiator": endpoint_a,
                        "responder": endpoint_b,
                        "timestamps": [],
                        "sizes": [],
                        "directions": [],
                        "tcp_flags": [],
                        "payload": bytearray(),
                    }
                )
            flow = session_list[-1]
            flow["timestamps"].append(timestamp)
            flow["sizes"].append(len(packet))
            flow["directions"].append(1 if endpoint_a == flow["initiator"] else -1)
            flags = int(getattr(packet.getlayer(TCP), "flags", 0)) if packet.haslayer(TCP) else 0
            flow["tcp_flags"].append(flags)
            payload = bytes(getattr(transport, "payload", b"")) if transport else b""
            if len(flow["payload"]) < 64:
                flow["payload"].extend(payload[: 64 - len(flow["payload"])])

    rows = []
    for (left, right, protocol), session_list in sessions.items():
        for session_number, flow in enumerate(session_list, start=1):
            timestamps = np.asarray(flow["timestamps"], dtype=float)
            sizes = np.asarray(flow["sizes"], dtype=float)
            directions = np.asarray(flow["directions"], dtype=int)
            gaps = np.diff(timestamps)
            active_gaps = gaps[gaps <= 1.0]
            duration = float(timestamps[-1] - timestamps[0])
            changes = int(np.count_nonzero(np.diff(directions)))
            mean_gap = float(gaps.mean()) if len(gaps) else 0.0
            std_gap = float(gaps.std()) if len(gaps) else 0.0
            gap_cv = std_gap / mean_gap if mean_gap > 0 else 0.0
            forward = directions == 1
            reverse = directions == -1
            row = {
                "session_id": f"{Path(pcap_path).stem}:{len(rows) + 1}",
                "session_number": session_number,
                "initiator": flow["initiator"][0],
                "initiator_port": flow["initiator"][1],
                "responder": flow["responder"][0],
                "responder_port": flow["responder"][1],
                "endpoint_a": left[0],
                "port_a": left[1],
                "endpoint_b": right[0],
                "port_b": right[1],
                "protocol": protocol,
                "session_state": _session_state(protocol, flow["tcp_flags"]),
                "start_time": float(timestamps[0]),
                "end_time": float(timestamps[-1]),
                "packets": len(sizes),
                "total_len": int(sizes.sum()),
                "duration": duration,
                "forward_packets": int(forward.sum()),
                "reverse_packets": int(reverse.sum()),
                "forward_bytes": int(sizes[forward].sum()),
                "reverse_bytes": int(sizes[reverse].sum()),
                "packet_size_mean": float(sizes.mean()),
                "packet_size_std": float(sizes.std()),
                "interarrival_mean": mean_gap,
                "interarrival_std": std_gap,
                "interarrival_cv": gap_cv,
                "packets_per_second": len(sizes) / max(duration, 1e-6),
                "bytes_per_second": sizes.sum() / max(duration, 1e-6),
                "direction_changes": changes,
                "request_response_turns": changes,
                "burst_count": 1 + int(np.count_nonzero(gaps > 1.0)),
                "idle_fraction": float((gaps > 1.0).mean()) if len(gaps) else 0.0,
                "timing_regularity": 1.0 / (1.0 + gap_cv),
                "active_interarrival_mean": float(active_gaps.mean()) if len(active_gaps) else 0.0,
                "syn_count": sum(bool(flags & 0x02) for flags in flow["tcp_flags"]),
                "fin_count": sum(bool(flags & 0x01) for flags in flow["tcp_flags"]),
                "rst_count": sum(bool(flags & 0x04) for flags in flow["tcp_flags"]),
                "payload": bytes(flow["payload"]).hex(),
                "capture_id": Path(pcap_path).stem,
            }
            if label is not None:
                row.update({"label": int(label), "attack_cat": attack_cat})
            rows.append(row)
    return pd.DataFrame(rows)


def _session_state(protocol: str, flags: list[int]) -> str:
    if protocol != "tcp":
        return "ACTIVE" if flags else "OBSERVED"
    if any(flag & 0x04 for flag in flags):
        return "RESET"
    if any(flag & 0x01 for flag in flags):
        return "CLOSED"
    if any(flag & 0x02 for flag in flags):
        return "ESTABLISHED" if any(flag & 0x10 for flag in flags) else "ATTEMPTED"
    return "ACTIVE"


def build_flow_dataset(manifest_path: str | Path) -> pd.DataFrame:
    manifest = load_label_manifest(manifest_path)
    frames = []
    for row in manifest.itertuples(index=False):
        frames.append(
            pcap_to_flows(
                row.pcap_path,
                int(row.label),
                str(getattr(row, "attack_cat", "unknown")),
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
