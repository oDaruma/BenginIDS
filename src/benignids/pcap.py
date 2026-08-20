from __future__ import annotations

from collections import defaultdict
from pathlib import Path

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
    pcap_path: str | Path, label: int | None = None, attack_cat: str = "unknown"
) -> pd.DataFrame:
    """Aggregate a capture into bidirectional five-tuple flow records.

    Labels are required when building training data and omitted for inference captures.
    """
    try:
        from scapy.all import IP, TCP, UDP, IPv6, PcapReader
    except ImportError as exc:
        raise RuntimeError("Install PCAP support with: pip install -e '.[pcap]'") from exc

    flows = defaultdict(lambda: {"packets": 0, "bytes": 0, "timestamps": [], "payload": bytearray()})
    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            network = packet.getlayer(IP) or packet.getlayer(IPv6)
            if network is None:
                continue
            transport = packet.getlayer(TCP) or packet.getlayer(UDP)
            protocol = "tcp" if packet.haslayer(TCP) else "udp" if packet.haslayer(UDP) else str(network.nh)
            source_port = int(getattr(transport, "sport", 0))
            destination_port = int(getattr(transport, "dport", 0))
            endpoint_a = (str(network.src), source_port)
            endpoint_b = (str(network.dst), destination_port)
            left, right = sorted((endpoint_a, endpoint_b))
            key = (left, right, protocol)
            flow = flows[key]
            flow["packets"] += 1
            flow["bytes"] += len(packet)
            flow["timestamps"].append(float(packet.time))
            payload = bytes(getattr(transport, "payload", b"")) if transport else b""
            if len(flow["payload"]) < 64:
                flow["payload"].extend(payload[: 64 - len(flow["payload"])])

    rows = []
    for (left, right, protocol), flow in flows.items():
        timestamps = flow["timestamps"]
        row = {
                "endpoint_a": left[0],
                "port_a": left[1],
                "endpoint_b": right[0],
                "port_b": right[1],
                "protocol": protocol,
                "packets": flow["packets"],
                "total_len": flow["bytes"],
                "duration": max(timestamps) - min(timestamps),
                "payload": bytes(flow["payload"]).hex(),
                "capture_id": Path(pcap_path).stem,
            }
        if label is not None:
            row.update({"label": int(label), "attack_cat": attack_cat})
        rows.append(row)
    return pd.DataFrame(rows)


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
