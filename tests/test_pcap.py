import pytest

from benignids.pcap import _session_state, pcap_to_flows


def test_session_state_uses_tcp_lifecycle_flags():
    assert _session_state("tcp", [0x02]) == "ATTEMPTED"
    assert _session_state("tcp", [0x02, 0x12, 0x10]) == "ESTABLISHED"
    assert _session_state("tcp", [0x10, 0x11]) == "CLOSED"
    assert _session_state("tcp", [0x10, 0x14]) == "RESET"
    assert _session_state("udp", [0]) == "ACTIVE"


def test_session_timeout_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        pcap_to_flows("unused.pcap", session_timeout=0)
