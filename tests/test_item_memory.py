import pytest
import torch
import torchhd
from hdlnn.hdc.item_memory import ItemMemory

def test_item_memory_dynamic_registration_and_oov_locking():
    D = 10000
    memory = ItemMemory(D=D)

    # 1. Unlocked mode (training)
    memory.unlock()
    
    vec_tcp, is_oov_tcp = memory.get_vector("proto", "tcp")
    assert is_oov_tcp is False
    assert vec_tcp.shape == (D,)
    
    # Retrieve again, should return identical vector
    vec_tcp_2, is_oov_tcp_2 = memory.get_vector("proto", "tcp")
    assert is_oov_tcp_2 is False
    assert torch.equal(vec_tcp, vec_tcp_2)

    vec_udp, is_oov_udp = memory.get_vector("proto", "udp")
    assert is_oov_udp is False
    assert not torch.equal(vec_tcp, vec_udp)

    # 2. Locked mode (inference/eval)
    memory.lock()

    # Query seen token - should return identical vector and not be OOV
    vec_tcp_locked, is_oov_tcp_locked = memory.get_vector("proto", "tcp")
    assert is_oov_tcp_locked is False
    assert torch.equal(vec_tcp, vec_tcp_locked)

    # Query unseen token - should return OOV vector and is_oov=True
    vec_icmp, is_oov_icmp = memory.get_vector("proto", "icmp")
    assert is_oov_icmp is True
    
    # Query another unseen token in the same column - should return the SAME OOV vector
    vec_ospf, is_oov_ospf = memory.get_vector("proto", "ospf")
    assert is_oov_ospf is True
    assert torch.equal(vec_icmp, vec_ospf)

    # Different column OOV should have its own OOV vector
    vec_http_oov, is_oov_http = memory.get_vector("service", "http")
    assert is_oov_http is True
    assert not torch.equal(vec_icmp, vec_http_oov)
