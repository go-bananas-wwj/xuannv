from __future__ import annotations

import pytest
import torch


@pytest.fixture
def torch_threads():
    """Shared fixture: temporarily raise PyTorch CPU thread count and restore it after the test."""
    prev = torch.get_num_threads()
    torch.set_num_threads(64)
    yield
    torch.set_num_threads(prev)
