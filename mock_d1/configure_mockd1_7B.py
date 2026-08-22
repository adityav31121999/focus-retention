# configure_mockd17B.py
# Backward-compatibility forwarding module
from .configure_mockd1_mini import MockD1Config, MockD1Config7B, MockD1Config1_7B

__all__ = ["MockD1Config", "MockD1Config7B", "MockD1Config1_7B"]
