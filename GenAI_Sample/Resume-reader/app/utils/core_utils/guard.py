from typing import Tuple

BLOCK_PATTERNS = [
    "ignore previous instructions",
    "reveal system prompt",
    "show your hidden prompt",
    "developer message",
    "bypass safety",
]


def check_query_safety(query: str) -> Tuple[bool, str]:
    lowered = query.lower()
    for pattern in BLOCK_PATTERNS:
        if pattern in lowered:
            return False, f"Blocked due to suspicious pattern: '{pattern}'"
    return True, ""
