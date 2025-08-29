from tau_bench.types import Message
from loguru import logger


def in_yellow(s:str):
    return f"\033[93m{s}\033[0m"


def log_message(msg: Message, idx = None):
    tcs = [tc.get("function") for tc in msg.get("tool_calls") or []]
    char = "»" if tcs == [] and msg.get("role") == "assistant" else ""
    log_msg = f"{msg.get('role', '').upper()}: {char}{msg.get('content')}"
    for tc in tcs:
        log_msg+=f"\n\t{tc.get('name')}\t{tc.get('arguments')}"
    if idx is None:
        logger.debug(log_msg )
    else:
        logger.debug(f"[{idx}]{log_msg}" )



class CostAccumulator():
    _cost = 0
    def __init__(self) -> None:
        self._cost = 0
    
    def add(self, cost:float):
        self._cost += cost

    @property
    def total(self):
        return self._cost