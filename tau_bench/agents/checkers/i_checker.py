import abc
from tau_bench.common import CostAccumulator
from tau_bench.types import Action, Message
from typing import List, Tuple


class IToolCallChecker(abc.ABC):

    @abc.abstractmethod
    def check_tool_call(self, action:Action, messages: List[Message], data: dict, cost_acc: CostAccumulator)->Tuple[bool, str]:
        ...
        