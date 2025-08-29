import json
import os
from tau_bench.agents.checkers.i_checker import IToolCallChecker
from tau_bench.common import CostAccumulator
from tau_bench.types import Action, Message
from typing import List, Optional, Dict, Any, Tuple, Type, Callable
from litellm import completion
from tau_bench.envs.tool import ToolsMap, Tool
from loguru import logger

BeforeToolCallArgs = {
    "params": Dict[str, Any], 
    "history": List[Message],
    "data": Dict[str, Any]
}
BeforeToolCallValidator = Callable[list(BeforeToolCallArgs.values()), Tuple[bool, str]]

class DeterministicToolCallChecker(IToolCallChecker):
    tool_validators = dict[str, BeforeToolCallValidator]
    def __init__(self, validators: dict[str, BeforeToolCallValidator]) -> None:
        super().__init__()
        self.tool_validators = validators
    
    def check_tool_call(self, action:Action, messages: List[Message], data: dict, cost_acc: CostAccumulator)->Tuple[bool, str]:
        tool_name = action.name
        validator = self.tool_validators.get(tool_name)
        if validator:
            return validator(action.kwargs, messages, data)
        return True, ""