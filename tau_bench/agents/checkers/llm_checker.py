import json
import os
from tau_bench.agents.checkers.i_checker import IToolCallChecker
from tau_bench.common import CostAccumulator
from tau_bench.types import Action, Message
from typing import List, Optional, Dict, Any, Tuple, Type
from litellm import completion
from tau_bench.envs.tool import ToolsMap, Tool
from loguru import logger

def get_tool_desc(tool_info:Dict):
    return tool_info.get("function").get("description")

class LLMToolCallChecker(IToolCallChecker):
    policy: str
    tools_map: ToolsMap
    llm_args: dict
    
    def __init__(self, policy:str, tools_map: ToolsMap, llm_args: dict) -> None:
        super().__init__()
        self.policy = policy
        self.tools_map = tools_map
        self.llm_args = llm_args

    def check_tool_call(self, action:Action, messages: List[Message], data: dict, cost_acc: CostAccumulator)->Tuple[bool, str]:
        policy = self.policy
        tool_name = action.name
        tool_cls = self.tools_map.get(tool_name)
        tool_info = tool_cls.get_info()
        tool_desc = get_tool_desc(tool_info)
        tool_args = action.kwargs
        
        sys_msg = f"""You are an assistant that validates tool calls in an AI system based on the provided policy.
Validate that the tool call in the last message adheres to the policy.
Return a JSON object with the following fields: 
\"valid\": boolean: true if the tool call is allowed, and 
\"reason\": string with the reason for your decision.

POLICY:
{policy}

HISTORIC MESSAGES:
{[json.dumps(m) for m in messages[1:]]}

TOOL CALL:
{tool_name} tool with the following description {tool_desc} and arguments {tool_args}.
# Is it according to the policy?
"""
        res = completion(
            messages=[{"role": "system", "content": sys_msg}],
            **self.llm_args,
            response_format={"type": "json_object"},
        )
        obj = json.loads(res.choices[0].message.content)
        cost_acc.add(res._hidden_params["response_cost"])
        valid, reason = obj.get("valid"), obj.get("reason")
        return (valid, reason)
