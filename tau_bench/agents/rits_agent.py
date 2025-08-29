# Copyright Sierra

import os
from litellm import completion
from typing import List, Optional, Dict, Any
import re, json

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME, Message
from tau_bench.common import log_message
from llm_inference import is_rits_model, model_to_endpoint
class Rits_Agent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        assert is_rits_model(self.model)
        self.model = self.model[len("openai/"):] if self.model.startswith("openai/") else self.model

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        sys_msg = {"role": "system", "content": self.wiki}
        human_msg = {"role": "user", "content": obs}
        messages: List[Message] = [ sys_msg, human_msg]
        log_message(human_msg)
        for _ in range(max_num_steps):
            cleaned_msg = [
                    {k: v for k, v in msg.items() if k not in {'tool_calls', 'function_call'}}
                    for msg in messages
                    if msg.get("role") != "tool"
                ]
            cleaned_msg[0]["content"] += """
            When you need to use a tool, respond only with the following JSON format, wrapped between <START_TOOL_CALL> and <END_TOOL_CALL> tags, and nothing else:

            <START_TOOL_CALL>
            {
            "tool_name": "<name of the tool>",
            "arguments": {
                "arg1": "value1",
                "arg2": "value2"
            }
            }
            <END_TOOL_CALL>

            Only use tools that are explicitly available to you. Do not invent or guess tool names.

            Use a tool only if it is necessary to make progress. If no tool is needed, respond in natural language with your reasoning.
            """

            res = completion(
                messages=cleaned_msg,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
                base_url=model_to_endpoint[self.model],
                extra_headers={
                'RITS_API_KEY': os.getenv("RITS_API_KEY")
                },
            )

            next_message = res.choices[0].message.model_dump()
            log_message(next_message)
            
            tool_call = parse_tool_call_from_text(next_message)
            if tool_call:
                next_message["tool_calls"] = [tool_call]

            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                tool_reply = {
                    "role": "tool",
                    "tool_call_id": next_message["tool_calls"][0]["id"],
                    "name": next_message["tool_calls"][0]["function"]["name"],
                    "content": env_response.observation,
                }
                log_message(tool_reply)
                messages.extend([next_message, tool_reply])
            else:
                user_reply = {"role": "user", "content": env_response.observation}
                log_message(user_reply)
                messages.extend([next_message, user_reply])
            if env_response.done:
                break
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )



def parse_tool_call_from_text(message):
    content = message.get("content", "")
    pattern = r"<START_TOOL_CALL>(.*?)<END_TOOL_CALL>"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None
    try:
        tool_json = json.loads(match.group(1).strip())
        return {
            "id": "call_1",  # dummy ID; update if needed
            "type": "function",
            "function": {
                "name": tool_json["tool_name"],
                "arguments": json.dumps(tool_json["arguments"])
            }
        }
    except Exception as e:
        print("Tool call parsing failed:", e)
        return None

def add_tool_calls_to_message(message: Dict[str, Any]) -> None:
    content = message.get("content", "")
    match = re.search(r"Action:\s*(\w+)\((.*?)\)", content)
    if match:
        tool_name = match.group(1)
        args_str = match.group(2)

        # Convert args_str to dict safely
        try:
            args = dict(re.findall(r"(\w+)\s*=\s*([^,]+)", args_str))
        except Exception:
            args = {}

        message["tool_calls"] = [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(args)
            }
        }]

def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})