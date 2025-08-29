# Copyright Sierra

import os
from litellm import completion
from typing import List, Optional, Dict, Any
import re, json
from tau_bench.types import hirarchy_prompt
from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME, Message
from tau_bench.common import log_message
from llm_inference import is_rits_model, model_to_endpoint

with open(os.path.join("./tau_bench/attacks_cache/airline/airline_cached_strategies/airline_relevant_policies_per_taskid.json"), "r") as f:
    airline_relevant_policies_per_taskid = json.load(f)


class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        use_hirarchy_prompt: bool = False,
        remind_policy: str = "",
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.is_rits=is_rits_model(self.model)
        if self.is_rits:
            self.model = self.model[len("openai/"):] if self.model.startswith("openai/") else self.model
        self.use_hirarchy_prompt = use_hirarchy_prompt
        self.remind_policy = remind_policy
    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        sys_msg = {"role": "system", "content": self.wiki} if not self.use_hirarchy_prompt else {"role": "system", "content":  hirarchy_prompt['additional_prompt'] + hirarchy_prompt['start_of_policy'] + self.wiki + hirarchy_prompt['end_of_policy']}
        human_msg = {"role": "user", "content": obs}
        messages: List[Message] = [sys_msg, human_msg]
        log_message(human_msg)
        for _ in range(max_num_steps):
            injected_messages = messages.copy()
            if self.remind_policy== "full" and len(messages) > 2:
                injected_messages.append({
                    "role": "system",
                    "content": "Reminder: The following policy must be strictly followed:\n" + self.wiki
                })
            elif self.remind_policy == "relevant" and len(messages) > 2:
                relevant_policy=airline_relevant_policies_per_taskid[str(task_index)]
                injected_messages.append({
                    "role": "system",
                    "content": "Reminder: The following policy must be strictly followed:\n" + relevant_policy
                })
            res = completion(
                messages=injected_messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
            )
            next_message = res.choices[0].message.model_dump()
            log_message(next_message)
       #     total_cost += res._hidden_params["response_cost"]
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
