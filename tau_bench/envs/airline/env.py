# Copyright Sierra

from tau_bench.envs.airline.data import load_data
from tau_bench.envs.airline.rules import RULES
from tau_bench.envs.airline.tools import ALL_TOOLS
from tau_bench.envs.airline.wiki import WIKI
from tau_bench.envs.wiki_no_policy import WIKI_NO_POLICY 
from tau_bench.envs.base import Env
from typing import Optional, Union
from tau_bench.envs.user import UserStrategy

class MockAirlineDomainEnv(Env):
    def __init__(
        self,
        user_strategy: Union[str, UserStrategy] = UserStrategy.LLM,
        user_model: str = "gpt-4o",
        no_policy: bool = False,
        user_provider: Optional[str] = None,
        task_split: str = "test",
        task_index: Optional[int] = None,
        user_model_atck: Optional[str] = None,
        user_policy_knowledge: Optional[str] = "",
        ckpt_path: Optional[str] = None,
        domain: Optional[str] = "airline"
    ):
        match task_split:
            case "test":
                from tau_bench.envs.airline.tasks_test import TASKS as tasks
            case _:
                raise ValueError(f"Unknown task split: {task_split}")
        super().__init__(
            data_load_func=load_data,
            tools=ALL_TOOLS,
            tasks=tasks,
            wiki=WIKI if not no_policy else WIKI_NO_POLICY,
            rules=RULES,
            user_strategy=user_strategy,
            user_model=user_model,
            user_provider=user_provider,
            task_index=task_index,
            user_model_atck=user_model_atck,
            user_policy_knowledge=user_policy_knowledge,
            ckpt_path=ckpt_path,
            domain=domain
        )
        self.terminate_tools = ["transfer_to_human_agents"]
