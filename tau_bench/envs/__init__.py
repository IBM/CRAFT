# Copyright Sierra

from typing import Optional, Union
from tau_bench.envs.base import Env
from tau_bench.envs.user import UserStrategy


def get_env(
    env_name: str,
    no_policy: bool,
    user_strategy: Union[str, UserStrategy],
    user_model: str,
    task_split: str,
    user_model_atck: Optional[str] = None,
    user_policy_knowledge: Optional[str] = "",
    user_provider: Optional[str] = None,
    task_index: Optional[int] = None,
    ckpt_path: Optional[str] = None
) -> Env:
    if env_name == "retail":
        from tau_bench.envs.retail import MockRetailDomainEnv

        return MockRetailDomainEnv(
            user_strategy=user_strategy,
            user_model=user_model,
            no_policy=no_policy,
            user_model_atck=user_model_atck,
            task_split=task_split,
            user_provider=user_provider,
            task_index=task_index,
            ckpt_path=ckpt_path,
            domain=env_name
        )
    elif env_name == "airline":
        from tau_bench.envs.airline import MockAirlineDomainEnv

        return MockAirlineDomainEnv(
            user_strategy=user_strategy,
            user_model=user_model,
            no_policy=no_policy,
            user_model_atck=user_model_atck,
            user_policy_knowledge=user_policy_knowledge,
            task_split=task_split,
            user_provider=user_provider,
            task_index=task_index,
            ckpt_path=ckpt_path,
            domain=env_name
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")
