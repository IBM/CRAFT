# Copyright Sierra

import argparse
import sys
from tau_bench.types import RunConfig
from tau_bench.run import run
from litellm import provider_list
from tau_bench.envs.user import UserStrategy
from dotenv import load_dotenv
from tau_bench.envs.attacks import attack_descriptions
load_dotenv()

from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}|{thread.id}|</green> <level>{message}</level>")

def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="airline"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the agent",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o",
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="tool-calling",
        choices=["tool-calling", "act", "react", "few-shot"],
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The sampling temperature for the action model",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="The split of tasks to run (only applies to the retail domain for now",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument("--task-ids", type=int, nargs="+", help="(Optional) run only the tasks with the given IDs")
    parser.add_argument("--log-dir", type=str, default="results/")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])
    parser.add_argument("--few-shot-displays-path", type=str, help="Path to a jsonlines file containing few shot displays")
    parser.add_argument("--no_policy", action="store_true",default=False ,help="Use empty policy, which just say to follow every user request")

    # NEW ADDED ARGUMENTS FOR ATTACKS:

    parser.add_argument(
        "--user-attack-strategy",
        type=str,
        default="None",
        choices = list(attack_descriptions.keys())+["None"],
        help="The attack strategy to use for the user simulator",
        )
    parser.add_argument(
        "--user_policy_knowledge",
        type=str,
        default="",
        help="The policy knowledge to use for the user simulator",
        choices=["", "airline", "retail"],
    )
    parser.add_argument(
        "--use_hirarchy_prompt",
        action="store_true",
        default=False,
        help="Use hirarchy prompt for the agent",
    )
    parser.add_argument(
        "--remind_policy",
        type=str,
        default="",
        help="The remind policy for the agent",
        choices=["", "full", "relevant"],
    )

    args = parser.parse_args()
    print(args)
    return RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=args.env,
        no_policy=args.no_policy,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
        user_attack_strategy=args.user_attack_strategy,
        user_policy_knowledge=args.user_policy_knowledge,
        use_hirarchy_prompt=args.use_hirarchy_prompt,
        remind_policy=args.remind_policy
    )

def main():
    config = parse_args()
    run(config)
#example run:
# python run.py --agent-strategy tool-calling --env airline --model gpt-4o-2024-08-06 --model-provider azure --user-model gpt-4o-2024-08-06 --user-model-provider azure --user-strategy llm --max-concurrency 1 --no_policy  --user-attack-strategy CRAFT --task-ids 49

if __name__ == "__main__":
    main()