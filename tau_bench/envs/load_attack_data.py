import os
import json
from tau_bench.envs.tasks_airline_to_load import tasks as tasks_airline
from tau_bench.envs.tasks_retail_to_load import tasks as tasks_retail


instruction_to_id_airlines = {task["instruction"]: task["task-id"] for task in tasks_airline}   
instruction_to_id_retail = {task["instruction"]: task["task-id"] for task in tasks_retail if 'task-id' in task}


with open(os.path.join("tau_bench/envs/airline/wiki.md"), "r") as f:
    WIKI_AIRLINE = f.read()

with open(os.path.join("tau_bench/envs/retail/wiki.md"), "r") as f:
    WIKI_RETAIL = f.read()



with open(os.path.join("tau_bench/attacks_cache/airline/airline_cached_strategies/airline_relevant_policies_per_taskid.json"), "r") as f:
    policyAnalyzer_airline = json.load(f)

with open(os.path.join("tau_bench/attacks_cache/airline/airline_cached_strategies/airline_strategies_per_taskid_DeceptionPlanner.json"), "r") as f:
    deceptionPlanner_airline = json.load(f)


with open(os.path.join("tau_bench/attacks_cache/airline/airline_cached_strategies/airline_strategies_per_taskid_AvoidanceAdvisor.json"), "r") as f:
    avoidanceAdvisor_airline = json.load(f)

with open(os.path.join("tau_bench/attacks_cache/retail/retail_strategies_per_taskid_DeceptionPlanner.json"), "r") as f:
    deceptionPlanner_retail = json.load(f)

with open(os.path.join("tau_bench/attacks_cache/retail/retail_strategies_per_taskid_AvoidanceAdvisor.json"), "r") as f:
    avoidanceAdvisor_retail = json.load(f)

with open(os.path.join("tau_bench/attacks_cache/retail/relevant_policies_per_taskid_retail.json"), "r") as f:
    policyAnalyzer_retail = json.load(f)

