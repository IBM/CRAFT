import json

def get_all_tool_calls(task_traj: str):
    """"User this function on a policy-free trajectory (or policy-adherence trajectory) to get the tool calls that were used in it. Then you
    can compare them and create the new ground truth as the difference, as in Section 3 at the paper https://arxiv.org/abs/2506.09600."""
    tool_calls=[]
    skip_tool_calls=['search_direct_flight','get_user_details','get_reservation_details','calculate','think','search_onestop_flight'] #getters tool calls that we don't care if the agent did
    for step in task_traj:
        if step.get("role") == "assistant" and step.get("tool_calls"):
            for call in step["tool_calls"]:
                if call["function"]["name"] not in skip_tool_calls:
                    tool_calls.append({
                        "function": call["function"]["name"],
                        "kwargs": json.loads(call["function"]["arguments"]),
                        "tool_call_id": call["id"]
                    })
    return tool_calls

def add_tool_calls_to_reward_info(json_file, output_file):
    """" Add to trajectory the tool call that happened in it - for gold labeling"""
    with open(json_file, "r") as f:
        data = json.load(f)
    for task in data:
        tool_calls = get_all_tool_calls(task["traj"])        
        if "info" not in task or not isinstance(task["info"], dict):
            task["info"] = {}
        if "reward_info" not in task["info"] or not isinstance(task["info"].get("reward_info"), dict):
            task["info"]["reward_info"] = {}
        
        task["info"]["reward_info"]["tool_calls"] = tool_calls

        task["info"]["tool_diff"]=[x for x in task["info"]["reward_info"]["tool_calls"]  if x not in task["info"]["task"]["actions"]]


    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)