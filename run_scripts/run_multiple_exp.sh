AGENT_TO_RUN=(
#    "deepseek-ai/DeepSeek-V2.5"   
    "gpt-4o-mini-2024-07-18"
)

USER_TO_RUN=(
     "gpt-4o-2024-08-06"
)

ENVS=(
    "airline"
)

USER_ATTACK_STRATEGIES=(
    "CRAFT"
)

for env in "${ENVS[@]}"; do
    for agent_model in "${AGENT_TO_RUN[@]}"; do
        for user_model in "${USER_TO_RUN[@]}"; do
            if [[ "$agent_model" == *"gpt"* ]]; then
                agent_strategy="tool-calling"
                model_provider="azure"
            else
                agent_strategy="react"
                model_provider="openai"
            fi
            if [[ "$user_model" == *"gpt"* ]]; then
                user_model_provider="azure"
            else
                user_model_provider="openai"
            fi

            for attack_strategy in "${USER_ATTACK_STRATEGIES[@]}"; do
                (
                python run.py \
                    --agent-strategy "$agent_strategy" \
                    --env "$env" \
                    --model "$agent_model" \
                    --model-provider "$model_provider" \
                    --user-model  "$user_model"  \
                    --user-model-provider "$user_model_provider" \
                    --user-strategy "llm" \
                    --user-attack-strategy "$attack_strategy" \
                    --user_policy_knowledge "" \
                    --max-concurrency "4" \
                    --num-trials "4" \
                    --log-dir "./results_base" \

                echo "Completed model: $model | strategy: $attack_strategy | agent-strategy: $agent_strategy | provider: $model_provider"
                echo "---------------------------------------------------"
                )
            done
        done
    done
done