# Effective Red-Teaming of Policy-Adherent Agents  

**Paper**: [https://arxiv.org/abs/2506.09600](https://arxiv.org/abs/2506.09600)  

This code is based on the paper ["Effective Red-Teaming of Policy-Adherent Agents"](https://github.com/sierra-research/tau-bench/tree/main).  

This codebase extends **τ-bench** and introduces two main components:  

- **τ-break**: a benchmark for evaluating the security of policy-adherent agents.  
  It includes modified ground-truth labels focused on security and new retail-domain policies designed to create policy violations.  

- **CRAFT**: a multi-agent, strategic red-teaming framework.  
  It simulates adversarial tactics, plans deceptive strategies, and probes edge cases to test the resilience of policy-adherent agents.  

## Setup

1. Clone this repository:

```bash
git clone git@github.com:IBM/CRAFT.git && cd ./CRAFT
```

2. Install from source (which also installs required packages):

```bash
pip install -e .
```

3. Set up your OpenAI / Anthropic / Google / Mistral / AnyScale API keys as environment variables.

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
```

## Run tau-break test cases

When calling a run, you should spesify information about your agent implementation strategy ("agent-strategy"), envirement (=domain, using the flag "env"), model (agent model), user-strategy (=wheather the user is LLM or Human), and additioanl configs that can be found under the "run.py" file. 

```bash
python run.py --agent-strategy tool-calling --env airline --model gpt-4o-2024-08-06 --model-provider azure --user-model gpt-4o --user-model-provider azure --user-strategy llm --max-concurrency 1
```

Set max concurrency according to your API limit(s).

To run specific tasks, use the `--task-ids` flag. Notice that every run will always limit your task-ids to be only within the sub-set tasks that are a part of Tau-break.

 For example:

```bash
python run.py --agent-strategy tool-calling --env airline --model gpt-4o-2024-08-06 --model-provider azure --user-model gpt-4o-2024-08-06 --user-model-provider azure --user-strategy llm --max-concurrency 1 --task-ids 10 15
```

This command will run only the tasks with IDs 10 and 15.

## User simulators and attacks

The attacks in this paper are based on changing the Engine behind the user simulator.
When not spesifing attack, you will have a 'non-strategic' user simulator, which will act overall cooperative, without any spesific adverserial behavior.
You can run one of the reported attacks by adding the '--user-attack-strategy' flag. The possible strategies are spesified in the tau_bench/envs/attacks.py.
For example, you can run:

```bash
python run.py --agent-strategy tool-calling --env airline --model gpt-4o-2024-08-06 --model-provider azure --user-model gpt-4o-2024-08-06 --user-model-provider azure --user-strategy llm --max-concurrency 1 --task-ids 10 15 --user-attack-strategy CRAFT
```

## Contact

Please contant one of the authors from the paper, or submit issues or pull requests if you find problems with the benchmark.

## Citation

```bibtex
@article{nakash2025effective,
  title={Effective Red-Teaming of Policy-Adherent Agents},
  author={Nakash, Itay and Kour, George and Lazar, Koren and Vetzler, Matan and Uziel, Guy and Anaby-Tavor, Ateret},
  journal={arXiv preprint arXiv:2506.09600},
  year={2025}
}
```
