
from litellm import completion
from tau_bench.model_utils.model.openai import OpenAIModel

class Empty:
    pass
class LiteLLMModel(OpenAIModel):
    def __init__(self, provider:str, model:str, api_key:str, temperature) -> None:
        super().__init__(model, api_key, temperature)
        self.client = Empty()
        self.client.chat = Empty()
        self.client.chat.completions = Empty()
        def create_fn(**kwargs):
            return completion(
                **kwargs,
                custom_llm_provider = provider
            )
        self.client.chat.completions.create = create_fn

