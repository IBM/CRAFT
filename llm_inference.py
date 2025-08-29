from abc import ABC, abstractmethod
from dotenv import dotenv_values
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

rits_model_name_to_endpoint_list=[{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/balma","model_name":"balma"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/bamba","model_name":"bamba"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/byom-gb-pr-3735-fina","model_name":"ibm-granite/granite-3.3-8b-instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/codellama-34b-instruct-hf","model_name":"codellama/CodeLlama-34b-Instruct-hf"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/deepseek-coder-33b-instruct","model_name":"deepseek-ai/deepseek-coder-33b-instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/deepseek-v2-5/v1/","model_name":"deepseek-ai/DeepSeek-V2.5"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/deepseek-v3/v1","model_name":"deepseek-ai/DeepSeek-V3"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/devstral-small-2505","model_name":"mistralai/Devstral-Small-2505"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-13b-chat-v2","model_name":"ibm-granite/granite-13b-chat-v2"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-13b-instruct-v2","model_name":"ibm-granite/granite-13b-instruct-v2"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-20b-ci-r1-1","model_name":"ibm-granite/granite-20b-code-instruct-r1.1"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-20b-code-8k-ansible","model_name":"ibm/granite-20b-code-8k-ansible"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-20b-code-instruct","model_name":"ibm/granite-20b-code-instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-20b-code-instruct-8k","model_name":"ibm-granite/granite-20b-code-instruct-8k"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-20b-code-instruct-uapi","model_name":"ibm-granite/granite-20b-code-instruct-unified-api"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-20b-functioncalling","model_name":"ibm-granite/granite-20b-functioncalling"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-0-8b-instruct/v1","model_name":"ibm-granite/granite-3.0-8b-instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-2b-instruct","model_name":"ibm-granite/granite-3.1-2b-instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-content-linking","model_name":"ibm/granite-3-1-8b-content-linking"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-gen-sl","model_name":"ibm/granite-3-1-8b-gen-sl"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-instruct","model_name":"ibm-granite/granite-3.1-8b-instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-question-gen","model_name":"ibm/granite-3-1-8b-question-gen"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-schema-linking","model_name":"ibm/granite-3-1-8b-schema-linking"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-sql-gen","model_name":"ibm/granite-3-1-8b-sql-gen"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-2-8b-instruct","model_name":"ibm-granite/granite-3.2-8b-instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-2-8b-instruct-chat","model_name":"ibm-granite/granite-3.2-8b-instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-2-8b-lora-answr-pred","model_name":"ibm-granite/granite-3.2-8b-lora-rag-answerability-prediction"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-2-8b-lora-cit-gen","model_name":"ibm-granite/granite-3.2-8b-lora-rag-citation-generation"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-2-8b-lora-hall-det","model_name":"ibm-granite/granite-3.2-8b-lora-rag-hallucination-detection"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-2-8b-lora-query-rw","model_name":"ibm-granite/granite-3.2-8b-lora-rag-query-rewrite"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-2-8b-lora-uncertainty","model_name":"ibm-granite/granite-3.2-8b-lora-uncertainty"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-3-8b-base","model_name":"ibm-granite/granite-3.3-8b-base"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-3-8b-instruct","model_name":"ibm-granite/granite-3.3-8b-instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-3-8b-lora-answr-pred","model_name":"ibm-granite/granite-3.3-8b-lora-rag-answerability-prediction"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-3-8b-lora-certainty","model_name":"ibm-granite/granite-3.3-8b-lora-certainty"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-3-8b-lora-cit-gen","model_name":"ibm-granite/granite-3.3-8b-lora-rag-citation-generation"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-3-8b-lora-context","model_name":"ibm-granite/granite-3.3-8b-lora-rag-context-relevancy"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-3-8b-lora-hall-det","model_name":"ibm-granite/granite-3.3-8b-lora-rag-hallucination-detection"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-3-8b-lora-query-rw","model_name":"ibm-granite/granite-3.3-8b-lora-rag-query-rewrite"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-34b-code-instruct-8k","model_name":"ibm-granite/granite-34b-code-instruct-8k"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-34b-content-linking","model_name":"ibm/granite-34b-content-linking"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-34b-question-gen","model_name":"ibm/granite-34b-question-gen"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-34b-schema-linking","model_name":"ibm/granite-34b-schema-linking"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-34b-sql-gen","model_name":"ibm/granite-34b-sql-gen"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-4-0-medium-franconia","model_name":"ibm-granite/granite-4.0-medium-prerelease-franconia.r250523a"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-4-0-small-franconia","model_name":"ibm-granite/granite-4.0-small-prerelease-franconia.r250523a"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-7b-lab","model_name":"ibm/granite-7b-lab"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-8b-code-instruct-128k","model_name":"ibm-granite/granite-8b-code-instruct-128k"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-8b-code-instruct-4k","model_name":"ibm-granite/granite-8b-code-instruct-4k"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-8b-instruct-preview-4k","model_name":"ibm-granite/granite-8b-instruct-preview-4k"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-8b-japanese-instruct","model_name":"ibm-granite/granite-8b-japanese-instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-guardian-3-2-3b-a800m","model_name":"ibm-granite/granite-guardian-3.2-3b-a800m"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-guardian-3-2-5b-ris","model_name":"ibm-granite/granite-guardian-3.2-5b"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-speech-3-3-8b","model_name":"ibm-granite/granite-speech-3.3-8b"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-vision-3-2-2b","model_name":"ibm-granite/granite-vision-3.2-2b"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-vision-3-3-2b","model_name":"ibm-granite/granite-vision-3.3-2b"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/internvl2-llama3-76b","model_name":"OpenGVLab/InternVL2-Llama3-76B"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-1-405b-instruct-fp8","model_name":"meta-llama/llama-3-1-405b-instruct-fp8"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-1-8b-instruct/v1","model_name":"meta-llama/Llama-3.1-8B-Instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-1-test","model_name":"meta-llama/llama-3-1-405b-instruct-fp8"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-2-11b-instruct","model_name":"meta-llama/Llama-3.2-11B-Vision-Instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-2-11b-ravi","model_name":"meta-llama/Llama-3.2-11B-Vision-Instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-2-90b-instruct","model_name":"meta-llama/Llama-3.2-90B-Vision-Instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1","model_name":"meta-llama/llama-3-3-70b-instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct-e","model_name":"meta-llama/llama-3-3-70b-instruct-embeddings"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-4-mvk-17b-128e-fp8","model_name":"meta-llama/llama-4-maverick-17b-128e-instruct-fp8"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-4-scout-17b-16e-instruct","model_name":"meta-llama/Llama-4-Scout-17B-16E-Instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/microsoft-phi-4","model_name":"microsoft/phi-4"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/microsoft-phi-4-test","model_name":"microsoft/phi-4"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mistral-small-3-1-24b-2503","model_name":"mistralai/Mistral-Small-3.1-24B-Instruct-2503"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mistralai-pixtral-12b-2409","model_name":"mistralai/pixtral-12b-2409"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x22b-instruct-v01","model_name":"mistralai/mixtral-8x22B-instruct-v0.1"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x7b-instruct-v01","model_name":"mistralai/mixtral-8x7B-instruct-v0.1"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x7b-instruct-v01-test","model_name":"mistralai/mixtral-8x7B-instruct-v0.1-test"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/phi-4-reasoning","model_name":"microsoft/Phi-4-reasoning"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/qwen2-5-72b-instruct/v1","model_name":"Qwen/Qwen2.5-72B-Instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/qwen2-vl-72b-instruct","model_name":"Qwen/Qwen2-VL-72B-Instruct"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/slate-125m-english-rtrvr-v2","model_name":"ibm/slate-125m-english-rtrvr-v2"},{"endpoint":"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/test-granite-speech-3-3-8b","model_name":"ibm-granite/granite-speech-3.3-8b"}]

model_to_endpoint = {
    entry["model_name"]: entry["endpoint"]
    for entry in rits_model_name_to_endpoint_list
}

def is_rits_model(model: str):
    if 'openai/' == model[:len('openai/')]:
        model = model[len('openai/'):]
    if model in model_to_endpoint:
    #    print(f"-----$$$$$$$$$$$$$$$$$$--------------              Model {model} is a RITS model              ----------------$$$$$$$$$$$$$$$$$$-----")
        return True
    
    else:
    #    print(f"-----$$$$$$$$$$$$$$$$$$--------------              Model {model} is an OPENAI model              ----------------$$$$$$$$$$$$$$$$$$-----")
        return False

class LLMInference(ABC):
    @abstractmethod
    def completion(self, prompt: str) -> str:
        pass

    @abstractmethod
    def get_model_name(self):
        pass

class WatsonLLMInference(LLMInference):
    def __init__(self, model_type="watsonx/ibm/granite-13b-chat-v2", temperature=1, max_tokens=10, env_path=".env"):
        # Load configuration from environment variables
        config = dotenv_values(env_path)

        self.url = config['WML_URL']
        self.api_key = config['WML_APIKEY']
        self.project_id = config['WML_PROJECT_ID']

        self.model_type = model_type
        self.temperature = temperature
        # self.max_tokens = max_tokens

    def generate_response(self, prompt: str) -> str:
        response = completion(
            model=self.model_type,
            temperature=self.temperature,
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            # max_tokens=self.max_tokens,
            url=self.url,
            api_key=self.api_key,
            project_id=self.project_id,
            # decoding_method=DecodingMethods.GREEDY
        )
        return response.choices[0].message.content

    def get_model_name(self):
        return self.model_type

class OpenAILLMInference(LLMInference):
    def __init__(self, model="gpt-4o-2024-08-06", temperature=1, env_path=".env"):
        # Load API key from environment variables
        config = dotenv_values(env_path)
        self.api_key = config['OPENAI_APIKEY']
        self.model_type = model
        self.temperature = temperature

    def completion(self, prompt: str, model_type: str = "gpt-4") -> str:
        """
        Generate a response using OpenAI's models.
        :param prompt: The input prompt for the model.
        :param model_type: The OpenAI model to use (e.g., "gpt-4").
        :return: The generated response as a string.
        """

        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}]
        return completion(
            messages=messages,
            model=self.model_type,
            custom_llm_provider="azure",
            temperature=self.temperature
        )

    def get_model_name(self):
        return self.model_type


class RITSInference(LLMInference):
    def __init__(self, url, model, max_tokens=100, env_path=".env", ):
        # Load API key from environment variables
        config = dotenv_values(env_path)
        self.api_key = config['RITS_API_KEY']
        self.model_type = model
        self.url = url
        # self.max_tokens = max_tokens

    def completion(self, prompt: str,temperature:int,seed:int=10) -> str:
        """
        Generate a response using OpenAI's models.
        :param prompt: The input prompt for the model.
        :param model_type: The OpenAI model to use (e.g., "gpt-4").
        :return: The generated response as a string.
        """

        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}]
        return completion(
            messages=messages,
            model=self.model_type,
            custom_llm_provider="openai",
            base_url=self.url,
            extra_headers={
                'RITS_API_KEY': self.api_key
            },
            # max_tokens=self.max_tokens,
            temperature=temperature,
            seed=seed
        )

    def get_model_name(self):
        return self.model_type

# main:
if __name__ == "__main__":
    model='meta-llama/llama-3-3-70b-instruct'
    url='https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1'
    model = "deepseek-ai/DeepSeek-V3"
    url='https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/deepseek-v3/v1'

    model ="ibm-granite/granite-3.0-8b-instruct"
    url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-0-8b-instruct/v1"


    config = dotenv_values(".env")
    api_key = config['RITS_API_KEY']
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}]
    response= completion(
                messages=messages, 
                model=model,
                custom_llm_provider="openai",
                base_url=url,
                extra_headers={
                    'RITS_API_KEY': api_key
                },
            )
    print(response.choices[0].message.content)