try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError(
        "If you'd like to use VLLM models, please install the vllm package by running `pip install vllm` or `pip install textgrad[vllm]."
    )

import os
import platformdirs
from .base import EngineLM, CachedEngine


class ChatVLLM(EngineLM, CachedEngine):
    # Default system prompt for VLLM models
    DEFAULT_SYSTEM_PROMPT = ""

    def __init__(
        self,
        model_string="meta-llama/Meta-Llama-3-8B-Instruct",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        **llm_config,
    ):
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_vllm_{model_string}.db")
        super().__init__(cache_path=cache_path)

        self.model_string = model_string
        self.system_prompt = system_prompt
        self.client = LLM(self.model_string, **llm_config)
        self.tokenizer = self.client.get_tokenizer()

    def generate(
        self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, top_p=0.95, n=1
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        # The chat template ignores the system prompt;
        # print("+"*100)
        # print("VLLM prompt ...")
        # print(prompt)
        conversation = []
        if sys_prompt_arg:
            conversation = [{"role": "system", "content": sys_prompt_arg}]

        conversation += [{"role": "user", "content": prompt}]
        chat_str = self.tokenizer.apply_chat_template(conversation, tokenize=False)

        sampling_params = SamplingParams(
            temperature=temperature, 
            max_tokens=max_tokens, 
            top_p=top_p, 
            n=n, 
            stop=['<|end_of_text|>', '<|eot_id|>'],
        )

        response = self.client.generate([chat_str], sampling_params)

        if n > 1:
            response = [out.text for out in response[0].outputs]
            return response
        else:
            response = response[0].outputs[0].text
            self._save_cache(sys_prompt_arg + prompt, response)
            return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
