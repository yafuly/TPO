

from typing import List, Dict, Any
import torch
import numpy as np

from transformers import AutoTokenizer, pipeline


class PipelineDataset(torch.utils.data.Dataset):
    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TPORewardModel():
    def __init__(self, reward_model: str = "sfairXC/FsfairX-LLaMA3-RM-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(reward_model)
        self.pipe = pipeline(
                "sentiment-analysis",
                model=reward_model,
                device_map="auto",
                tokenizer=self.tokenizer,
                model_kwargs={"torch_dtype": torch.bfloat16})

    def compute_reward_scores(self, messages: List, pipe_kwargs: Dict[str, Any]) -> List[float]:
        test_texts = [_.replace(self.tokenizer.bos_token, "") for _ in self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)]

        test_dataset = PipelineDataset(test_texts)
        rewards = [output[0]["score"] for output in self.pipe(test_dataset, **pipe_kwargs)]
        
        return rewards

    def perform_rm(self, qa_pairs):
        pipe_kwargs = {
            # "return_all_scores": True,
            "top_k": None,
            "function_to_apply": "none",
            "batch_size": 1, 
        }
        messages = []
        for prompt, generation in qa_pairs:
                curr_messages = [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': generation}
                ]
                messages.append(curr_messages)
        rewards = self.compute_reward_scores(messages, pipe_kwargs)

        return rewards

    def get_contrastive_samples(self, scores, qa_pairs):
        """
        Get contrastive samples based on the best and worst scores from a reward model.

        Args:
            qa_pairs (list of dict): List of question-answer pairs, each a dictionary containing 'question' and 'answer'.
            rm_pipe (Callable): Reward model pipeline that returns scores for QA pairs.
            rm_tokenizer (Callable): Tokenizer for preprocessing QA pairs before scoring.

        Returns:
            dict: A dictionary containing the best and worst QA pairs.
        """

        def truncate_text(text, tokenizer, max_length=2048):
            """
            Truncate a text to a maximum token length using a given tokenizer.
            """
            token_ids = tokenizer.encode(text, truncation=False)
            if len(token_ids) <= max_length:
                return text
            truncated_token_ids = token_ids[:max_length]
            truncated_text = tokenizer.decode(truncated_token_ids, skip_special_tokens=True)

            return truncated_text

        
        # Get the indices of the best and worst scores
        best_index = np.argmax(scores)
        worst_index = np.argmin(scores)
        delta = max(scores) - min(scores)
        # Extract the best and worst QA pairs
        best_answer = truncate_text(qa_pairs[best_index][1], self.tokenizer)
        worst_answer = truncate_text(qa_pairs[worst_index][1], self.tokenizer)
        return {
            "best": best_answer,
            "worst": worst_answer
        }, delta

