import argparse
import json
import random
import time
import os
import concurrent.futures

from tqdm import tqdm

import textgrad as tg
from reward_model import TPORewardModel

# Import TPO methods from test_time_training.py
from tpo_utils import (
    run_test_time_training_bon,
    run_test_time_training_tpo
)




def set_random(seed: int = 7) -> None:
    """
    Sets the global random seed for reproducibility.
    """
    random.seed(seed)

VALID_TPO_MODES = ["tpo", "revision", "bon"]


def config() -> argparse.Namespace:
    """
    Parses command-line arguments and returns a namespace object
    containing all configuration parameters.
    """
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
    parser.add_argument("--data_path", type=str, help="Path to the data file (JSON).")
    parser.add_argument("--output_path", type=str, default="./results", help="Path to the save results.")
    parser.add_argument("--reward_model", type=str, default="sfairXC/FsfairX-LLaMA3-RM-v0.1",
                        help="Identifier or path for the reward model.")
    parser.add_argument("--server_model", type=str, default="meta-llama/Llama-3.1-70B-Instruct",
                        help="Base model used for serving via an API.")
    parser.add_argument("--ip", type=str, help="Server IP (i.e. 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port number for the model server.")
    parser.add_argument("--tpo_mode", type=str, default="tpo",
                        help="Mode for test-time preference optimization (tpo, revision, or bon).")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Number of threads to use for evaluation.")
    parser.add_argument("--batch_size", type=int, default=3,
                        help="Batch size for test-time optimization.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed for reproducibility.")
    parser.add_argument("--max_tokens_response", type=int, default=2048,
                        help="Max tokens to generate for each output sequence.")
    parser.add_argument("--max_tokens_all", type=int, default=8192,
                        help="Max tokens for the entire context during generation.")
    parser.add_argument("--max_iterations", type=int, default=5,
                        help="Max number of test-time optimization iterations.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for generation.")
    parser.add_argument("--sample_size", type=int, default=5,
                        help="Number of responses to sample for each step.")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = config()
    assert args.tpo_mode in VALID_TPO_MODES, f"Invalid TPO mode: {args.tpo_mode}"
    os.makedirs(args.output_path, exist_ok=True)
    
    # Set random seed
    set_random(args.seed)

    # Construct model name and engine
    model_name = f"server-{args.server_model}"
    llm_engine = tg.get_engine(
        model_name,
        base_url=f"http://{args.ip}:{args.port}/v1",
        api_key="token-abc123",
        max_token=args.max_tokens_all,
    )

    # Initialize reward model
    reward_model_name = args.reward_model
    rm = TPORewardModel(reward_model_name)

    # Load data
    data_path = args.data_path
    data_name = data_path.split("/")[-1]
    with open(data_path, "r", encoding="utf-8") as f:
        datas = json.load(f)
    random.shuffle(datas)

    # Prepare generation params
    diverse_gen_params = {
        "n": args.sample_size,
        "temperature": args.temperature,
        "top_p": 0.95,
        "seed": args.seed,
        "max_tokens": args.max_tokens_response,
    }

    # Decide which function to call based on tpo_mode
    run_test_time_training = (
        run_test_time_training_bon if args.tpo_mode == "bon" else run_test_time_training_tpo
    )

    # Prepare output path
    model_suffix = model_name.split("/")[-1]
    reward_suffix = reward_model_name.split("/")[-1]
    out_path = (
        f"{args.output_path}/{data_name}_model_{model_suffix}_mode_{args.tpo_mode}_"
        f"rm_{reward_suffix}_max_iters{args.max_iterations}_"
        f"sample_size{args.sample_size}_seed{args.seed}.json"
    )

    # Attempt to load any existing results to resume
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            results = json.load(f)
            # Example caching approach
            cache = {k.split("<SEP>")[1]: "" for res in results for k in res.keys()}
    except (FileNotFoundError, json.JSONDecodeError):
        results = []
        cache = {}

    # Process data in chunks
    chunk_size = args.num_threads * 4
    start_time = time.time()

    for i in range(0, len(datas), chunk_size):
        # Attempt reloading partial results
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            results = []

        start_idx, end_idx = i, min(i + chunk_size, len(datas))
        data_chunk = datas[start_idx:end_idx]

        # Parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = []
            for query in tqdm(data_chunk, desc="Processing Chunk", ncols=80):
                if query in cache:
                    continue
                future = executor.submit(
                    run_test_time_training,
                    query,
                    llm_engine,
                    rm,
                    gen_params=diverse_gen_params,
                    tpo_mode=args.tpo_mode,
                    max_iters=args.max_iterations,
                )
                futures.append(future)

            # Collect results
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures), desc="Completing Futures", ncols=80):
                answer = future.result()
                results.append(answer)

        # Save intermediate results
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("$" * 100)
    print(f"ELAPSED TIME: {elapsed_time:.4f} SECONDS")