import os
import logging
from networkx import constraint
import pytest
import json
import random
import torch
import json
import sys
import concurrent
from typing import Union, List
import time

from transformers import AutoTokenizer, pipeline
from datasets import load_dataset, Dataset, DatasetDict
from textgrad import Variable, BlackboxLLM, TextLoss
from textgrad.optimizer import TextualGradientDescent
from textgrad.engine.vllm import ChatVLLM
import textgrad as tg
from textgrad.variable import Variable
from textgrad.tasks import load_task
from textgrad.loss import PreOptTestTime
from textgrad.optimizer import TextualGradientDescent, TextualGradientDescentWithRM

from rm import TPORewardModel

import argparse

def set_random(seed=7):
    random.seed(seed)

VALID_TPO_MODES = ["dynamic", "static", "revision", "bon"]


EVALUATION_SYS_TEMPLATE = """You are a language model tasked with evaluating a chosen response by comparing with a rejected response to a user query. Analyze the strengths and weaknesses of each response, step by step, and explain why one is chosen or rejected. 

**User Query**:

{query}

**Rejected Response**:

{rejected_response}

**Do NOT generate a response to the query. Be concise.** Below is the chosen response."""

EVALUATION_SYS_TEMPLATE_REVISION = """You are a language model tasked with evaluating a model response to a user query. Analyze the strengths and weaknesses of the response, step by step. 

**User Query**:

{query}

**Do NOT generate a response to the query. Be concise.** Below is the model response."""





def cache_scores(score_cache, scores, qa_pairs, index=-1):
    for score, qa_pair, in zip(scores, qa_pairs):
        q,a = qa_pair
        key = f"INDEX{index}<SEP>{q}<SEP>{a}"
        if key in score_cache:
            print("KEY EXISTED ...")
            continue
        score_cache[key] = score



def run_test_time_training_bon(query, llm_engine, rm, gen_params, **kwargs):
    tg.set_backward_engine(llm_engine, override=True)
    all_scores = {}
    sample_responses = llm_engine(query, **gen_params)
    sample_qas = [(query,o) for o in sample_responses]
    sample_scores = rm.perform_rm(sample_qas)
    cache_scores(all_scores, sample_scores, sample_qas, index=-1)

    return all_scores


def run_test_time_training_tpo(query, llm_engine, rm, gen_params, tpo_mode="dynamic", max_iters=5):
    tg.set_backward_engine(llm_engine, override=True)
    # rm.clear_cache()
    all_scores = {}
    print("*"*100)
    print(f"QUERY: {query}")


    sample_responses = llm_engine(query, **gen_params)
    sample_qas = [(query,o) for o in sample_responses]
    sample_scores = rm.perform_rm(sample_qas)
    cache_scores(all_scores, sample_scores, sample_qas, index=-1)
    contrastive_responses, delta = rm.get_contrastive_samples(sample_scores, sample_qas)
    print("*"*100)
    print(f"DELTA: {delta}")
    chosen_response_text, rej_response_text = contrastive_responses['best'], contrastive_responses['worst']

    # init response variables and loss function
    evaluation_text = chosen_response_text
    response = tg.Variable(evaluation_text,
                            requires_grad=True,
                            role_description="a model response to a user query" if tpo_mode=="revision" else "a chosen response and a rejected response to a user query")    
    
    
    constraints = ["Only generate a model response."] if tpo_mode=="revision" else ["Only generate a chosen response.", "Do NOT generate a rejected response."]
    optimizer = TextualGradientDescent(engine=llm_engine, 
                                        parameters=[response], 
                                        # reward_model =rm
                                          constraints=constraints
                                        ) # add constraint to norm length?/


    if tpo_mode == "revision": # for revision mode, no bad samples are provided
        evaluation_sys_text = EVALUATION_SYS_TEMPLATE_REVISION.format(
            query=query,
        )
    else: # dynamic or static
        evaluation_sys_text = EVALUATION_SYS_TEMPLATE.format(
            query=query,
            rejected_response=rej_response_text,
        )

    loss_fn = tg.TextLoss(evaluation_sys_text)
    print("*"*100)
    print(f"EVAL SYS: {evaluation_sys_text}")
    print("*"*100)
    print(f"INIT SOLUTION: {response.value}")

    for i in range(max_iters):
        optimizer.zero_grad()
        loss = loss_fn(response) # gen loss
        print("*"*100)
        print(f"LOSS: {loss}")
        loss.backward() # gen grad
        new_samples = optimizer.step(**gen_params) # apply grad
        if len(new_samples) == 0:
            new_samples = [evaluation_text] # fail to extract improved variables, use last-turn best response
            print("Warning: none valid improved variables, reuse the best response from last turn ...".upper())
        print("*"*100)
        print(f"POST SOLUTION: {new_samples[0]}")


        # get rm scores for samples at this round
        sample_qas = [(query,o) for o in new_samples]
        sample_scores= rm.perform_rm(sample_qas)
        cache_scores(all_scores, sample_scores, sample_qas, index=i)
        _merge = [(k.split("<SEP>")[1],k.split("<SEP>")[2],v) for k,v in all_scores.items()] #q,a,s
        # update scores in the pool and get contrastive samples
        sample_scores = [m[2] for m in _merge]
        sample_qas = [(m[0],m[1]) for m in _merge]
        print("+"*100)
        print("sample_scores".upper())
        print(sample_scores)
        # model_response_text = response.value
        contrastive_responses, delta = rm.get_contrastive_samples(sample_scores, sample_qas)
        print("*"*100)
        print(f"DELTA: {delta}")
        chosen_response_text, rej_response_text = contrastive_responses['best'], contrastive_responses['worst']

        evaluation_text = chosen_response_text
        response.set_value(evaluation_text) # update response
        if tpo_mode == "dynamic": # in dynamic mode, update worst sample in each turn to update the loss function
            evaluation_sys_text = EVALUATION_SYS_TEMPLATE.format(
                query=query,
                rejected_response=rej_response_text,
            )
        # update loss function
        loss_fn = tg.TextLoss(evaluation_sys_text)
        print("*"*100)
        print(f"EVAL SYS: {evaluation_sys_text}")
    # eval all responses
    
    return all_scores
    

def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
    parser.add_argument("--data_path", type=str, help="The data path.")
    parser.add_argument("--reward_model", type=str, default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
    parser.add_argument("--tpo_mode", type=str, default="dynamic")
    parser.add_argument("--ip", type=int, help="server ip.")
    parser.add_argument("--batch_size", type=int, default=3, help="The batch size to use for training.")
    parser.add_argument("--max_epochs", type=int, default=3, help="The maximum number of epochs to train for.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for decoding.")
    parser.add_argument("--num_threads", type=int, default=4, help="The number of threads to use for evaluation.")
    # customized
    parser.add_argument("--ngpu", type=int, default=1, help="The number of GPUs to use for backend LLM optimizer.")
    parser.add_argument("--max_tokens_response", type=int, default=2048, help="Maximum number of tokens to generate per output sequence.")
    parser.add_argument("--max_tokens_all", type=int, default=8192, help="Maximum number of tokens to generate per output sequence.")
    parser.add_argument("--max_iterations", type=int, default=5, help="The maximum number of iterations of test-time updates.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for decoding.")
    parser.add_argument("--sample_size", type=int, default=5, help="Temperature for decoding.")
    parser.add_argument("--server_model", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="Server model.")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--len_norm", action='store_true', default=False)
    parser.add_argument("--suffix", type=str, default="", help="Output path suffix.")
    return parser.parse_args()
    

# init seed and params
args = config()
assert args.tpo_mode in VALID_TPO_MODES

# model init
set_random(args.seed)
print("+"*100)
print(f"Server: http://10.140.54.{args.ip}:{args.port}/v1")
model_name = f"server-{args.server_model}"
llm_engine = tg.get_engine(
    model_name, # Qwen/Qwen2.5-72B-Instruct
    base_url=f"http://10.140.54.{args.ip}:{args.port}/v1",
    api_key="token-abc123",
    max_token=args.max_tokens_all,
)

if args.reward_model == "sfairXC/FsfairX-LLaMA3-RM-v0.1" or "allenai/Llama-3.1-Tulu-3-8B-RM":
    rm = TPORewardModel(args.reward_model, len_norm=args.len_norm)


# data init
data_path = args.data_path
data_name = data_path.split("/")[-1]
datas = json.load(open(data_path))
random.shuffle(datas)


# gen func init
diverse_gen_params = {
    "n": args.sample_size,
    "temperature": args.temperature,
    "top_p": 0.95,
    "seed": args.seed,
    "max_tokens": args.max_tokens_response,
}
run_test_time_training = run_test_time_training_bon if args.tpo_mode == "bon" else run_test_time_training_tpo


# results init
model_suffix = model_name.split("/")[-1]
out_path = f"./results/{data_name}_model_{model_suffix}_mode_{args.tpo_mode}_max_iters{args.max_iterations}_sample_size{args.sample_size}_seed{args.seed}.json"
if args.len_norm:
    out_path = f"./results/{data_name}_model_{model_suffix}_mode_{args.tpo_mode}_len_norm_max_iters{args.max_iterations}_sample_size{args.sample_size}_seed{args.seed}.json"
if args.reward_model == "allenai/Llama-3.1-Tulu-3-8B-RM":
    out_path = f"{out_path}.TuluRM.json"
if args.suffix != "":
    out_path = f"{out_path}.{args.suffix}.json"

print("*"*100)
print(out_path)
try:
    cache = {}
    results = json.load(open(out_path))
    for res in results:
        for k,v in res.items():
            cache[k.split("<SEP>")[1]] = "" # i,q,a
except:
    cache = {}


# ... ready to go
print(len(cache))
from tqdm import tqdm
scores = []
chunk_size = args.num_threads * 4
start_time = time.time()
for i in range(0,len(datas),chunk_size):
    try:
        results = json.load(open(out_path, encoding='utf-8'))
    except:
        results = []

    start, end = i, min(i + chunk_size, len(datas))
    data_chunks = datas[start:end]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for query in tqdm(data_chunks):
            if query in cache:
                continue
            future = executor.submit(
                run_test_time_training, 
                query, 
                llm_engine, 
                rm, 
                gen_params=diverse_gen_params, 
                tpo_mode=args.tpo_mode,
                max_iters=args.max_iterations)
            futures.append(future)

        all_history = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0):
            answer = future.result()
            results.append(answer)

    # save results

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
end_time = time.time()
elapsed_time = end_time - start_time
print("$"*100)
print(f"Elapsed time: {elapsed_time:.4f} seconds".upper())