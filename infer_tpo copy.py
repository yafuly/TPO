import argparse
import json
import random
import json
import concurrent
import time
from tqdm import tqdm
import textgrad as tg
from textgrad.optimizer import TextualGradientDescent
from rm import TPORewardModel


def set_random(seed=7):
    random.seed(seed)

VALID_TPO_MODES = ["tpo", "revision", "bon"]


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


def run_test_time_training_tpo(query, llm_engine, rm, gen_params, tpo_mode="tpo", max_iters=5):
    # init engine and cache
    tg.set_backward_engine(llm_engine, override=True)
    all_scores = {}

    def _update_cache(sample_responses, all_scores):
        sample_qas = [(query,o) for o in sample_responses]
        sample_scores = rm.perform_rm(sample_qas)
        # update scores in the pool and get contrastive samples
        cache_scores(all_scores, sample_scores, sample_qas, index=-1)
        _merge = [(k.split("<SEP>")[1],k.split("<SEP>")[2],v) for k,v in all_scores.items()] #q,a,s
        sample_scores = [m[2] for m in _merge] 
        sample_qas = [(m[0],m[1]) for m in _merge]
        contrastive_responses, _ = rm.get_contrastive_samples(sample_scores, sample_qas)
        chosen_response_text, rej_response_text = contrastive_responses['best'], contrastive_responses['worst']
        return chosen_response_text, rej_response_text
    
    # init generation for initial candiates
    sample_responses = llm_engine(query, **gen_params)
    chosen_response_text, rej_response_text = _update_cache(sample_responses, all_scores)

    # define variable
    evaluation_text = chosen_response_text
    response = tg.Variable(evaluation_text,
                            requires_grad=True,
                            role_description="a model response to a user query" if tpo_mode=="revision" else "a chosen response to a user query")    
    
    # define optimizer with constraints
    constraints = ["Only generate a model response."] if tpo_mode=="revision" else ["Only generate a chosen response.", "Do NOT generate a rejected response."]
    optimizer = TextualGradientDescent(engine=llm_engine, 
                                        parameters=[response], 
                                        # reward_model =rm
                                          constraints=constraints
                                        ) # add constraint to norm length?/

    # define loss function
    if tpo_mode == "revision": # for revision mode, no bad samples are provided
        evaluation_sys_text = EVALUATION_SYS_TEMPLATE_REVISION.format(
            query=query,
        )
    else: # tpo
        evaluation_sys_text = EVALUATION_SYS_TEMPLATE.format(
            query=query,
            rejected_response=rej_response_text,
        )
    loss_fn = tg.TextLoss(evaluation_sys_text)

    # start test-time training
    for i in range(max_iters):
        optimizer.zero_grad()
        # test-time preference optimziation with textual reward
        loss = loss_fn(response) # loss calculation
        loss.backward() # gradient computation
        sample_responses = optimizer.step(**gen_params) # applying gradient for variable optimization

        # prepare for next-iteration samples
        # update cache
        chosen_response_text, rej_response_text = _update_cache(sample_responses, all_scores)
        
        # update variable and loss function
        evaluation_text = chosen_response_text
        response.set_value(evaluation_text)
        if tpo_mode == "tpo": # in tpo mode, update worst sample in each turn to update the loss function
            evaluation_sys_text = EVALUATION_SYS_TEMPLATE.format(
                query=query,
                rejected_response=rej_response_text,
            )
        loss_fn = tg.TextLoss(evaluation_sys_text)

    return all_scores
    

def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
    parser.add_argument("--data_path", type=str, help="The data path.")
    parser.add_argument("--output_path", type=str, default="./results", help="The data path.")
    parser.add_argument("--reward_model", type=str, default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
    parser.add_argument("--server_model", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="Server model.")
    parser.add_argument("--ip", type=int, help="server ip.")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--tpo_mode", type=str, default="tpo")
    parser.add_argument("--num_threads", type=int, default=4, help="The number of threads to use for evaluation.")
    parser.add_argument("--batch_size", type=int, default=3, help="The batch size to use for training.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for decoding.")
    parser.add_argument("--max_tokens_response", type=int, default=2048, help="Maximum number of tokens to generate per output sequence.")
    parser.add_argument("--max_tokens_all", type=int, default=8192, help="Maximum number of tokens to generate per output sequence.")
    parser.add_argument("--max_iterations", type=int, default=5, help="The maximum number of iterations of test-time updates.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for decoding.")
    parser.add_argument("--sample_size", type=int, default=5, help="Temperature for decoding.")

    return parser.parse_args()
    

# init seed and params
args = config()
assert args.tpo_mode in VALID_TPO_MODES

# model init
set_random(args.seed)
model_name = f"server-{args.server_model}"
llm_engine = tg.get_engine(
    model_name, 
    base_url=f"http://{args.ip}:{args.port}/v1",
    api_key="token-abc123",
    max_token=args.max_tokens_all,
)
reward_model_name = args.reward_model
rm = TPORewardModel(reward_model_name)


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
rm_suffix = reward_model_name.split("/")[-1]
out_path = f"./{args.output_path}/{data_name}_model_{model_suffix}_mode_{args.tpo_mode}_rm_{rm_suffix}_max_iters{args.max_iterations}_sample_size{args.sample_size}_seed{args.seed}.json"




# load cache
try:
    cache = {}
    results = json.load(open(out_path))
    for res in results:
        for k,v in res.items():
            cache[k.split("<SEP>")[1]] = "" # i,q,a
except:
    cache = {}


# ... ready to go

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