import textgrad as tg
from textgrad.optimizer import TextualGradientDescent
from rm import TPORewardModel

############################################################
# Prompt Templates
############################################################

EVALUATION_SYS_TEMPLATE = """You are a language model tasked with evaluating a chosen response by comparing it with a rejected response to a user query. Analyze the strengths and weaknesses of each response, step by step, and explain why one is chosen or rejected.

**User Query**:
{query}

**Rejected Response**:
{rejected_response}

**Do NOT generate a response to the query. Be concise.** Below is the chosen response."""

EVALUATION_SYS_TEMPLATE_REVISION = """You are a language model tasked with evaluating a model response to a user query. Analyze the strengths and weaknesses of the response, step by step.

**User Query**:
{query}

**Do NOT generate a response to the query. Be concise.** Below is the model response."""


############################################################
# Caching Utilities
############################################################

def cache_scores(score_cache: dict,
                 scores: list,
                 qa_pairs: list,
                 index: int = -1) -> None:
    """
    Caches the reward model scores for a set of (question, answer) pairs.

    :param score_cache: dictionary to store scores keyed by 'INDEX{index}<SEP>{q}<SEP>{a}'.
    :param scores: list of scores returned by the reward model.
    :param qa_pairs: list of (question, answer) tuples.
    :param index: index to uniquely identify the caching iteration.
    """
    for score, qa_pair in zip(scores, qa_pairs):
        q, a = qa_pair
        key = f"INDEX{index}<SEP>{q}<SEP>{a}"
        if key not in score_cache:
            score_cache[key] = score


############################################################
# Best-of-N (BoN) Inference-time Alignment
############################################################

def run_test_time_training_bon(query: str,
                               llm_engine,
                               rm: TPORewardModel,
                               gen_params: dict,
                               **kwargs) -> dict:
    """
    Runs the Best-of-N (BoN) sampling approach at test time, without iterative refinement.
    Samples responses, computes reward model scores, and returns a cache of scores.

    :param query: The user query (string).
    :param llm_engine: LLM inference engine from textgrad.get_engine().
    :param rm: TPORewardModel instance for reward scoring.
    :param gen_params: Generation parameters for the LLM engine.
    :return: Dictionary of all scores keyed by 'INDEX-1<SEP>{q}<SEP>{a}'.
    """
    tg.set_backward_engine(llm_engine, override=True)

    all_scores = {}
    sample_responses = llm_engine(query, **gen_params)
    sample_qas = [(query, resp) for resp in sample_responses]

    # Compute reward model scores
    sample_scores = rm.perform_rm(sample_qas)
    cache_scores(all_scores, sample_scores, sample_qas, index=-1)

    return all_scores


############################################################
# Test-time Preference Optimization (TPO)
############################################################

def run_test_time_training_tpo(query: str,
                               llm_engine,
                               rm: TPORewardModel,
                               gen_params: dict,
                               tpo_mode: str = "tpo",
                               max_iters: int = 5) -> dict:
    """
    Runs the Test-time Preference Optimization (TPO) process by repeatedly
    refining the chosen response according to reward model feedback.

    :param query: The user query (string).
    :param llm_engine: LLM inference engine from textgrad.
    :param rm: TPORewardModel for scoring.
    :param gen_params: Generation parameters for sampling responses.
    :param tpo_mode: Mode of TPO - 'tpo', 'revision', or 'bon'.
    :param max_iters: Number of optimization iterations to perform.
    :return: Dictionary of all scored (query, answer) pairs.
    """
    tg.set_backward_engine(llm_engine, override=True)
    all_scores = {}

    def _update_cache(sample_resps: list, score_db: dict, index:int):
        # Compute scores for new responses
        sample_qas_ = [(query, resp) for resp in sample_resps]
        sample_scores_ = rm.perform_rm(sample_qas_)
        cache_scores(score_db, sample_scores_, sample_qas_, index=index)

        # Flatten the cached data into (q, a, score) list
        merged = []
        for k, v in score_db.items():
            # k looks like 'INDEX-1<SEP>{q}<SEP>{a}'
            _, q_, a_ = k.split("<SEP>")
            merged.append((q_, a_, v))

        # Identify best and worst samples from the updated cache
        sample_scores_vals = [m[2] for m in merged]
        sample_qas_vals = [(m[0], m[1]) for m in merged]

        contrastive_responses, _ = rm.get_contrastive_samples(sample_scores_vals, sample_qas_vals)
        chosen_resp_text = contrastive_responses['best']
        rej_resp_text = contrastive_responses['worst']
        return chosen_resp_text, rej_resp_text

    # 1) Initial sampling for candidates
    init_responses = llm_engine(query, **gen_params)
    chosen_resp_text, rej_resp_text = _update_cache(init_responses, all_scores, index=-1)

    # 2) Define the variable to be optimized
    response_role = ("a model response to a user query"
                     if tpo_mode == "revision"
                     else "a chosen response to a user query")
    response = tg.Variable(
        chosen_resp_text,
        requires_grad=True,
        role_description=response_role,
    )

    # 3) Constraints for textual updates
    constraints = (["Only generate a model response."]
                   if tpo_mode == "revision"
                   else ["Only generate a chosen response.", "Do NOT generate a rejected response."])

    # 4) Create the TPO optimizer
    optimizer = TextualGradientDescent(
        engine=llm_engine,
        parameters=[response],
        constraints=constraints,
    )

    # 5) Define the loss function (TextLoss)
    if tpo_mode == "revision":
        # No rejected sample provided
        evaluation_sys_text = EVALUATION_SYS_TEMPLATE_REVISION.format(query=query)
    else:
        # TPO mode, includes rejected response
        evaluation_sys_text = EVALUATION_SYS_TEMPLATE.format(
            query=query,
            rejected_response=rej_resp_text,
        )
    loss_fn = tg.TextLoss(evaluation_sys_text)

    # 6) Start test-time training loop
    for i in range(max_iters):
        optimizer.zero_grad()

        # 6.1) Compute textual loss
        loss = loss_fn(response)

        # 6.2) Compute textual gradients
        loss.backward()

        # 6.3) Update variable using textual gradients
        new_responses = optimizer.step(**gen_params)

        # 6.4) Update cache with new responses, get chosen and rejected
        chosen_resp_text, rej_resp_text = _update_cache(new_responses, all_scores, index=i)

        # 6.5) Update the variable's content
        response.set_value(chosen_resp_text)

        # 6.6) Update the loss function if needed
        if tpo_mode == "tpo":
            # In TPO mode, update the rejected response for the next iteration
            evaluation_sys_text = EVALUATION_SYS_TEMPLATE.format(
                query=query,
                rejected_response=rej_resp_text,
            )
        loss_fn = tg.TextLoss(evaluation_sys_text)

    return all_scores
