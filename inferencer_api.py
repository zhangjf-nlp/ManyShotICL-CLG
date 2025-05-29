import argparse
import json
import logging
import os
import re
import time
from typing import List
from datetime import datetime
from tqdm import tqdm

import numpy as np
import requests
from transformers import AutoTokenizer

from inferencer_fast import stack_demos
from shared_context_finder import get_shared_ctxs_path
from src.dataset_readers.dataset_wrappers import get_dataset_wrapper
from src.dataset_readers.dataset_wrappers.base_dsw import ABC
from src.metrics import get_metric

logger = logging.getLogger(__name__)

def get_base_model_name(model_name):
    if 'llama' in model_name.lower():
        return "Llama-3-8b"
    elif 'qwen' in model_name.lower():
        return 'Qwen2.5-3B'
    else:
        raise NotImplementedError(model_name)


headers = ... # some sensitive information
url = ... # some sensitive information
system_message = r"""
You are an In-Context Learner to learn from task demonstrations provided in context.
User will present you a prompt, composed of several demonstrations (query + response) and a test-query (only query).
You need to under stand the task concept and response format through the demonstrations, and return your response to the test query.

Example of a valid JSON response:
```json
{
    "response": "...",
}```
""".strip()
def make_api_call(api_model_name, prompt, stop_strings, max_retry=5):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    data = {
        "model": api_model_name,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.0, "top_p": 0.0,
        "stream": False,
        "type": "json_object",
    }
    for retry in range(max_retry):
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.encoding = 'utf-8'  # 避免乱码
        if response.status_code == 200:
            try:
                response_string = response.text
                text = json.loads(response_string)["choices"][0]["message"]["content"]
                text_json = text.strip('```json\n').rstrip('```')
                return json.loads(text_json)["response"]
            except:
                pass
        elif response.status_code in [408, 429]:
            time.sleep(2**retry)
        else:
            time.sleep(3)
    return None


class APIInferencer:
    def __init__(
        self,
        api_model_name: str,
        model_name: str,
        tokenizer,
        task_names: List[str],
        method_name: str,
        num_shots: List[int],
        seed: int,
        erase: int,
        max_prompt_len: int,
    ):
        self.api_model_name = api_model_name
        self.model_name = model_name
        self.task_names = task_names
        self.method_name = method_name
        self.num_shots = num_shots
        self.seed = seed
        self.erase = erase
        self.max_prompt_len = max_prompt_len
        self.tokenizer = tokenizer

    def prepare_prompts(self, queries, task_name, ice_separator, method_name, num_shot, seed):
        ctxs_file = get_shared_ctxs_path(
            model_name=get_base_model_name(self.model_name),
            task_name=task_name, method=method_name,
            num_ice=num_shot, seed=seed,
        )
        if not os.path.exists(ctxs_file):
            print(f"ctxs file not exists: {ctxs_file}")
            return None, None
        with open(ctxs_file, "r", encoding='utf-8') as f:
            ctxs = json.loads(f.read()) # list of qa
        assert len(ctxs) == num_shot, ctxs_file
        dsw: ABC = get_dataset_wrapper(task_name, dataset_split="train", ds_size=None)
        qa_corpus = dsw.get_corpus(field="qa")
        q_corpus = [_.replace("{ice_prompt}", "") for _ in dsw.get_corpus(field="gen_a")]
        a_corpus = dsw.get_corpus(field="a")
        demonstrations = []
        for qa in ctxs:
            assert qa in qa_corpus, qa
            index = qa_corpus.index(qa)
            q = q_corpus[index]
            a = a_corpus[index]
            assert qa == q + a, (qa, q, a)
            demonstrations.append({
                "query": q,
                "response": a,
            })
            if len(self.tokenizer(json.dumps(demonstrations, ensure_ascii=False, indent=4)).input_ids) > self.max_prompt_len:
                demonstrations.pop(-1)
                print(f"only use {len(demonstrations)}/{len(ctxs)} demonstrations for limited context length")
                break
        prompts = []
        for query in queries:
            assert query.startswith("{ice_prompt}"), query
            query = query.replace("{ice_prompt}", "")
            prompts.append("```json\n" + json.dumps({
                "demonstrations": demonstrations,
                "test-query": {
                    "query": query
                }
            }, ensure_ascii=False, indent=4) + "\n```")
        return prompts, list(range(len(prompts)))

    def write_prompts(self):
        all_prompts = []
        for task_name in self.task_names:
            inference_dataset_wrapper: ABC = get_dataset_wrapper(
                task_name,
                dataset_split="validation",
                ds_size=300,
            )
            ice_separator = inference_dataset_wrapper.ice_separator
            a_prefix = inference_dataset_wrapper.a_prefix
            queries = inference_dataset_wrapper.get_corpus("gen_a")
            reference = inference_dataset_wrapper.dataset
            if 'choices' in inference_dataset_wrapper.field_getter:
                choices = inference_dataset_wrapper.get_field(inference_dataset_wrapper[0], "choices")
            else:
                choices = None
            for method_name, seed in [
                ("random", 42),
                ("random", 43),
                ("random", 44),
                ("random", 45),
                ("random", 46),
                ("best_of_5", 42),
                ("latent_grad", 42),
            ]:
                prompts, indexes = self.prepare_prompts(queries, task_name, ice_separator, method_name, 128, seed)
            all_prompts += prompts
        with open(f"api_prompts.json", "w", encoding='utf-8') as f:
            f.write(json.dumps(all_prompts, ensure_ascii=False, indent=4))

    def run(self):
        results = {}
        total_tasks = len(self.task_names)
        total_shots = len(self.num_shots)
        for task_idx, task_name in enumerate(self.task_names, 1):
            results[task_name] = {}
            print(f"Processing Task {task_idx}/{total_tasks}: {task_name}")
            inference_dataset_wrapper: ABC = get_dataset_wrapper(
                task_name,
                dataset_split="validation",
                ds_size=300,
            )
            ice_separator = inference_dataset_wrapper.ice_separator
            a_prefix = inference_dataset_wrapper.a_prefix
            queries = inference_dataset_wrapper.get_corpus("gen_a")
            #queries = [query + a_prefix for query in queries]
            reference = inference_dataset_wrapper.dataset
            if 'choices' in inference_dataset_wrapper.field_getter:
                choices = inference_dataset_wrapper.get_field(inference_dataset_wrapper[0], "choices")
            else:
                choices = None
            evaluator = get_metric(task_name)
            for shot_idx, num_shot in enumerate(self.num_shots, 1):
                print(f"    Processing Num-Shot {shot_idx}/{total_shots}: {num_shot}")
                prompts, indexes = self.prepare_prompts(queries, task_name, ice_separator, self.method_name, num_shot, self.seed)
                cached_file = f"output/inferencer_api_cached/{task_name}-{self.method_name}-{self.api_model_name}-{num_shot}shots-{self.seed}seed.json"
                record_file = f"records/time_and_score/{task_name}-{self.method_name}-{self.api_model_name}.jsonl"
                reference = [reference[i] for i in indexes]
                prediction, num_none_response = self.inference(
                    prompts=prompts, choices=choices,
                    stop_strings=inference_dataset_wrapper.stop_strings,
                    cached_file=cached_file,
                    reference=reference[:3]
                )
                if len(a_prefix) > 0:
                    prediction_wo_a_prefix = []
                    for pred in prediction:
                        if isinstance(pred, str) and pred.strip().startswith(a_prefix):
                            pred = pred.strip()[len(a_prefix):]
                        prediction_wo_a_prefix.append(pred)
                    prediction = prediction_wo_a_prefix
                metric = evaluator.evaluate(prediction, reference)
                score = list(metric.values())[0]
                with open(record_file, "a+", encoding='utf-8') as f:
                    f.write(json.dumps({
                        "score": f"{score:.4f}",
                        "num_shots": num_shot,
                        "seed": self.seed,
                        "num_none_response": num_none_response,
                        "datetime": datetime.now().strftime('%Y%m%d%H%M'),
                    }, ensure_ascii=False) + "\n")
                print(f"method-seed: {self.method_name}-{self.seed}")
                print(f"task: {task_name}")
                print(f"none response: {num_none_response}/{len(prediction)}")
                print(f"score: {score:.4f}")
        return

    def inference(self, prompts, choices, stop_strings, cached_file, reference):
        if os.path.exists(cached_file) and not self.erase:
            with open(cached_file, 'r', encoding='utf-8') as f:
                loaded_cache = json.loads(f.read())
        else:
            loaded_cache = []

        cache = []
        prediction = []
        num_none_response = 0
        for i, prompt in enumerate(tqdm(prompts)):
            if len(loaded_cache) > i and loaded_cache[i]["prompt"] == prompt and loaded_cache[i]["response"] is not None:
                response = str(loaded_cache[i]["response"])
            else:
                response = make_api_call(self.api_model_name, prompt, stop_strings, max_retry=5)
            cache.append({
                "prompt": prompt,
                "response": response,
            })
            if response is None:
                num_none_response += 1
                pred = -1 if choices is not None else ""
            else:
                if choices is not None:
                    pred = choices.index(response) if response in choices else -1
                else:
                    pred = response
            prediction.append(pred)
            if i < 1:
                query = prompts[i].split("\n")[-4]
                print(f"query: {query}")
                print(f"response: {response}")
                print(f"reference[{i}]: {reference[i]}")
        with open(cached_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(cache, ensure_ascii=False, indent=4))

        return prediction, num_none_response


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using vLLMInferencer.")
    parser.add_argument("--model_name", type=str,
                        default="Qwen2.5-3B",
                        help="Name of the model for demonstration selection.")
    parser.add_argument("--api_model_name", type=str,
                        default="qwen-turbo-latest",
                        choices=[
                            "qwen-turbo-latest",
                            "glm-4-flash",
                            "gpt-4o-mini",
                            "Doubao-pro-32k-0528",
                            "deepseek-v3",
                            "deepseek-v3-local",
                        ],
                        help="Name of the api model.")
    parser.add_argument("--task_names", type=str,
                        default="sst5-mnli-cmsqa-swag-geoquery-nl2bash-break-mtop-smcalflow",
                        help="Hyphen-separated list of tasks.")
    parser.add_argument("--method_name", type=str,
                        default=None,
                        help="Hyphen-separated list of method names.")
    parser.add_argument("--num_shots", type=str,
                        default="128",
                        help="Hyphen-separated list of number of shots.")
    parser.add_argument("--seed", type=str,
                        default="42",
                        help="Hyphen-separated list of seeds.")
    parser.add_argument("--max_prompt_len", type=int,
                        default=None,
                        help="Maximum prompt length.")
    parser.add_argument("--erase", type=int, default=0,
                        help="Ignore the cached results if erase.")
    args = parser.parse_args()

    # Convert comma-separated strings to lists
    args.task_names = args.task_names.split('-')
    args.num_shots = list(map(int, args.num_shots.split('-')))

    if args.max_prompt_len is None:
        args.max_prompt_len = 7000

    return args


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if args.api_model_name == "deepseek-v3-local":
        inferencer = APIInferencer(
            api_model_name=args.api_model_name,
            model_name=args.model_name,
            tokenizer=tokenizer,
            task_names=args.task_names,
            method_name="random",
            num_shots=[128],
            seed=42,
            max_prompt_len=args.max_prompt_len,
            erase=args.erase,
        )
        inferencer.write_prompts()
    elif args.method_name is None:
        # run default settings
        default_params = [
            ("latent_grad", 42),
            ("random", 42),
            ("random", 43),
            ("random", 44),
            ("random", 45),
            ("random", 46),
            ("best_of_5", 42),
            ("latent_post", 42),
            ("dense_epr_kmeans", 42),
            ("dense_bge_kmeans", 42),
            ("bm25", 42),
        ]
        #default_params.reverse()
        for method_name, seed in default_params:
            inferencer = APIInferencer(
                api_model_name=args.api_model_name,
                model_name=args.model_name,
                tokenizer=tokenizer,
                task_names=args.task_names,
                method_name=method_name,
                num_shots=[128],
                seed=seed,
                max_prompt_len=args.max_prompt_len,
                erase=args.erase,
            )
            inferencer.run()
    else:
        inferencer = APIInferencer(
            api_model_name=args.api_model_name,
            model_name=args.model_name,
            tokenizer=tokenizer,
            task_names=args.task_names,
            method_name=args.method_name,
            num_shots=args.num_shots,
            seed=args.seed,
            max_prompt_len=args.max_prompt_len,
            erase=args.erase,
        )
        inferencer.run()


if __name__ == "__main__":
    main()
