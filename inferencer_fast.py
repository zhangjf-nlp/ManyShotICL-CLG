import argparse
import json
import logging
import os
import time
from typing import List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from shared_context_finder import get_shared_ctxs_path, get_dependent_ctxs_path
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


def get_chat_template(model_name):
    if not 'instruct' in model_name.lower():
        return "{demonstrations}{query}"
    elif 'llama' in model_name.lower():
        raise NotImplementedError(model_name)
    elif 'qwen' in model_name.lower():
        return ("<|im_start|>system\n"
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "Please perform In-Context Learning on the following query-response demonstrations:\n"
                "{demonstrations}\n"
                "Now, directly complete the response for the following query:\n"
                "{query}<|im_end|>\n"
                "<|im_start|>assistant\n"
                "{query}")
    else:
        raise NotImplementedError(model_name)



def stack_demos(ctxs, ice_separator, tokenizer, max_prompt_len, verbose=1):
    prompt = ""
    prompt_length = 0
    for i, demo in enumerate(ctxs, start=1):
        prompt_length += len(tokenizer(demo+ice_separator).input_ids)
        if prompt_length > max_prompt_len:
            if verbose:
                print(f"The selected demonstrations exceed max_prompt_len of {max_prompt_len}, "
                      f"so only the previous ({i}/{len(ctxs)}) ones are used.")
            break
        prompt = prompt + demo + ice_separator
    return prompt, prompt_length


class vLLMInferencer:
    def __init__(
        self,
        model_name: str,
        task_names: List[str],
        method_names: List[str],
        dependent: bool,
        hybrid: bool,
        shuffle: bool,
        erase: bool,
        num_shots: List[int],
        seeds: List[int],
        max_prompt_len: int,
        tp: int,
    ):
        self.model_name = model_name
        self.task_names = task_names
        self.method_names = method_names
        self.num_shots = num_shots
        self.dependent = dependent
        self.hybrid = hybrid
        self.shuffle = shuffle
        self.erase = erase
        self.seeds = seeds
        self.max_prompt_len = max_prompt_len
        self.tp = tp
        self.initialize_model_and_tokenizer()

    def initialize_model_and_tokenizer(self):
        from vllm import LLM
        from transformers import AutoTokenizer
        self.model = LLM(
            model=self.model_name,
            enable_prefix_caching=True, trust_remote_code=True, dtype="auto",
            swap_space=32, seed=42, block_size=32, tensor_parallel_size=self.tp,
            distributed_executor_backend="ray",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    def prepare_prompts(self, queries, task_name, ice_separator, method_name, num_shot, seed) -> Tuple:
        template = get_chat_template(self.model_name)
        if self.dependent:
            ctxs_dp_file = get_dependent_ctxs_path(
                model_name=get_base_model_name(self.model_name),
                task_name=task_name, method=method_name,
                num_ice=num_shot, seed=seed,
            )
            if not os.path.exists(ctxs_dp_file):
                print(f"ctxs_dp file not exists: {ctxs_dp_file}")
                return None, None
            with open(ctxs_dp_file, "r", encoding='utf-8') as f:
                ctxs_dp = json.loads(f.read()) # {"ctxs_dp": List[List[qa]], "test_index": List[int]}
            prompts = []
            for index, ctxs in zip(ctxs_dp["test_index"], ctxs_dp["ctxs_dp"]):
                query = queries[index]
                assert query.startswith("{ice_prompt}"), query
                query = query.replace("{ice_prompt}", "")
                query = template.replace("{query}", query)
                query_length = len(self.tokenizer(query).input_ids)
                demonstrations, demonstrations_length = stack_demos(
                    ctxs=ctxs, ice_separator=ice_separator,
                    tokenizer=self.tokenizer, max_prompt_len=self.max_prompt_len-query_length
                )
                prompts.append(query.replace("{demonstrations}", demonstrations))
            return prompts, ctxs_dp["test_index"]
        elif self.hybrid:
            ctxs_dp_file = get_dependent_ctxs_path(
                model_name=get_base_model_name(self.model_name),
                task_name=task_name, method="dense_bge",
                num_ice=4, seed=42,
            )
            if not os.path.exists(ctxs_dp_file):
                print(f"ctxs_dp file not exists: {ctxs_dp_file}")
                return None, None
            with open(ctxs_dp_file, "r", encoding='utf-8') as f:
                ctxs_dp = json.loads(f.read()) # {"ctxs_dp": List[List[qa]], "test_index": List[int]}
            if method_name == "none":
                shared_ctxs = []
            else:
                ctxs_file = get_shared_ctxs_path(
                    model_name=get_base_model_name(self.model_name),
                    task_name=task_name, method=method_name,
                    num_ice=num_shot, seed=seed,
                )
                if not os.path.exists(ctxs_file):
                    print(f"ctxs file not exists: {ctxs_file}")
                    return None, None
                with open(ctxs_file, "r", encoding='utf-8') as f:
                    shared_ctxs = json.loads(f.read()) # list of qa
                assert len(shared_ctxs) == num_shot, ctxs_file
            prompts, demonstrations_lengths = [], []
            for index, ctxs in zip(ctxs_dp["test_index"], ctxs_dp["ctxs_dp"]):
                query = queries[index]
                assert query.startswith("{ice_prompt}"), query
                query = query.replace("{ice_prompt}", "")
                query = template.replace("{query}", query)
                query_length = len(self.tokenizer(query).input_ids)
                instance_demonstrations, instance_demonstrations_length = stack_demos(
                    ctxs=ctxs, ice_separator=ice_separator,
                    tokenizer=self.tokenizer, max_prompt_len=self.max_prompt_len-query_length,
                )
                task_demonstrations, task_demonstrations_length = stack_demos(
                    ctxs=shared_ctxs, ice_separator=ice_separator,
                    tokenizer=self.tokenizer, max_prompt_len=self.max_prompt_len-query_length-instance_demonstrations_length,
                )
                prompts.append(query.replace("{demonstrations}", task_demonstrations+instance_demonstrations))
                demonstrations_lengths.append(task_demonstrations_length+instance_demonstrations_length)
            print(f"avg demonstrations_length for {num_shot}-shots: {np.mean(demonstrations_lengths):.2f} tokens")
            return prompts, ctxs_dp["test_index"]
        else:
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
            if self.shuffle:
                ctxs = [ctxs[i] for i in np.random.permutation(len(ctxs))]
            prompts, demonstrations_lengths = [], []
            for query in queries:
                assert query.startswith("{ice_prompt}"), query
                query = query.replace("{ice_prompt}", "")
                query = template.replace("{query}", query)
                query_length = len(self.tokenizer(query).input_ids)
                demonstrations, demonstrations_length = stack_demos(
                    ctxs=ctxs, ice_separator=ice_separator,
                    tokenizer=self.tokenizer, max_prompt_len=self.max_prompt_len-query_length
                )
                prompts.append(query.replace("{demonstrations}", demonstrations))
                demonstrations_lengths.append(demonstrations_length)
            print(f"avg demonstrations_length for {num_shot}-shots: {np.mean(demonstrations_lengths):.2f} tokens")
            return prompts, list(range(len(prompts)))

    def run(self):
        results = {}
        total_tasks = len(self.task_names)
        total_methods = len(self.method_names)
        total_shots = len(self.num_shots)
        total_seeds = len(self.seeds)
        for task_idx, task_name in enumerate(self.task_names, 1):
            results[task_name] = {}
            print(f"Processing Task {task_idx}/{total_tasks}: {task_name}")
            inference_dataset_wrapper: ABC = get_dataset_wrapper(
                task_name,
                dataset_split="validation",
                ds_size=10000,
            )
            ice_separator = inference_dataset_wrapper.ice_separator
            a_prefix = inference_dataset_wrapper.a_prefix
            queries = inference_dataset_wrapper.get_corpus("gen_a")
            queries = [query + a_prefix for query in queries]
            reference = inference_dataset_wrapper.dataset
            if 'choices' in inference_dataset_wrapper.field_getter:
                choices = inference_dataset_wrapper.get_field(inference_dataset_wrapper[0], "choices")
            else:
                choices = None
            evaluator = get_metric(task_name)
            for method_idx, method_name in enumerate(self.method_names, 1):
                results[task_name][method_name] = {}
                print(f"  Processing Method {method_idx}/{total_methods}: {method_name}")
                for shot_idx, num_shot in enumerate(self.num_shots, 1):
                    results[task_name][method_name][num_shot] = []
                    print(f"    Processing Num-Shot {shot_idx}/{total_shots}: {num_shot}")
                    for seed_idx, seed in enumerate(self.seeds, 1):
                        if seed_idx > 1 and method_name not in ["random"]:
                            continue
                        prompts, indexes = self.prepare_prompts(queries, task_name, ice_separator, method_name, num_shot, seed)
                        dp = "-instance_level" if self.dependent else ""
                        hb = "-hybrid" if self.hybrid else ""
                        sf = "-shuffle" if self.shuffle else ""
                        cached_file = f"output/inferencer_fast_cached/{task_name}{dp}{hb}{sf}-{method_name}-{os.path.basename(self.model_name)}-{num_shot}shots-{seed}seed.json"
                        record_file = f"records/time_and_score/{task_name}{dp}{hb}{sf}-{method_name}-{os.path.basename(self.model_name)}.jsonl"
                        if prompts is None:
                            metric = "unfound"
                            score = np.nan
                            duration = np.nan
                        else:
                            reference = [reference[i] for i in indexes]
                            prediction, duration = self.inference(
                                prompts=prompts, choices=choices,
                                stop_strings=inference_dataset_wrapper.stop_strings,
                                allowed_strings=inference_dataset_wrapper.allowed_strings,
                                cached_file=cached_file, record_file=record_file, num_shot=num_shot,
                            )
                            if seed_idx == 1 and shot_idx == 1:
                                print(f"prompts[0]: {prompts[0]}")
                                print(f"prediction[0]: {prediction[0]}")
                                print(f"reference[0]: {reference[0]}")
                            metric = evaluator.evaluate(prediction, reference)
                            score = list(metric.values())[0]
                            """
                            for zhuang especially
                            """
                            if task_name in ["za2zh", "zh2za"]:
                                for metric_name, metric_value in metric.items():
                                    if not f"{task_name}-{metric_name}" in results:
                                        results[f"{task_name}-{metric_name}"] = {}
                                    if not method_name in results[f"{task_name}-{metric_name}"]:
                                        results[f"{task_name}-{metric_name}"][method_name] = {}
                                    if not num_shot in results[f"{task_name}-{metric_name}"][method_name]:
                                        results[f"{task_name}-{metric_name}"][method_name][num_shot] = []
                                    results[f"{task_name}-{metric_name}"][method_name][num_shot].append(metric_value)
                        results[task_name][method_name][num_shot].append(score)
                        print(f"      Processing Seed {seed_idx}/{total_seeds}: {seed} - Metric: {metric}")
                        with open(record_file, "a+", encoding='utf-8') as f:
                            f.write(json.dumps({
                                "duration": duration,
                                "score": f"{score:.4f}",
                                "num_shots": num_shot,
                                "seed": seed,
                                "datetime": datetime.now().strftime('%Y%m%d%H%M'),
                            }, ensure_ascii=False) + "\n")
                self.print_results(results)
        return results

    def print_results(self, results):
        pd.set_option('display.max_columns', None)  # 不省略列
        pd.set_option('display.max_rows', None)     # 不省略行
        for task_name in results.keys():
            # Mean
            df = pd.DataFrame()
            for method_name, method_data in results[task_name].items():
                avg_scores = {num_shot: np.mean([score for score in scores if not np.isnan(score)])
                              for num_shot, scores in method_data.items()}
                avg_scores["avg"] = np.mean(list(avg_scores.values()))
                df[method_name] = pd.Series(avg_scores)
            print(f"Task: {task_name} - mean")
            print(df)
            # Std
            df = pd.DataFrame()
            for method_name, method_data in results[task_name].items():
                std_scores = {num_shot: np.std([score for score in scores if not np.isnan(score)])
                              for num_shot, scores in method_data.items()}
                std_scores["avg"] = np.mean(list(std_scores.values()))
                df[method_name] = pd.Series(std_scores)
            print(f"Task: {task_name} - std")
            print(df)
        return

    def inference(self, prompts, choices, stop_strings, allowed_strings, cached_file, record_file, num_shot):
        from vllm import SamplingParams
        from vllm.entrypoints.openai.logits_processors import _get_allowed_token_ids_logits_processor
        if os.path.exists(cached_file) and not self.erase:
            with open(cached_file, 'r', encoding='utf-8') as f:
                content = json.loads(f.read())
            if content["prompts"] == prompts:
                prediction = content["prediction"]
                duration = content["duration"]
                print(f"resuse cached prediction from {cached_file}")
                return prediction, duration

        prediction = []
        if choices is not None:
            choices_ids = self.tokenizer([choice+"\n" for choice in choices]).input_ids
            allowed_token_ids = set(id for ids in choices_ids for id in ids)
            start_time = time.time()
            outputs = self.model.generate(
                prompts=prompts,
                sampling_params=SamplingParams(
                    n=1, temperature=0.0, stop=stop_strings,
                    max_tokens=max([len(ids) for ids in choices_ids]),
                    logits_processors=[
                        _get_allowed_token_ids_logits_processor(
                            frozenset(allowed_token_ids),
                            len(self.tokenizer)
                        ),
                    ],
                ),
                use_tqdm=True,
            )
            end_time = time.time()
            for output in outputs:
                text = output.outputs[0].text
                pred = choices.index(text) if text in choices else -1
                prediction.append(pred)
        else:
            if allowed_strings is not None:
                allowed_token_ids = self.tokenizer([string+"\n" for string in allowed_strings]).input_ids
                allowed_token_ids = set(id for ids in allowed_token_ids for id in ids)
                logits_processors = [_get_allowed_token_ids_logits_processor(
                    frozenset(allowed_token_ids),
                    len(self.tokenizer)
                )]
            else:
                logits_processors = []
            start_time = time.time()
            outputs = self.model.generate(
                prompts=prompts,
                sampling_params=SamplingParams(
                    n=1, temperature=0.0, stop=stop_strings,
                    max_tokens=1024,
                    logits_processors=logits_processors
                ),
                use_tqdm=True,
            )
            end_time = time.time()
            prediction = [output.outputs[0].text for output in outputs]
        duration = int(end_time - start_time)
        with open(cached_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                "prompts": prompts,
                "prediction": prediction,
                "duration": duration,
            }, ensure_ascii=False, indent=4))

        return prediction, duration


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using vLLMInferencer.")
    parser.add_argument("--model_name", type=str,
                        default="Llama-3-8b",
                        help="Name of the model.")
    parser.add_argument("--task_names", type=str,
                        default="sst5-mnli-mtop-break-smcalflow",
                        help="Hyphen-separated list of tasks.")
    parser.add_argument("--method_names", type=str,
                        default="random-bm25-dense_bert-dense_bge-hybrid_bge-dense_bge_kmeans-"
                                "dense_epr-dense_epr_kmeans-dense_ceil-latent_grad",
                        help="Hyphen-separated list of method names.")
    parser.add_argument("--dependent", type=int, default=0,
                        help="Dependent ICL (instance-level demo) or independent ICL (task-level).")
    parser.add_argument("--hybrid", type=int, default=0,
                        help="ICL using task-level demons + a few instance-level demo.")
    parser.add_argument("--shuffle", type=int, default=0,
                        help="Shuffle the original order when set to 1. Reorder when set to 2.")
    parser.add_argument("--erase", type=int, default=0,
                        help="Ignore the cached results if erase.")
    parser.add_argument("--num_shots", type=str,
                        default="4-8-16-32-64-128",
                        help="Hyphen-separated list of number of shots.")
    parser.add_argument("--seeds", type=str,
                        default="42",
                        help="Hyphen-separated list of seeds.")
    parser.add_argument("--max_prompt_len", type=int,
                        default=None,
                        help="Maximum prompt length.")
    parser.add_argument("--tp", type=int,
                        default=1,
                        help="The tensor-parallel size for vLLM.")
    args = parser.parse_args()

    # Convert comma-separated strings to lists
    args.task_names = args.task_names.split('-')
    args.method_names = args.method_names.split('-')
    args.num_shots = list(map(int, args.num_shots.split('-')))
    args.seeds = list(map(int, args.seeds.split('-')))

    if args.task_names[0] in ["za2zh", "zh2za"]:
        args.num_shots = [int(_) for _ in "4 8 16 32 64 128 256 512 1024 2048".split()]

    if args.method_names == ["random"] or args.shuffle:
        args.seeds = [42, 43, 44, 45, 46]

    if args.hybrid:
        args.num_shots = [128]

    if args.max_prompt_len is None:
        if 'llama' in args.model_name.lower():
            args.max_prompt_len = 7000
        elif 'qwen' in args.model_name.lower():
            args.max_prompt_len = 130000
        else:
            raise NotImplementedError(args.model_name)

    return args


def save_results(results, args):
    # Create output directory with current datetime
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"output/inference-{current_time}"
    os.makedirs(output_dir, exist_ok=True)

    # Save results to JSON file
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    # Save args to JSON file
    args_file = os.path.join(output_dir, "args.json")
    with open(args_file, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)

    print(f"Results saved to {results_file}")
    print(f"Args saved to {args_file}")


def main():
    args = parse_args()
    inferencer = vLLMInferencer(
        model_name=args.model_name,
        task_names=args.task_names,
        method_names=args.method_names,
        dependent=args.dependent,
        hybrid=args.hybrid,
        shuffle=args.shuffle,
        erase=args.erase,
        num_shots=args.num_shots,
        seeds=args.seeds,
        max_prompt_len=args.max_prompt_len,
        tp=args.tp
    )

    # Run inference
    results = inferencer.run()

    # Save results and args
    save_results(results, args)


if __name__ == "__main__":
    main()
