import json
import logging
import os
from collections import defaultdict, Counter
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import set_seed

from src.dataset_readers.dataset_wrappers import get_dataset_wrapper
from src.dataset_readers.dataset_wrappers.base_dsw import ABC
from src.utils.misc import parallel_run

logger = logging.getLogger(__name__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_shared_ctxs_path(model_name, task_name, method, num_ice, seed):
    if method in [
        "random", "bm25", "dense_bert", "dense_bge", "hybrid_bge", "dense_bge_kmeans",
        "dense_epr", "dense_epr_kmeans", "dense_ceil"
    ]:
        model_name = "Llama-3-8b"
    return (f"output/ctxs/{model_name}-{task_name}-{method}"
            f"/{task_name}-{num_ice}shots-{seed}seed.json")


def get_dependent_ctxs_path(model_name, task_name, method, num_ice, seed):
    if method in [
        "random", "bm25", "dense_bert", "dense_bge", "hybrid_bge", "dense_bge_kmeans",
        "dense_epr", "dense_epr_kmeans", "dense_ceil"
    ]:
        model_name = "Llama-3-8b"
    return (f"output/ctxs_dp/{model_name}-{task_name}-{method}"
            f"/{task_name}-{num_ice}shots-{seed}seed.json")


def numpy_to_cuda_tensor_and_batch_matmul(embeddings_a, embeddings_b, batch_size=8):
    if isinstance(embeddings_a, np.ndarray):
        embeddings_a = torch.from_numpy(embeddings_a).float().cuda()
    if isinstance(embeddings_b, np.ndarray):
        embeddings_b = torch.from_numpy(embeddings_b).float().cuda()
    scores = []
    for index in range(0, embeddings_a.shape[0], batch_size):
        scores.append(embeddings_a[index:index+batch_size, :] @ embeddings_b.T)
    return torch.cat(scores, dim=0).cpu().numpy()


N_DEPENDENT_TEST_CASES = 10000

class BaseFinder:
    method = None
    def __init__(self, task_name, model_name):
        self.task_name = task_name
        self.model_name = model_name # the base model, may be not the inference model
        self.index_dataset_wrapper: ABC = get_dataset_wrapper(
            task_name,
            dataset_split="train",
            ds_size=50000
        )

    def find_and_save(self, num_ice, seed=42, erase=False):
        output_file = get_shared_ctxs_path(
            model_name=self.model_name,
            task_name=self.task_name,
            method=self.method,
            num_ice=num_ice,
            seed=seed
        )
        if os.path.exists(output_file) and not erase:
            return
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with torch.no_grad():
            ctxs = self.find_shared_contexts(num_ice, seed=seed)
        assert [i in range(len(self.index_dataset_wrapper)) for i in ctxs]
        logger.info(f"{self.__class__.__name__} found for {self.task_name}:\n{ctxs}")
        qa_corpus = self.index_dataset_wrapper.get_corpus(field="qa")
        ctxs = [qa_corpus[int(i)] for i in ctxs]
        logger.info(f"ctxs[0]: {ctxs[0]}")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(json.dumps(ctxs, ensure_ascii=False, indent=4))
        return

    def find_and_save_dp(self, num_ice, seed=42, erase=False):
        output_file = get_dependent_ctxs_path(
            model_name=self.model_name,
            task_name=self.task_name,
            method=self.method,
            num_ice=num_ice,
            seed=seed
        )
        if os.path.exists(output_file) and not erase:
            return
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with torch.no_grad():
            ctxs_dp, test_indexes = self.find_dependent_contexts(num_ice, seed=seed)
            assert len(ctxs_dp) == len(test_indexes), f"{len(ctxs_dp)} != {len(test_indexes)}"
        assert [i in range(len(self.index_dataset_wrapper)) for i in ctxs_dp]
        logger.info(f"{self.__class__.__name__} found for {self.task_name}:\n{np.array(ctxs_dp)}")
        qa_corpus = self.index_dataset_wrapper.get_corpus(field="qa")
        ctxs_dp = [[qa_corpus[int(i)] for i in ctxs] for ctxs in ctxs_dp]
        logger.info(f"ctxs_dp[0]: {ctxs_dp[0]}")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(json.dumps({"ctxs_dp": ctxs_dp, "test_index": test_indexes}, ensure_ascii=False, indent=4))
        return

    def find_shared_contexts(self, num_ice, seed) -> List[int]:
        raise NotImplementedError

    def find_dependent_contexts(self, num_ice, seed) -> Tuple[List[List[int]], List[int]]:
        raise NotImplementedError


class RandomFinder(BaseFinder):
    method = "random"
    def find_shared_contexts(self, num_ice, seed) -> List[int]:
        set_seed(seed)
        return np.random.permutation(len(self.index_dataset_wrapper))[:num_ice].tolist()

    def find_dependent_contexts(self, num_ice, seed) -> Tuple[List[List[int]], List[int]]:
        set_seed(seed)
        query_dataset_wrapper: ABC = get_dataset_wrapper(
            self.task_name,
            dataset_split="validation",
            ds_size=N_DEPENDENT_TEST_CASES
        )
        return [
            np.random.permutation(len(self.index_dataset_wrapper))[:num_ice].tolist()
            for _ in query_dataset_wrapper
        ], list(range(len(query_dataset_wrapper)))


class BM25Finder(BaseFinder):
    method = "bm25"
    def find_shared_contexts(self, num_ice, seed) -> List[int]:
        set_seed(seed)
        if not hasattr(self, "all_scores"):
            from nltk.tokenize import word_tokenize
            index_corpus = [word_tokenize(qa) for qa in self.index_dataset_wrapper.get_corpus(field="qa")]
            import multiprocessing
            from multiprocessing import Pool
            num_processes = multiprocessing.cpu_count()
            chunk_size = len(index_corpus) // num_processes + 1
            chunks = [index_corpus[i:i + chunk_size] for i in range(0, len(index_corpus), chunk_size)]
            with Pool(processes=num_processes) as pool:
                results = pool.starmap(BM25Finder.calculate_scores, zip([index_corpus]*len(chunks), chunks))
            self.all_scores = np.concatenate(results, axis=0)
        retrieval_scores = np.mean(self.all_scores, axis=0)
        return np.argsort(retrieval_scores)[::-1][:num_ice].tolist()

    @staticmethod
    def calculate_scores(index_corpus, chunk):
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(index_corpus)
        chunk_scores = []
        for qa in chunk:
            chunk_scores.append(bm25.get_scores(qa))
        return np.stack(chunk_scores, axis=0)

    def find_dependent_contexts(self, num_ice, seed) -> Tuple[List[List[int]], List[int]]:
        set_seed(seed)
        query_dataset_wrapper: ABC = get_dataset_wrapper(
            self.task_name,
            dataset_split="validation",
            ds_size=N_DEPENDENT_TEST_CASES
        )
        if not hasattr(self, "all_scores_dp"):
            from nltk.tokenize import word_tokenize
            index_corpus = [word_tokenize(qa) for qa in self.index_dataset_wrapper.get_corpus(field="q")]
            query_corpus = [word_tokenize(qa) for qa in query_dataset_wrapper.get_corpus(field="q")]
            import multiprocessing
            from multiprocessing import Pool
            num_processes = multiprocessing.cpu_count()
            chunk_size = len(query_corpus) // num_processes + 1
            chunks = [query_corpus[i:i + chunk_size] for i in range(0, len(query_corpus), chunk_size)]
            with Pool(processes=num_processes) as pool:
                results = pool.starmap(BM25Finder.calculate_scores, zip([index_corpus]*len(chunks), chunks))
            self.all_scores_dp = np.concatenate(results, axis=0)
            assert self.all_scores_dp.shape == (len(query_corpus), len(index_corpus))
        sorted_indices = np.argsort(-self.all_scores_dp, axis=1)
        return sorted_indices[:, :num_ice].tolist(), list(range(len(query_dataset_wrapper)))


class BestOf5Finder(BaseFinder):
    method = "best_of_5"
    n = 5
    def find_shared_contexts(self, num_ice, seed) -> List[int]:
        set_seed(seed)
        from inferencer_fast import vLLMInferencer
        from src.metrics import get_metric
        if not hasattr(self, "inferencer"):
            if 'llama' in self.model_name.lower():
                max_prompt_len = 7000
            elif 'qwen' in self.model_name.lower():
                max_prompt_len = 130000
            else:
                raise NotImplementedError(self.model_name)
            self.inferencer = vLLMInferencer(
                model_name=self.model_name,
                task_names=self.task_name,
                method_names=[], dependent=False, hybrid=False, shuffle=False,
                num_shots=[], seeds=[],
                max_prompt_len=max_prompt_len, tp=1, erase=False,
            )
        ctxs_samples = [
            np.random.permutation(len(self.index_dataset_wrapper))[:num_ice].tolist()
            for _ in range(self.n)
        ]
        all_prompts, all_reference = [], []
        for ctxs_sample in tqdm(ctxs_samples, desc="calling prepare_prompts_for_scoring"):
            prompts, reference = self.prepare_prompts_for_scoring(ctxs_sample)
            all_prompts.append(prompts)
            all_reference.append(reference)
        split_length = len(all_prompts[0])
        assert all(len(_) == split_length for _ in all_prompts)
        assert all(len(_) == split_length for _ in all_reference)
        all_prompts = [_ for __ in all_prompts for _ in __]
        all_reference = [_ for __ in all_reference for _ in __]
        inference_dataset_wrapper = self.index_dataset_wrapper
        if 'choices' in inference_dataset_wrapper.field_getter:
            choices = inference_dataset_wrapper.get_field(inference_dataset_wrapper[0], "choices")
        else:
            choices = None
        prediction, duration = self.inferencer.inference(
            prompts=all_prompts, choices=choices,
            stop_strings=inference_dataset_wrapper.stop_strings,
            allowed_strings=inference_dataset_wrapper.allowed_strings,
            cached_file="null_cache_file.json", record_file="null_record_file.json", num_shot=None,
        )
        evaluator = get_metric(self.task_name)
        ctxs_scores = []
        for index in tqdm(range(0, len(prediction), split_length), desc="evaluating"):
            metric = evaluator.evaluate(prediction[index:index+split_length], all_reference[index:index+split_length])
            ctxs_scores.append(list(metric.values())[0])
        return ctxs_samples[np.argmax(ctxs_scores)]

    def prepare_prompts_for_scoring(self, ctxs_sample):
        from inferencer_fast import stack_demos
        inference_dataset_wrapper = self.index_dataset_wrapper
        ice_separator = inference_dataset_wrapper.ice_separator
        a_prefix = inference_dataset_wrapper.a_prefix
        queries = inference_dataset_wrapper.get_corpus("gen_a")
        queries = [query + a_prefix for query in queries]
        sentences = inference_dataset_wrapper.get_corpus("qa")
        ctxs = [sentences[i] for i in ctxs_sample] # List[int] -> List[str]
        reference = self.index_dataset_wrapper.dataset
        queries = [query for i, query in enumerate(queries) if i not in ctxs_sample][:1000]
        reference = [refer for i, refer in enumerate(reference) if i not in ctxs_sample][:1000]
        prompts, prompt_lengths = [], []
        for query in queries:
            query_length = len(self.inferencer.tokenizer(query).input_ids)
            prompt, prompt_length = stack_demos(
                ctxs=ctxs, ice_separator=ice_separator,
                tokenizer=self.inferencer.tokenizer, max_prompt_len=self.inferencer.max_prompt_len - query_length
            )
            prompts.append(query.replace("{ice_prompt}", prompt))
            prompt_lengths.append(prompt_length)
        print(f"avg prompt_length: {np.mean(prompt_lengths):.2f} tokens")
        return prompts, reference


class DenseBertFinder(BaseFinder):
    method = "dense_bert"
    def find_shared_contexts(self, num_ice, seed, batch_size=8) -> List[int]:
        set_seed(seed)
        if not hasattr(self, "all_scores"):
            sentences = self.index_dataset_wrapper.get_corpus(field="qa")
            from transformers import BertModel, AutoTokenizer
            bert = BertModel.from_pretrained('bert-base-uncased', device_map=device)
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            tokenizer.padding_side = "right"
            embeddings = []
            for i in tqdm(range(0, len(sentences), batch_size), desc="encode sentences"):
                inputs = tokenizer(sentences[i:i+batch_size], return_tensors="pt",
                                   padding="longest", max_length=512, truncation=True)
                outputs = bert(**inputs.to(device))[0]
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                batch_embeddings = torch.sum(outputs * attention_mask, dim=1) / (torch.sum(attention_mask, dim=1) + 1e-8)
                embeddings.append(batch_embeddings)
            embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
            self.all_scores = numpy_to_cuda_tensor_and_batch_matmul(embeddings, embeddings)
        retrieval_scores = np.mean(self.all_scores, axis=1)
        return np.argsort(retrieval_scores)[::-1][:num_ice].tolist()

    def find_dependent_contexts(self, num_ice, seed, batch_size=8) -> Tuple[List[List[int]], List[int]]:
        set_seed(seed)
        query_dataset_wrapper: ABC = get_dataset_wrapper(
            self.task_name,
            dataset_split="validation",
            ds_size=N_DEPENDENT_TEST_CASES
        )
        if not hasattr(self, "all_scores_dp"):
            sentences = self.index_dataset_wrapper.get_corpus(field="q") + query_dataset_wrapper.get_corpus(field="q")
            assert len(sentences) == len(self.index_dataset_wrapper) + len(query_dataset_wrapper)
            from transformers import BertModel, AutoTokenizer
            bert = BertModel.from_pretrained('bert-base-uncased', device_map=device)
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            tokenizer.padding_side = "right"
            embeddings = []
            for i in tqdm(range(0, len(sentences), batch_size), desc="encode sentences"):
                inputs = tokenizer(sentences[i:i+batch_size], return_tensors="pt",
                                   padding="longest", max_length=512, truncation=True)
                outputs = bert(**inputs.to(device))[0]
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                batch_embeddings = torch.sum(outputs * attention_mask, dim=1) / (torch.sum(attention_mask, dim=1) + 1e-8)
                embeddings.append(batch_embeddings)
            embeddings = torch.cat(embeddings, dim=0)
            assert embeddings.shape[0] == len(sentences)
            self.all_scores_dp = numpy_to_cuda_tensor_and_batch_matmul(
                embeddings[len(self.index_dataset_wrapper):, :],
                embeddings[:len(self.index_dataset_wrapper), :]
            )
        sorted_indices = np.argsort(-self.all_scores_dp, axis=1)
        return sorted_indices[:, :num_ice].tolist(), list(range(len(query_dataset_wrapper)))


class DenseBGEFinder(BaseFinder):
    method = "dense_bge"
    def find_shared_contexts(self, num_ice, seed) -> List[int]:
        set_seed(seed)
        if not hasattr(self, "all_scores"):
            sentences = self.index_dataset_wrapper.get_corpus(field="qa")
            from FlagEmbedding import BGEM3FlagModel
            model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices=device)
            embeddings = model.encode(sentences, return_dense=True, batch_size=8, max_length=512)["dense_vecs"]
            self.all_scores = numpy_to_cuda_tensor_and_batch_matmul(embeddings, embeddings)
        retrieval_scores = np.mean(self.all_scores, axis=1)
        return np.argsort(retrieval_scores)[::-1][:num_ice].tolist()

    def find_dependent_contexts(self, num_ice, seed) -> Tuple[List[List[int]], List[int]]:
        set_seed(seed)
        query_dataset_wrapper: ABC = get_dataset_wrapper(
            self.task_name,
            dataset_split="validation",
            ds_size=N_DEPENDENT_TEST_CASES
        )
        if not hasattr(self, "all_scores_dp"):
            sentences = self.index_dataset_wrapper.get_corpus(field="q") + query_dataset_wrapper.get_corpus(field="q")
            assert len(sentences) == len(self.index_dataset_wrapper) + len(query_dataset_wrapper)
            from FlagEmbedding import BGEM3FlagModel
            model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices=device)
            embeddings = model.encode(sentences, return_dense=True, batch_size=8, max_length=512)["dense_vecs"]
            assert embeddings.shape[0] == len(sentences)
            self.all_scores_dp = numpy_to_cuda_tensor_and_batch_matmul(
                embeddings[len(self.index_dataset_wrapper):, :],
                embeddings[:len(self.index_dataset_wrapper), :]
            )
        sorted_indices = np.argsort(-self.all_scores_dp, axis=1)
        return sorted_indices[:, :num_ice].tolist(), list(range(len(query_dataset_wrapper)))


class HybridBGEFinder(BaseFinder):
    method = "hybrid_bge"
    def find_shared_contexts(self, num_ice, seed) -> List[int]:
        set_seed(seed)
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, batch_size=32, devices=device)
        sentences = self.index_dataset_wrapper.get_corpus(field="qa")
        N = len(self.index_dataset_wrapper)
        query_indices = np.random.permutation(N)[:num_ice]
        selected_indices = []
        for i in tqdm(query_indices, desc="hybrid scoring through BEG-M3"):
            candidate_indices = [j for j in range(N) if i!=j and j not in selected_indices]
            sentence_pairs = [(sentences[i], sentences[j]) for j in candidate_indices]
            scores = model.compute_score(
                sentence_pairs,
                max_passage_length=128,
                weights_for_different_modes=[0.4, 0.2, 0.4],
            ) # set according to https://huggingface.co/BAAI/bge-m3
            scores = scores["colbert+sparse+dense"]
            selected_indices.append(candidate_indices[np.argmax(scores)])
        return selected_indices

    def find_dependent_contexts(self, num_ice, seed) -> Tuple[List[List[int]], List[int]]:
        set_seed(seed)
        query_dataset_wrapper: ABC = get_dataset_wrapper(
            self.task_name,
            dataset_split="validation",
            ds_size=N_DEPENDENT_TEST_CASES
        )
        if not hasattr(self, "all_scores_dp"):
            from FlagEmbedding import BGEM3FlagModel
            model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, batch_size=32, devices=device)
            index_sentences = self.index_dataset_wrapper.get_corpus(field="q")
            query_sentences = query_dataset_wrapper.get_corpus(field="q")
            all_scores_dp = []
            for query_sentence in tqdm(query_sentences, desc="hybrid scoring through BEG-M3"):
                sentence_pairs = [(query_sentence, index_sentence) for index_sentence in index_sentences]
                scores = model.compute_score(
                    sentence_pairs,
                    max_passage_length=128,
                    weights_for_different_modes=[0.4, 0.2, 0.4],
                )  # set according to https://huggingface.co/BAAI/bge-m3
                scores = scores["colbert+sparse+dense"]
                all_scores_dp.append(scores)
            self.all_scores_dp = np.array(all_scores_dp)
            assert self.all_scores_dp.shape == (len(query_sentences), len(index_sentences))
        sorted_indices = np.argsort(-self.all_scores_dp, axis=1)
        return sorted_indices[:, :num_ice].tolist(), list(range(len(query_dataset_wrapper)))


class DenseBGEKmeansFinder(BaseFinder):
    method = "dense_bge_kmeans"
    def find_shared_contexts(self, num_ice, seed) -> List[int]:
        set_seed(seed)
        if not hasattr(self, "all_embeddings"):
            sentences = self.index_dataset_wrapper.get_corpus(field="qa")
            from FlagEmbedding import BGEM3FlagModel
            model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices=device)
            self.all_embeddings: np.array = model.encode(sentences, return_dense=True, batch_size=8, max_length=512)["dense_vecs"]
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_ice, random_state=seed)
        kmeans.fit(self.all_embeddings)
        selected_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(self.all_embeddings - center, axis=1)
            selected_indices.append(np.argmin(distances))
        return selected_indices


class DenseEPRFinder(BaseFinder):
    method = "dense_epr"
    def find_shared_contexts(self, num_ice, seed, batch_size=8) -> List[int]:
        set_seed(seed)
        if not hasattr(self, "all_embeddings"):
            questions = self.index_dataset_wrapper.get_corpus(field="q")
            from transformers import AutoTokenizer
            from src.models.biencoder import BiEncoder
            epr_model_path = f"output/epr/{self.task_name}/Llama-3-8b/bert-fix_ctx-shared-bs64"
            model = BiEncoder.from_pretrained(epr_model_path, device_map=device)
            tokenizer = AutoTokenizer.from_pretrained(epr_model_path)
            all_embeddings = []
            for i in tqdm(range(0, len(questions), batch_size), desc="encode questions"):
                inputs = tokenizer(questions[i:i+batch_size], return_tensors='pt',
                                   padding="longest", max_length=512, truncation=True)
                batch_embeddings = model.encode(**inputs.to(device))
                all_embeddings.append(batch_embeddings)
            self.all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
        all_scores = numpy_to_cuda_tensor_and_batch_matmul(self.all_embeddings, self.all_embeddings)
        retrieval_scores = np.mean(all_scores, axis=1)
        return np.argsort(retrieval_scores)[::-1][:num_ice].tolist()

    def find_dependent_contexts(self, num_ice, seed, batch_size=8) -> Tuple[List[List[int]], List[int]]:
        set_seed(seed)
        query_dataset_wrapper: ABC = get_dataset_wrapper(
            self.task_name,
            dataset_split="validation",
            ds_size=N_DEPENDENT_TEST_CASES
        )
        if not hasattr(self, "all_scores_dp"):
            questions = self.index_dataset_wrapper.get_corpus(field="q") + query_dataset_wrapper.get_corpus(field="q")
            assert len(questions) == len(self.index_dataset_wrapper) + len(query_dataset_wrapper)
            from transformers import AutoTokenizer
            from src.models.biencoder import BiEncoder
            epr_model_path = f"output/epr/{self.task_name}/Llama-3-8b/bert-fix_ctx-shared-bs64"
            model = BiEncoder.from_pretrained(epr_model_path, device_map=device)
            tokenizer = AutoTokenizer.from_pretrained(epr_model_path)
            all_embeddings = []
            for i in tqdm(range(0, len(questions), batch_size), desc="encode questions"):
                inputs = tokenizer(questions[i:i+batch_size], return_tensors='pt',
                                   padding="longest", max_length=512, truncation=True)
                batch_embeddings = model.encode(**inputs.to(device))
                all_embeddings.append(batch_embeddings)
            all_embeddings = torch.cat(all_embeddings, dim=0)
            assert all_embeddings.shape[0] == len(questions)
            self.all_scores_dp = numpy_to_cuda_tensor_and_batch_matmul(
                all_embeddings[len(self.index_dataset_wrapper):, :],
                all_embeddings[:len(self.index_dataset_wrapper), :],
            )
        sorted_indices = np.argsort(-self.all_scores_dp, axis=1)
        return sorted_indices[:, :num_ice].tolist(), list(range(len(query_dataset_wrapper)))



class DenseEPRKmeansFinder(BaseFinder):
    method = "dense_epr_kmeans"
    def find_shared_contexts(self, num_ice, seed, batch_size=8) -> List[int]:
        set_seed(seed)
        if not hasattr(self, "all_embeddings"):
            questions = self.index_dataset_wrapper.get_corpus(field="q")
            from transformers import AutoTokenizer
            from src.models.biencoder import BiEncoder
            epr_model_path = f"output/epr/{self.task_name}/Llama-3-8b/bert-fix_ctx-shared-bs64"
            model = BiEncoder.from_pretrained(epr_model_path, device_map=device)
            tokenizer = AutoTokenizer.from_pretrained(epr_model_path)
            all_embeddings = []
            for i in tqdm(range(0, len(questions), batch_size), desc="encode questions"):
                inputs = tokenizer(questions[i:i+batch_size], return_tensors='pt',
                                   padding="longest", max_length=512, truncation=True)
                batch_embeddings = model.encode(**inputs.to(device))
                all_embeddings.append(batch_embeddings)
            self.all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_ice, random_state=seed)
        kmeans.fit(self.all_embeddings)
        selected_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(self.all_embeddings - center, axis=1)
            selected_indices.append(np.argmin(distances))
        return selected_indices


class DenseCEILFinder(BaseFinder):
    method = "dense_ceil" # [N, 128] -> 128, majority voting
    def find_shared_contexts(self, num_ice, seed, dpp_topk=100, num_candidates=1, scale_factor=0.1, batch_size=8) -> List[int]:
        set_seed(seed)
        questions = self.index_dataset_wrapper.get_corpus(field="q")
        if not hasattr(self, "all_embeddings"):
            from transformers import AutoTokenizer
            from src.models.biencoder import BiEncoder, BiEncoderConfig
            ceil_model_path = f"output/dpp-epr-random/{self.task_name}/Llama-3-8b/base-mg0.02-s0.1-fix"
            ceil_config = BiEncoderConfig(
                norm_embed=True, scale_factor=scale_factor,
                q_model_name="bert-base-uncased",
                ctx_model_name="bert-base-uncased"
            )
            model = BiEncoder.from_pretrained(ceil_model_path, config=ceil_config, device_map=device)
            tokenizer = AutoTokenizer.from_pretrained(ceil_model_path)
            all_embeddings, all_ctx_embeddings = [], []
            for i in tqdm(range(0, len(questions), batch_size), desc="encode questions"):
                inputs = tokenizer(questions[i:i+batch_size], return_tensors='pt',
                                   padding="longest", max_length=512, truncation=True)
                all_embeddings.append(model.encode(**inputs.to(device)))
                all_ctx_embeddings.append(model.encode(**inputs.to(device), encode_ctx=True))
            self.all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
            self.all_ctx_embeddings = torch.cat(all_ctx_embeddings, dim=0).cpu().numpy()
        import faiss
        index_global = faiss.IndexIDMap(faiss.IndexFlatIP(self.all_ctx_embeddings.shape[-1]))
        index_global.add_with_ids(self.all_ctx_embeddings, np.arange(self.all_ctx_embeddings.shape[0]))
        from dense_retriever import dpp, partial, set_global_object
        func = partial(dpp, num_candidates=num_candidates, num_ice=num_ice,
                       mode="map", dpp_topk=dpp_topk, scale_factor=scale_factor)
        res_list = [{"embed": embed, "entry": self.index_dataset_wrapper[i]}
                    for i, embed in enumerate(self.all_embeddings)]

        results = parallel_run(func=func, args_list=res_list, initializer=set_global_object,
                               initargs=(index_global, True))
        all_ctxs = [ctx for entry in results for ctx in entry['ctxs']]
        counter = Counter(all_ctxs)
        elements = list(counter.keys())
        frequencies = list(counter.values())
        probabilities = np.array(frequencies) / sum(frequencies)
        sampled_shared_ctxs = np.random.choice(elements, size=num_ice, replace=False, p=probabilities)
        return sampled_shared_ctxs


class LatentPosteriorFinder(BaseFinder):
    method = "latent_post"
    def __init__(self, task_name, model_name):
        super().__init__(task_name, model_name)
        from transformers import AutoTokenizer
        from train_prefix import auto_model_class, parse_args, get_last_checkpoint
        args = parse_args(["--task_name", task_name, "--model_name", model_name,
                           "--num_p", "4", "--learning_rate", "1e-3"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = len(self.tokenizer) - 1
        self.model = auto_model_class(args).from_pretrained(get_last_checkpoint(args.output_dir), device_map=device)

    def find_shared_contexts(self, num_ice, seed, batch_size=8) -> List[int]:
        set_seed(seed)
        sentences = self.index_dataset_wrapper.get_corpus(field="qa")
        if not hasattr(self, "prefix_posterior_nll"):
            prefix_posterior_nll = []
            for i in tqdm(range(0, len(sentences), batch_size), desc="calculating prefix posterior probabilities"):
                batch_sentences = sentences[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_sentences, return_tensors="pt",
                    padding="longest", max_length=512, truncation=True
                )
                inputs = inputs.to(device)
                prefix_embeds = self.model.get_prefix_embeds(batch_size=len(batch_sentences))
                input_embeds = self.model.model.embed_tokens(inputs["input_ids"]).detach()
                inputs_embeds = torch.cat([input_embeds, prefix_embeds], dim=1)
                prefix_attention_mask = inputs["attention_mask"].new_ones(size=(len(batch_sentences), self.model.num_p))
                attention_mask = torch.cat([inputs["attention_mask"], prefix_attention_mask], dim=1)
                if i == 0:
                    print(f"input_ids: {inputs['input_ids']}")
                    print(f"attention_mask: {attention_mask}")
                transformer_outputs = self.model.model.forward(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                hidden_states = transformer_outputs.last_hidden_state
                lm_logits = self.model.lm_head(hidden_states)
                prefix_logits = torch.einsum("bse,bpe->bsp", hidden_states, prefix_embeds)
                concat_logits = torch.cat([lm_logits, prefix_logits], dim=-1)
                concat_labels = (torch.arange(self.model.num_p) + lm_logits.shape[1]).unsqueeze(0).repeat(len(batch_sentences), 1)
                batch_prefix_posterior_nll = torch.nn.functional.cross_entropy(
                    input=concat_logits[:, -self.model.num_p-1:-1, :].contiguous().transpose(1, 2),
                    target=concat_labels.to(self.model.device).long(),
                    reduction='none'
                ).sum(dim=-1)
                prefix_posterior_nll.append(batch_prefix_posterior_nll.cpu().numpy())
            self.prefix_posterior_nll = np.concatenate(prefix_posterior_nll, axis=0)
        assert self.prefix_posterior_nll.shape == (len(sentences),), (self.prefix_posterior_nll.shape, len(sentences))
        return np.argsort(self.prefix_posterior_nll)[:num_ice].tolist()


class LatentGradFinder(BaseFinder):
    method = "latent_grad"
    def find_shared_contexts(self, num_ice, seed, first_n=None, last_n=None, reverse=False) -> List[int]:
        # first_n / last_n: only use the first / last n checkpoints for gradient matching
        set_seed(seed)
        from condense_tensor import find_subset_for_task
        from train_prefix import DatasetForCVAE
        train_dataset = DatasetForCVAE(task_name=self.task_name, usage="train", model_name=self.model_name)
        indices = find_subset_for_task(
            task_name=self.task_name, model_name=self.model_name,
            n=num_ice, train_dataset=train_dataset,
            first_n=first_n, last_n=last_n, reverse=reverse,
        )
        # some datapoints may be excluded from training and gradient matching for exceeding in length
        indexes = [train_dataset[indice]["index"] for indice in indices]
        logger.info("-------------check consistency-------------")
        for indice, index in zip(indices, indexes):
            logger.info(f"demo ({index}): {train_dataset.tokenizer.decode(train_dataset[indice]['posterior_ids'])}")
            logger.info(f"demo ({index}): {self.index_dataset_wrapper[index]}")
        logger.info("-------------check consistency-------------")
        return indexes

    def find_dependent_contexts(self, num_ice, seed) -> Tuple[List[List[int]], List[int]]:
        set_seed(seed)
        from condense_tensor import find_subset_for_cases
        from train_prefix import DatasetForCVAE
        train_dataset = DatasetForCVAE(task_name=self.task_name, usage="train", model_name=self.model_name)
        valid_dataset = DatasetForCVAE(task_name=self.task_name, usage="validation", model_name=self.model_name)
        all_indices = find_subset_for_cases(
            task_name=self.task_name, model_name=self.model_name,
            n=num_ice, train_dataset=train_dataset, valid_dataset=valid_dataset,
            n_test_caces=N_DEPENDENT_TEST_CASES,
            first_n=None, last_n=None,
        )
        # some datapoints may be excluded from training and gradient matching for exceeding in length
        all_indexes = []
        for indices in all_indices:
            indexes = [train_dataset[indice]["index"] for indice in indices]
            logger.info("-------------check consistency-------------")
            for indice, index in zip(indices, indexes):
                logger.info(f"demo ({index}): {train_dataset.tokenizer.decode(train_dataset[indice]['posterior_ids'])}")
                logger.info(f"demo ({index}): {self.index_dataset_wrapper[index]}")
            logger.info("-------------check consistency-------------")
            all_indexes.append(indexes)
        test_indexes = [valid_dataset[i]["index"] for i in range(len(all_indices))]
        return all_indexes, test_indexes


class LatentGradFirst1Finder(LatentGradFinder):
    method = "latent_grad_first_1"
    def find_shared_contexts(self, num_ice, seed, first_n=None, last_n=None, reverse=False) -> List[int]:
        return super().find_shared_contexts(num_ice, seed=seed, first_n=1)

class LatentGradFirst2Finder(LatentGradFinder):
    method = "latent_grad_first_2"
    def find_shared_contexts(self, num_ice, seed, first_n=None, last_n=None, reverse=False) -> List[int]:
        return super().find_shared_contexts(num_ice, seed=seed, first_n=2)

class LatentGradFirst4Finder(LatentGradFinder):
    method = "latent_grad_first_4"
    def find_shared_contexts(self, num_ice, seed, first_n=None, last_n=None, reverse=False) -> List[int]:
        return super().find_shared_contexts(num_ice, seed=seed, first_n=4)

class LatentGradFirst8Finder(LatentGradFinder):
    method = "latent_grad_first_8"
    def find_shared_contexts(self, num_ice, seed, first_n=None, last_n=None, reverse=False) -> List[int]:
        return super().find_shared_contexts(num_ice, seed=seed, first_n=8)

class LatentGradReverseFinder(LatentGradFinder):
    method = "latent_grad_reverse"
    def find_shared_contexts(self, num_ice, seed, first_n=None, last_n=None, reverse=False) -> List[int]:
        return super().find_shared_contexts(num_ice, seed=seed, reverse=True)


finder_classes = [
    RandomFinder,
    BM25Finder,
    BestOf5Finder,
    DenseBertFinder,
    DenseBGEFinder,
    HybridBGEFinder,
    DenseBGEKmeansFinder,
    DenseEPRFinder,
    DenseEPRKmeansFinder,
    DenseCEILFinder,
    LatentPosteriorFinder,
    LatentGradFinder,
    LatentGradFirst1Finder,
    LatentGradFirst2Finder,
    LatentGradFirst4Finder,
    LatentGradFirst8Finder,
    LatentGradReverseFinder,
]
method_to_finder = {
    finder_class.method: finder_class for finder_class in finder_classes
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run different finders to select in-context examples.")
    parser.add_argument("--method", type=str, required=True,
                        choices=list(method_to_finder.keys()),
                        help="The method to use for finding in-context examples.")
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["sst5", "mnli", "mtop", "break", "smcalflow",
                                 "cmsqa", "swag", "geoquery", "nl2bash", "math", "gsm8k", "gpqa"],
                        help="The name of the task to run the finder on.")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["Llama-3-8b", "Qwen2.5-0.5B", "Qwen2.5-3B"],
                        help="The base language model to use if needed.")
    parser.add_argument("--num_ice", type=int, default=None,
                        help="The number of in-context examples to find. If None, it will iterate over [4, 8, 16, 32, 64, 128].")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Hyphen-separated list of random seeds.")
    parser.add_argument("--erase", action="store_true",
                        help="Ignore and erase existing outputs in this run.")
    parser.add_argument("--dependent", type=int, default=0,
                        help="Find the test-case-dependent in-context demonstrations.")

    args = parser.parse_args()

    # Get the appropriate Finder class based on the method
    FinderClass = method_to_finder[args.method]

    # Initialize the Finder
    finder = FinderClass(task_name=args.task_name, model_name=args.model_name)

    # If num_ice is None, iterate over the list [4, 8, 16, 32, 64, 128]
    if args.num_ice is None:
        num_ice_list = [4, 8, 16, 32, 64, 128]
    else:
        num_ice_list = [args.num_ice]

    if args.method == "random":
        args.seeds = "42-43-44-45-46"

    # Run the find_and_save method for each num_ice value
    for num_ice in num_ice_list:
        for i, seed in enumerate(list(map(int, args.seeds.split('-')))):
            if args.dependent:
                finder.find_and_save_dp(num_ice=num_ice, seed=seed, erase=args.erase)
            else:
                finder.find_and_save(num_ice=num_ice, seed=seed, erase=args.erase)

if __name__ == "__main__":
    main()