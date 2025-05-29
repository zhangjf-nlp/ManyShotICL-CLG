import json
import os

from datasets import load_dataset, DatasetDict
for hf_dataset, hf_dataset_name in [
    ('SetFit/sst5', None),
    ('LysandreJik/glue-mnli-train', None),
    ('commonsense_qa', None),
    ('swag', 'regular'),
    ('iohadrubin/mtop', 'mtop'),
    ('break_data', 'QDMR'),
    ('iohadrubin/smcalflow', 'smcalflow'),
    ("swag", "regular"),
    ("src/hf_datasets/geoquery.py", 'standard'),
    ("src/hf_datasets/nl2bash.py", 'nl2bash'),
    ("src/hf_datasets/math.py", None),
    ("openai/gsm8k", "socratic"),
    ("src/hf_datasets/gpqa.py", None),
]:
    dataset = load_dataset(hf_dataset, hf_dataset_name, trust_remote_code=True)
    dataset_path = os.path.join("dataset_cache", f"{hf_dataset}-{hf_dataset_name}")
    if "train" not in dataset:
        assert "validation" in dataset and "test" in dataset
        # MMLU-Pro
        dataset["train"] = dataset.pop("test")
    elif "validation" not in dataset:
        assert "test" in dataset
        dataset["validation"] = dataset.pop("test")
    for usage in dataset:
        dataset[usage].to_json(os.path.join(dataset_path, f"{usage}.json"))
    json_dict = {
        usage: os.path.join(dataset_path, f"{usage}.json")
        for usage in ["train", "validation", "test"]
        if os.path.exists(os.path.join(dataset_path, f"{usage}.json"))
    }
    print(f"json_dict:\n{json_dict}")
    print(DatasetDict.from_json(json_dict))