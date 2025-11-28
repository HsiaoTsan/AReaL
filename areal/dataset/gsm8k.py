from typing import Optional

from datasets import load_dataset


def get_gsm8k_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
):
    dataset = load_dataset(path=path, name="main", split=split)

    def process(sample):
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["question", "answer"])

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_gsm8k_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
    level_filter: Optional[list] = None,  # Add level filter parameter
):
    # Load dataset - support both GSM8K and SimpleRL-Zoo-Data
    # Check for SimpleRL dataset (handles both HF path and local cache path)
    path_lower = path.lower()
    is_simplerl = any(x in path_lower for x in ["simplerl", "hkust-nlp", "dapo-math", "open-r1"])

    if is_simplerl:
        # SimpleRL-Zoo-Data format - handle schema differences between splits
        import os

        # Check if this is a HuggingFace cache directory (not user-prepared local files)
        is_hf_cache = os.path.isdir(path) and ("/hub/datasets--" in path or "/snapshots/" in path or "\\hub\\datasets--" in path or "\\snapshots\\" in path)

        # For local directory with schema mismatch, only load train.parquet
        # But skip if this is HuggingFace cache (let HF handle it)
        if os.path.isdir(path) and not is_hf_cache:
            train_file = os.path.join(path, "train.parquet")
            test_file = os.path.join(path, "test.parquet")

            if os.path.exists(train_file):
                print(f"Loading SimpleRL-Zoo-Data from local parquet: {train_file}")
                # Only load train file to avoid schema mismatch
                dataset = load_dataset("parquet", data_files={"train": train_file}, split="train")

                # Manually split train into train/test if test split is requested
                if split == "test":
                    print("Using last 10% of train data as test split (schema mismatch workaround)")
                    dataset = dataset.train_test_split(test_size=0.1, seed=42)["test"]
                elif split == "train":
                    # Use first 90% as train
                    dataset = dataset.train_test_split(test_size=0.1, seed=42)["train"]
            else:
                raise FileNotFoundError(f"Train file not found: {train_file}")
        else:
            # For HuggingFace Hub path or HF cache, load normally
            try:
                # If this is a HF cache directory, find parquet files
                if is_hf_cache:
                    import glob
                    parquet_files = glob.glob(os.path.join(path, "*.parquet"))
                    if parquet_files:
                        print(f"Loading from HF cache: {path}, found {len(parquet_files)} parquet files")
                        dataset = load_dataset("parquet", data_files=parquet_files, split="train")
                    else:
                        # Try the original HF path by extracting dataset name
                        # From: /data/hf_home/hub/datasets--open-r1--DAPO-Math-17k-Processed/...
                        # To: open-r1/DAPO-Math-17k-Processed
                        dataset_name = path.split("datasets--")[-1].split("/snapshots/")[0].replace("--", "/")
                        print(f"Extracted dataset name: {dataset_name}")
                        dataset = load_dataset(path=dataset_name, split=split)
                else:
                    dataset = load_dataset(path=path, split=split)
            except Exception as e:
                print(f"Warning: Failed to load {split} split, falling back to train split. Error: {e}")
                try:
                    dataset = load_dataset(path=path, split="train")
                except:
                    # Last resort: try to extract dataset name from cache path
                    if "datasets--" in path:
                        dataset_name = path.split("datasets--")[-1].split("/snapshots/")[0].split("\\snapshots\\")[0].replace("--", "/")
                        print(f"Fallback: using extracted dataset name: {dataset_name}")
                        dataset = load_dataset(path=dataset_name, split="train")
                    else:
                        raise
                if split == "test":
                    dataset = dataset.train_test_split(test_size=0.1, seed=42)["test"]
                elif split == "train":
                    dataset = dataset.train_test_split(test_size=0.1, seed=42)["train"]

        def process(sample):
            # SimpleRL-Zoo-Data format: prompt is already a list of messages
            # Extract question - handle both preprocessed and raw formats
            question = None

            # Try different field names for question
            if "question" in sample:
                question = sample["question"]
            elif "extra_info" in sample and isinstance(sample["extra_info"], dict) and "question" in sample["extra_info"]:
                question = sample["extra_info"]["question"]
            elif "source_prompt" in sample and isinstance(sample["source_prompt"], list):
                # DAPO-Math format: source_prompt is a list of messages
                question = next(
                    (msg["content"] for msg in sample["source_prompt"] if msg.get("role") == "user"),
                    None
                )
            elif "prompt" in sample and isinstance(sample["prompt"], list):
                # Extract content from first user message
                question = next(
                    (msg["content"] for msg in sample["prompt"] if msg.get("role") == "user"),
                    None
                )
            elif "prompt" in sample and isinstance(sample["prompt"], str):
                question = sample["prompt"]

            if question is None:
                question = ""

            messages = [
                {
                    "role": "user",
                    "content": question + "\nPlease put your final answer within \\boxed{}.",
                }
            ]

            # Extract ground truth answer - handle different field names
            answer = None
            if "answer" in sample:
                answer = sample["answer"]
            elif "gt_answer" in sample:
                answer = sample["gt_answer"]
            elif "reward_model" in sample and isinstance(sample["reward_model"], dict) and "ground_truth" in sample["reward_model"]:
                answer = sample["reward_model"]["ground_truth"]
            elif "extra_info" in sample and isinstance(sample["extra_info"], dict) and "answer" in sample["extra_info"]:
                answer = sample["extra_info"]["answer"]
            elif "target" in sample:
                answer = sample["target"]
            elif "solution" in sample:
                answer = sample["solution"]

            if answer is None:
                answer = ""

            # Ensure answer is in \boxed{} format for correct extraction
            # SimpleRL-Zoo-Data's ground_truth doesn't have \boxed{}, add it here
            # This is critical for math_parser.extract_answer to work correctly
            if answer and "boxed" not in answer.lower():
                answer = f"\\boxed{{{answer}}}"

            # Keep level info for filtering - handle different structures
            result = {"messages": messages, "answer": answer}
            level = None
            if "level" in sample:
                level = sample["level"]
            elif "extra_info" in sample and isinstance(sample["extra_info"], dict) and "level" in sample["extra_info"]:
                level = sample["extra_info"]["level"]

            if level is not None:
                result["level"] = level

            return result

        dataset = dataset.map(process)

        # Filter by level if specified
        if level_filter is not None:
            print(f"Filtering dataset to levels: {level_filter}")
            dataset = dataset.filter(lambda x: x.get("level") in level_filter if "level" in x else True)
            print(f"Dataset size after level filtering: {len(dataset)}")

        # Remove unnecessary columns if they exist
        cols_to_remove = [col for col in ["prompt", "source_prompt", "solution", "target", "reward_model", "extra_info", "ability", "data_source", "level"] if col in dataset.column_names]
        if cols_to_remove:
            dataset = dataset.remove_columns(cols_to_remove)
    else:
        # Original GSM8K format
        dataset = load_dataset(path=path, name="main", split=split)

        def process(sample):
            messages = [
                {
                    "role": "user",
                    "content": sample["question"]
                    + "\nPlease put your final answer within \\boxed{}.",
                }
            ]
            return {"messages": messages, "answer": sample["answer"]}

        dataset = dataset.map(process).remove_columns(["question", "answer"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
