import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Tokenize and concatenate text dataset for pre-training.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset from Hugging Face Hub (e.g., wikitext)")
    parser.add_argument("--dataset_config", type=str, default=None, help="Configuration name for the dataset (e.g., wikitext-103-raw-v1)")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name of the tokenizer (e.g., meta-llama/Meta-Llama-3-8B)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the tokenized data (.npy)")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column in the dataset")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes for dataset mapping")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process (e.g., train, validation)")

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Ensure EOS token is added if tokenizer doesn't do it by default
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'}) # Or use a different token if needed
        print("Added EOS token to tokenizer.")

    print(f"Loading dataset: {args.dataset_name} ({args.dataset_config}), split: {args.split}")
    # Use streaming=True for potentially very large datasets during initial loading,
    # but mapping might still require substantial memory depending on implementation.
    # For moderate datasets like wikitext, streaming=False is fine.
    raw_dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split, streaming=False)

    def tokenize_function(examples):
        # Tokenize texts and add EOS token to each document
        tokenized_outputs = tokenizer(
            examples[args.text_column],
            add_special_tokens=False, # We handle EOS manually below if needed per doc
            truncation=False # Don't truncate here, handle sequence length in training dataloader
        )

        # Add EOS token ID to the end of each document's token list
        eos_token_id = tokenizer.eos_token_id
        for i in range(len(tokenized_outputs["input_ids"])):
             tokenized_outputs["input_ids"][i].append(eos_token_id)
             if "attention_mask" in tokenized_outputs: # Add mask for EOS if needed
                 tokenized_outputs["attention_mask"][i].append(1)

        return tokenized_outputs

    print("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=raw_dataset.column_names # Remove original text columns
    )

    print("Concatenating token IDs...")
    # Concatenate all token IDs into a single list/array
    # This might be memory intensive for huge datasets. Consider alternatives
    # like saving multiple shards or using memory-mapped files directly if needed.
    all_token_ids = []
    for example in tqdm(tokenized_dataset, desc="Concatenating"):
        all_token_ids.extend(example['input_ids'])

    print(f"Total number of tokens: {len(all_token_ids)}")

    # Save as NumPy array
    print(f"Saving concatenated tokens to {args.output_file}")
    np.save(args.output_file, np.array(all_token_ids, dtype=np.uint32)) # Use uint32 for efficiency if vocab_size allows

    print("Data preparation finished.")

if __name__ == "__main__":
    main()