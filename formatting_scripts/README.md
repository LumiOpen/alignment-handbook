These scripts are for formatting datasets downloaded from HF (for example) into the format used by the TRL trainers such as SFTTrainer, DPOTrainer, etc.
[SFTTrainer format](https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support):
```
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "..."}]}
```

### Example usage
To transform the English portion of the Aya dataset into SFTTrainer format:
```
    python format_sft_data.py \
        --filepath /scratch/project_462000444/finetuning_data/Aya/english/train.jsonl \
        --outfile /scratch/project_462000444/finetuning_data/SFTTrainer_format/english/train.jsonl
```
