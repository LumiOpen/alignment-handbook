#!/bin/bash
#SBATCH --job-name=format_aya  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=small       # Partition (queue) name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --cpus-per-task=16     # Number of cores (threads)
#SBATCH --time=01:30:00         # Run time (hh:mm:ss)
#SBATCH --account=project_462000353  # Project for billing

module use /appl/local/csc/modulefiles/
module load pytorch

export HF_HOME=/scratch/project_462000444/cache

# Define language codes and their full names
declare -A lang_dict=(
    ['bul']='bulgarian'
    ['hrv']='croatian'
    ['cze']='czech'
    ['dan']='danish'
    ['nld']='dutch'
    ['eng']='english'
    ['est']='estonian'
    ['fra']='french'
    ['deu']='german'
    ['ell']='greek'
    ['hun']='hungarian'
    ['gle']='irish'
    ['ita']='italian'
    ['lit']='lithuanian'
    ['mlt']='maltese'
    ['pol']='polish'
    ['por']='portuguese'
    ['ron']='romanian'
    ['slk']='slovak'
    ['slv']='slovenian'
    ['spa']='spanish'
    ['swe']='swedish'
    ['nob']='norwegian_bokmal'
    ['nno']='norwegian_nynorsk'
    ['isl']='icelandic'
)

# Loop through each language code and format the dataset files
for trg in "${!lang_dict[@]}"
do
    lang_full=${lang_dict[$trg]}
    echo "Processing language: $lang_full ($trg)"

    # Check if the directory exists, if not, create it
    output_dir="/scratch/project_462000444/finetuning_data/SFTTrainer_format/${trg}/Aya-Dataset"
    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
        echo "Created directory: $output_dir"
    fi

    python format_sft_data.py \
        --filepath /scratch/project_462000444/finetuning_data/Aya/${lang_full}/train.jsonl \
        --outfile ${output_dir}/train.jsonl \
        --dataset_name Aya-Dataset \

    python format_sft_data.py \
        --filepath /scratch/project_462000444/finetuning_data/Aya/${lang_full}/test.jsonl \
        --outfile ${output_dir}/test.jsonl \
        --max_samples 1000 \
        --dataset_name Aya-Dataset \

done
