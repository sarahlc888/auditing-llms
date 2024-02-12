#!/bin/bash
export label="temp_trial"
export model_id="meta-llama/Llama-2-7b-chat-hf"
export filename="/nlp/scr/sachen/discrete_optimization/prompt_opt/copy_data.icl_raw.csv"
export lam_perp=0 # 0.2
python icl_experiment.py --save_every 10 \
--n_trials 1 \
--arca_iters 50 \
--arca_batch_size 32 \
--prompt_length 3 \
--lam_perp $lam_perp \
--label $label \
--filename $filename \
--opts_to_run arca \
--model_id $model_id 
# --dry_run --device cpu