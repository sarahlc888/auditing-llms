from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from utils import to_jsonl, get_output_file, load_outputs
from args_utils import parse_args
from arca import run_arca_multi
from model_utils import get_raw_embedding_table, get_model_and_tokenizer
import pandas as pd 

def run_opts(args, model, tokenizer, embedding_table, dataset, slice_positions_labels):
    output_filename = './temp.out.txt'

    assert args.opts_to_run == ['arca']
    args.autoprompt = False
    attack_name = 'arca'

    results_dict = {}

    prompts = []
    n_iters = []
    attack_times = []
    all_prompt_toks = []
    metadata = defaultdict(list)
    successes = 0
    for trial in range(args.n_trials):
        start = datetime.now()
        prompt_toks, n_iter, run_metadata = run_arca_multi(args, model, tokenizer, embedding_table, dataset=dataset, slice_positions_labels=slice_positions_labels)
        if n_iter == -1:
            prompt = None
        else:
            prompt = tokenizer.decode(prompt_toks)
            prompt_toks = list(prompt_toks)
            successes += 1
        prompts.append(prompt)
        all_prompt_toks.append(prompt_toks)
        n_iters.append(n_iter)
        attack_times.append((datetime.now() - start).seconds)
        for key in run_metadata:
            metadata[key].append(run_metadata[key])

    # Log results 
    results_dict[f'{attack_name}'] = {}
    results_dict[f'{attack_name}']['prompts'] = prompts
    results_dict[f'{attack_name}']['prompt_toks'] = all_prompt_toks
    results_dict[f'{attack_name}']['iters'] = n_iters
    results_dict[f'{attack_name}']['time'] = attack_times 
    results_dict[f'{attack_name}']['success_rate'] = successes / args.n_trials
    for key in metadata:
        results_dict[f'{attack_name}'][key] = metadata[key]
    

    print("Saving...")
    all_dicts = [vars(args)] + [results_dict]
    to_jsonl(all_dicts, output_filename)


if __name__ == '__main__':
    args = parse_args()


    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    import sys 
    sys.path.append('/nlp/scr/sachen/discrete_optimization/prompt_opt')
    from data_utils import get_data_generic, slice_positions_labels
    placement = 'userEnd'
    dataset = get_data_generic(tokenizer, data_mode='icl', control_config={'control_string': '! ! !'}, 
                    data_logfile=None,
                    inject_target_prefix="",
                    placement=placement)
    print(dataset.keys())
    dataset = dataset['train']
    print(dataset[0].keys())


    if args.dry_run:
        model, embedding_table = None, None
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)


    else:
        model, tokenizer = get_model_and_tokenizer(args)
        embedding_table = get_raw_embedding_table(model)

    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    run_opts(args, model, tokenizer, embedding_table, dataset=dataset, slice_positions_labels=slice_positions_labels[placement])
