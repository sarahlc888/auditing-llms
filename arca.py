from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from losses import log_prob_loss, log_perplexity
from utils import get_forbidden_toks, filter_forbidden_toks, get_unigram_probs 

def run_arca_multi(args, model, tokenizer, embedding_table, dataset, slice_positions_labels):
    """Version of run_arca that optimizes a single string for multiple data points"""
    # Fixed output is used in the reverse case
    fixed_output = True # output_strs is not None
    run_metadata = {}
    args.batch_size = args.arca_batch_size

    # Avoid degenerate solutions + additional constraints specified in args
    forbidden_input_toks = set()
    for i in range(len(dataset)):
        ids = dataset['input_ids'][i]

        slice_pos = dataset['slice_positions'][i]
        j = slice_positions_labels.index('target')
        target_slice = slice(slice_pos[j], slice_pos[j+1])
        output_str = tokenizer.decode(ids[target_slice])
        print(f"DEBUG: {output_str=}")

        temp_toks = get_forbidden_toks(args, tokenizer, n_total_toks = embedding_table.shape[0], 
            output = False, output_str = output_str)
        forbidden_input_toks = forbidden_input_toks.union(temp_toks)
    forbidden_input_toks = np.array(list(forbidden_input_toks))

    # Set up
    vocab_size = embedding_table.shape[0]
    embedding_dim = embedding_table.shape[1]

    assert fixed_output # TODO: support other case
    assert args.unigram_output_constraint is None

    if args.unigram_input_constraint is not None:
        input_unigram_lps = get_unigram_probs(args.unigram_input_constraint, gptj = args.model_id == 'gptj')

    # Populate initial tokens
    prompt_toks = np.random.choice(vocab_size, size = args.prompt_length, replace=True)
    stacked_prompt_toks = np.tile(prompt_toks, (args.batch_size, 1))
    prompt_toks_tensor = torch.Tensor(stacked_prompt_toks).long().to(args.device)
    
    prompt_embeddings = torch.zeros(args.batch_size, args.prompt_length, embedding_dim).to(args.device)
    for i in range(args.prompt_length):
        prompt_embeddings[:, i] = embedding_table[prompt_toks[i]].unsqueeze(0).repeat(args.batch_size, 1)
    print(f"DEBUG: {prompt_embeddings.shape=}")

    # Iterate
    for it in tqdm(range(args.arca_iters)):        
        for tok_id in range(args.prompt_length): # index within prompt_toks
            print(f"DEBUG: {tok_id=}")

            # iterate over the dataset and accumulate `batch_grad`            
            batch_grad = torch.zeros(embedding_dim).to(args.device)
            if not args.model_id.startswith('gpt2'):
                batch_grad = batch_grad.half()
            for i in range(len(dataset)):
                ids = dataset['input_ids'][i]

                slice_pos = dataset['slice_positions'][i]
                print(f"DEBUG: {slice_pos=}")
                j = slice_positions_labels.index('control')
                control_slice = slice(slice_pos[j], slice_pos[j+1])
                j = slice_positions_labels.index('target')
                target_slice = slice(slice_pos[j], slice_pos[j+1])
                offset = slice_pos[0]
                example_tok_len = target_slice.stop - offset

                # Output tokens remain fixed in the reversing case
                tok_in_output = (tok_id + control_slice.start) >= target_slice.start and (tok_id + control_slice.start) < target_slice.stop  
                if tok_in_output and fixed_output:
                    continue

                update_idx = tok_id + control_slice.start - offset # index within full embeddings

                # Randomly pick some replacement tokens
                new_indices = np.random.choice(vocab_size, size = args.batch_size, replace = True) 
                print(f"DEBUG: {new_indices.shape=}")

                prompt_embeddings[:, tok_id, :] = embedding_table[new_indices, :] # populate full_embeddings with the random tokens at the current index
                if not args.model_id.startswith('gpt2'):
                    prompt_embeddings = prompt_embeddings.half()

                # Update to compute the perplexity loss
                stacked_prompt_toks[:, tok_id] = new_indices
                prompt_toks_tensor[:, tok_id] = torch.Tensor(new_indices).long().to(args.device)

                # Get full embeddings (prefix, prompt, and output)
                full_embeddings = torch.zeros(args.batch_size, example_tok_len, embedding_dim).to(args.device)
                print(f"debug: {control_slice=} {target_slice=} {offset=} {full_embeddings.shape=} {len(ids)=}")
                full_embeddings[:, :control_slice.start-offset] = embedding_table[ids[offset:control_slice.start]] # broadcast 
                full_embeddings[:, control_slice.stop-offset:] = embedding_table[ids[control_slice.stop:target_slice.stop]] # broadcast 
                full_embeddings[:, control_slice.start-offset:control_slice.stop-offset] = prompt_embeddings
                if not args.model_id.startswith('gpt2'):
                    full_embeddings = full_embeddings.half()
                full_embeddings = full_embeddings.detach()
                if full_embeddings.requires_grad:
                    full_embeddings.grad.zero_()
                full_embeddings.requires_grad = True
                full_embeddings.retain_grad()
                # Get the labels
                labels = torch.cat(
                    [-100 * torch.ones(target_slice.start - offset).to(args.device).unsqueeze(0).repeat(args.batch_size, 1), 
                    torch.Tensor(ids[target_slice.start:target_slice.stop]).to(args.device).unsqueeze(0).repeat(args.batch_size, 1)
                    ], 
                    dim = 1).long()
                print(f"DEBUG: {full_embeddings.shape=}")
                print(f"DEBUG: {labels.shape=}")
                print(f"{tokenizer.batch_decode(labels[:3, target_slice.start - offset:])=}")

                out = model(inputs_embeds=full_embeddings, labels=labels)
                loss = log_prob_loss(out, labels, temp=1)
                
                # Compute the perplexity loss
                if args.lam_perp > 0:
                    raise NotImplementedError # TODO: implement this
                    # perp_loss = log_perplexity(out, stacked_cur_toks[:,:target_slice.start])
                    # loss += args.lam_perp * perp_loss
                
                # Get first order approximation to get candidates (averaged)
                loss.backward(retain_graph = True)
                grad = full_embeddings.grad # shape (args.batch_size, args.prompt_length + args.output_length, embedding_dim)
                if grad is None:
                    print("WARNING: grad is none")
                batch_grad += grad[:,update_idx,:].mean(dim = 0) # shape (embedding_dim,); averaged over dim 0 which was the diff sampled random embeds
                
            print(f"DEBUG: {batch_grad.shape=} {embedding_table.type()=} {batch_grad.type()=}")
            scores = - torch.matmul(embedding_table, batch_grad) # shape (vocab_size,)
            if args.unigram_input_constraint is not None:
                scores += args.unigram_weight * input_unigram_lps
        
            # Get the best scores and calculate loss exactly
            best_scores_idxs = scores.argsort(descending = True)
            best_scores_idxs = filter_forbidden_toks(best_scores_idxs, forbidden_input_toks)
            print(f"DEBUG: {best_scores_idxs.shape=}")

            full_embeddings = full_embeddings.detach()
            with torch.no_grad():
                batch_log_probs = torch.zeros(args.batch_size).to(args.device)
                for i in range(len(dataset)):
                    k = 12 # debug TODO

                    # indexing within ids
                    ids = dataset['input_ids'][i]
                    slice_pos = dataset['slice_positions'][i]
                    j = slice_positions_labels.index('control')
                    control_slice = slice(slice_pos[j], slice_pos[j+1])
                    j = slice_positions_labels.index('target')
                    target_slice = slice(slice_pos[j], slice_pos[j+1])
                    offset = slice_pos[0]
                    example_tok_len = target_slice.stop - offset

                    update_idx = tok_id + control_slice.start - offset # index within full embeddings

                    # populate embeddings for this example; remove padding offset since it's processing 1 at a atime
                    full_embeddings = torch.zeros(args.batch_size, example_tok_len, embedding_dim).to(args.device)
                    full_embeddings[:, :control_slice.start-offset] = embedding_table[ids[offset:control_slice.start]] # broadcast 
                    full_embeddings[:, control_slice.stop-offset:] = embedding_table[ids[control_slice.stop:target_slice.stop]] # broadcast 
                    full_embeddings[:, control_slice.start-offset:control_slice.stop-offset] = prompt_embeddings
                    print(f"DEBUG: 0-{control_slice.start-offset}; {control_slice.stop-offset}-end")


                    full_embeddings[:, update_idx, :] = embedding_table[best_scores_idxs[:args.batch_size], :]                
                    stacked_prompt_toks[:, tok_id] = best_scores_idxs[:args.batch_size].cpu().detach().numpy()
                    prompt_toks_tensor[:, tok_id] = best_scores_idxs[:args.batch_size]

                    full_ids = torch.cat( [ torch.Tensor(ids[offset:control_slice.start]).to(args.device), 
                                            prompt_toks_tensor[k],
                                            torch.Tensor(ids[control_slice.stop:target_slice.stop]).to(args.device) ],  ).long()
                    full_embeddings_copy = embedding_table[full_ids]
                    if not args.model_id.startswith('gpt2'):
                        full_embeddings = full_embeddings.half()
                        full_embeddings_copy= full_embeddings_copy.half()
                    print(f'DEBUG: {k=}', 
                        'comparison2', 
                        full_embeddings[k].shape, 
                        full_embeddings_copy.shape, 
                        # torch.allclose(full_embeddings[k], full_embeddings_copy, atol=1e-4),
                        torch.allclose(full_embeddings[k, :control_slice.start-offset], full_embeddings_copy[:control_slice.start-offset], atol=1e-4),
                        torch.allclose(full_embeddings[k, control_slice.start-offset:control_slice.stop-offset], full_embeddings_copy[control_slice.start-offset:control_slice.stop-offset], atol=1e-4),
                        torch.allclose(full_embeddings[k, control_slice.stop-offset:], full_embeddings_copy[control_slice.stop-offset:], atol=1e-4),
                        )
                    for idx in range(control_slice.start-offset, control_slice.stop-offset):
                        isclose = torch.allclose(full_embeddings[k, idx], full_embeddings_copy[idx], atol=1e-4)
                        print(idx, isclose)
                        if not isclose:
                            print('\t', full_embeddings[k, idx], full_embeddings_copy[idx])

                    # Run forward for exact calculation
                    if not args.model_id.startswith('gpt2'):
                        full_embeddings = full_embeddings.half()
                    out = model(inputs_embeds=full_embeddings)
                    out_logits = out.logits[:, -1 - (target_slice.stop-target_slice.start): -1, :]
                    print(f"DEBUG: {out_logits.shape=}")
                    log_probs = F.log_softmax(out_logits, dim = 2)
                    # log_probs = F.log_softmax(out.logits[:, -1 - args.output_length: -1, :], dim = 2)
                    # log_probs.shape is (args.batch_size, example_tok_len, vocab_size)
                    # shape batch_log_probs is 
                    output_ids = ids[target_slice.start:target_slice.stop]
                    print(f"DEBUG: {log_probs.shape=} {(target_slice.stop-target_slice.start)=}")
                    print('debug', log_probs[1, torch.arange(target_slice.stop-target_slice.start), output_ids].sum())
                    cur_batch_log_probs = torch.stack([
                        log_probs[k, torch.arange(target_slice.stop-target_slice.start), output_ids].sum() for k in range(args.batch_size)
                    ])
                    print(f"DEBUG: {cur_batch_log_probs.shape=}")
                    if args.lam_perp > 0:
                        raise NotImplementedError
                    elif args.unigram_input_constraint is not None and not tok_in_output:
                        cur_batch_log_probs += args.unigram_weight * input_unigram_lps[best_scores_idxs[:args.batch_size]]
                    batch_log_probs += cur_batch_log_probs
                # Identify the best item 
                best_batch_idx = batch_log_probs.argmax()
                best_idx = best_scores_idxs[best_batch_idx]
                prompt_toks[tok_id] = best_idx.item()
                stacked_prompt_toks[:, tok_id] = best_idx.item()
                prompt_toks_tensor[:, tok_id] = best_idx.item()
                prompt_embeddings[:, tok_id, :] = embedding_table[best_idx].unsqueeze(0).repeat(args.batch_size, 1)
                full_embeddings[:, update_idx, :] = embedding_table[best_idx].unsqueeze(0).repeat(args.batch_size, 1)
                
                # TODO: evaluate, somehow 
                print(f"DEBUG: produced {tokenizer.decode(prompt_toks)=}")


                # gen_output = log_probs[best_batch_idx].argmax(dim = 1)
                # actual_output = curr_toks_tensor[0][output_start:]
                # print(f"DEBUG: {gen_output=} {actual_output=}")
                # print(f"DEBUG: |{tokenizer.decode(gen_output)}| |{tokenizer.decode(actual_output)}|")
                # # Can modify success conditions here to stop running the algorithm
                # output_matches = (actual_output == gen_output).all().item()
                # if args.unigram_input_constraint is not None:
                #     input_unigram_satisfied  = torch.exp(input_unigram_lps[prompt_toks].min()).item() > 0.99
                # else:
                #     input_unigram_satisfied = True
                # output_unigram_satisfied = True
                # # Success condition
                # if output_matches and input_unigram_satisfied and output_unigram_satisfied:
                #     if args.lam_perp > 0:
                #         run_metadata['perplexity'] = output_perps[best_batch_idx].item()
                #     if args.unigram_input_constraint is not None:
                #         run_metadata['input_unigram'] = torch.exp(input_unigram_lps[prompt_toks]).mean().item()
                #         run_metadata['max_input_unigram'] = torch.exp(input_unigram_lps[prompt_toks].max()).item()
                #         run_metadata['min_input_unigram'] = torch.exp(input_unigram_lps[prompt_toks].min()).item()
                #     if fixed_output:
                #         curr_toks = curr_toks[:-args.output_length]
                #     return curr_toks, it, run_metadata
    # Failure case
    if args.lam_perp > 0:
        run_metadata['perplexity'] = None
        if args.unigram_output_constraint is not None:
            run_metadata['output_unigram'] = -1
        elif args.unigram_input_constraint is not None:
            run_metadata['input_unigram'] = -1
    return -1, -1, run_metadata

def run_arca(args, model, tokenizer, embedding_table, output_str = None):
    # Fixed output is used in the reverse case
    fixed_output = output_str is not None
    run_metadata = {}
    args.batch_size = args.arca_batch_size
    embedding_dim = embedding_table.shape[1]
    # Avoid degenerate solutions + additional constraints specified in args
    forbidden_input_toks = get_forbidden_toks(args, tokenizer, n_total_toks = embedding_table.shape[0], 
            output = False, output_str = output_str)
    if not fixed_output:
        forbidden_output_toks = get_forbidden_toks(args, tokenizer, n_total_toks = embedding_table.shape[0], 
                output = True, output_str = output_str)
    # Whether or not to use a fixed prompt prefix
    use_pp = args.prompt_prefix is not None
    if use_pp:
        prefix_toks = torch.Tensor(tokenizer(args.prompt_prefix)['input_ids']).long().to(args.device)
        prefix_embeddings = embedding_table[prefix_toks].unsqueeze(0)
        prefix_embeddings = prefix_embeddings.repeat(args.batch_size, 1, 1).detach()
        prefix_length = prefix_embeddings.shape[1]

    vocab_size = embedding_table.shape[0]
    embedding_dim = embedding_table.shape[1]
    if fixed_output:
        output_toks = np.array(tokenizer(output_str)['input_ids'])
        output_toks_tensor = torch.Tensor(tokenizer(output_str)['input_ids']).long().to(args.device)
        args.output_length = output_toks.shape[0]
        run_metadata['n_output_toks'] = args.output_length
        assert args.unigram_output_constraint is None

    curr_toks = np.random.choice(vocab_size, size = args.prompt_length + args.output_length, replace = True)
    if fixed_output:
        curr_toks[args.prompt_length:] = output_toks
    if use_pp:
        curr_toks = np.concatenate([prefix_toks.detach().cpu().numpy(), curr_toks], axis = 0)
    stacked_cur_toks = np.tile(curr_toks, (args.batch_size, 1))
    curr_toks_tensor = torch.Tensor(stacked_cur_toks).long().to(args.device)
    
    if args.unigram_output_constraint is not None:
        output_unigram_lps = get_unigram_probs(args.unigram_output_constraint, gptj = args.model_id == 'gptj')
    if args.unigram_input_constraint is not None:
        input_unigram_lps = get_unigram_probs(args.unigram_input_constraint, gptj = args.model_id == 'gptj')

    output_start = args.prompt_length + prefix_length if use_pp else args.prompt_length
    full_embeddings = torch.zeros(args.batch_size, args.prompt_length + args.output_length, embedding_dim).to(args.device)
    # Initialize full embeddings
    for i in range(args.prompt_length + args.output_length):
        rel_idx = i + prefix_length if use_pp else i
        full_embeddings[:, i] = embedding_table[curr_toks[rel_idx]].unsqueeze(0).repeat(args.batch_size, 1)
    # Iterate
    for it in tqdm(range(args.arca_iters)):
        for tok_id in range(args.prompt_length + args.output_length):
            tok_in_output = tok_id >= args.prompt_length
            # Output tokens remain fixed in the reversing case
            if tok_in_output and fixed_output:
                continue
            update_idx = tok_id + prefix_length if use_pp else tok_id
            new_indices = np.random.choice(vocab_size, size = args.batch_size, replace = True) # randomly pick some tokens
            if args.autoprompt:
                new_indices = curr_toks[update_idx].repeat(args.batch_size)
            full_embeddings[:, tok_id, :] = embedding_table[new_indices, :] # populate full_embeddings with the random tokens at the current index
            if args.model_id == 'gptj':
                full_embeddings = full_embeddings.half()
            # Update to compute the perplexity loss
            stacked_cur_toks[:, update_idx] = new_indices
            curr_toks_tensor[:, update_idx] = torch.Tensor(new_indices).long().to(args.device)
            if use_pp:
                labels = torch.cat([-100 * torch.ones(args.prompt_length + prefix_length).to(args.device).unsqueeze(0).repeat(args.batch_size, 1), curr_toks_tensor[:, args.prompt_length + prefix_length:]], dim = 1).long()
            else:
                labels = torch.cat([-100 * torch.ones(args.prompt_length).to(args.device).unsqueeze(0).repeat(args.batch_size, 1), curr_toks_tensor[:, args.prompt_length:]], dim = 1).long()
            full_embeddings = full_embeddings.detach()
            if full_embeddings.requires_grad:
                full_embeddings.grad.zero_()
            full_embeddings.requires_grad = True
            full_embeddings.retain_grad()
            if use_pp:
                out = model(inputs_embeds = torch.cat([prefix_embeddings, full_embeddings], dim = 1), labels = labels)
            else:
                out = model(inputs_embeds = full_embeddings, labels = labels)
            loss = log_prob_loss(out, labels, temp = 1)
            # Comptue the perplexity loss
            if args.lam_perp > 0:
                perp_loss = log_perplexity(out, stacked_cur_toks[:,:output_start])
                loss += args.lam_perp * perp_loss
            
            # Get first order approximation to get candidates (averaged)
            loss.backward(retain_graph = True)
            grad = full_embeddings.grad # shape (args.batch_size, args.prompt_length + args.output_length, embedding_dim)
            batch_grad = grad[:,tok_id,:].mean(dim = 0) # shape (embedding_dim,); averaged over dim 0 which was the diff sampled random embeds
            backward_scores = - torch.matmul(embedding_table, batch_grad) 
            if tok_in_output and not args.autoprompt:
                forward_log_probs = F.log_softmax(out.logits[0, update_idx - 1, :], dim = 0)
                scores = backward_scores + forward_log_probs
                if args.unigram_output_constraint is not None:
                    scores += args.unigram_weight * output_unigram_lps
            else:
                scores = backward_scores
                if args.unigram_input_constraint is not None:
                    scores += args.unigram_weight * input_unigram_lps
            
            # Get the best scores and calculate loss exactly
            best_scores_idxs = scores.argsort(descending = True)
            if tok_in_output:
                best_scores_idxs = filter_forbidden_toks(best_scores_idxs, forbidden_output_toks)
            else:
                best_scores_idxs = filter_forbidden_toks(best_scores_idxs, forbidden_input_toks)
            full_embeddings= full_embeddings.detach()
            with torch.no_grad():
                full_embeddings[:, tok_id, :] = embedding_table[best_scores_idxs[:args.batch_size], :]                
                stacked_cur_toks[:, update_idx] = best_scores_idxs[:args.batch_size].cpu().detach().numpy()
                curr_toks_tensor[:, tok_id] = best_scores_idxs[:args.batch_size]
                # Run forward for exact calculation
                if use_pp:
                    out = model(inputs_embeds = torch.cat([prefix_embeddings, full_embeddings], dim = 1))
                else:
                    out = model(inputs_embeds = full_embeddings)
                log_probs = F.log_softmax(out.logits[:, -1 - args.output_length: -1, :], dim = 2)
                batch_log_probs = torch.stack([log_probs[i, torch.arange(args.output_length), curr_toks_tensor[i, output_start:]].sum() for i in range(args.batch_size)])
                if args.lam_perp > 0:
                    output_perps = log_perplexity(out, stacked_cur_toks[:,:output_start], ret_all = True)
                    batch_log_probs -= args.lam_perp * output_perps
                if args.unigram_output_constraint is not None and tok_in_output:
                    batch_log_probs += args.unigram_weight * output_unigram_lps[best_scores_idxs[:args.batch_size]]
                elif args.unigram_input_constraint is not None and not tok_in_output:
                    batch_log_probs += args.unigram_weight * input_unigram_lps[best_scores_idxs[:args.batch_size]]

                # Identify the best item 
                best_batch_idx = batch_log_probs.argmax()
                best_idx = best_scores_idxs[best_batch_idx]
                curr_toks[update_idx] = best_idx.item()
                stacked_cur_toks[:, update_idx] = best_idx.item()
                curr_toks_tensor[:, update_idx] = best_idx.item()
                full_embeddings[:, tok_id, :] = embedding_table[best_idx].unsqueeze(0).repeat(args.batch_size, 1)
                gen_output = log_probs[best_batch_idx].argmax(dim = 1)
                actual_output = curr_toks_tensor[0][output_start:]
                # Can modify success conditions here to stop running the algorithm
                output_matches = (actual_output == gen_output).all().item()
                if args.unigram_input_constraint is not None:
                    input_unigram_satisfied  = torch.exp(input_unigram_lps[curr_toks[:output_start]].min()).item() > 0.99
                else:
                    input_unigram_satisfied = True
                if args.unigram_output_constraint is not None and not fixed_output:
                    output_unigram_satisfied = torch.exp(output_unigram_lps[curr_toks[output_start:]].max()).item() > 0.5
                else:
                    output_unigram_satisfied = True
                # Success condition
                if output_matches and input_unigram_satisfied and output_unigram_satisfied:
                    if args.lam_perp > 0:
                        run_metadata['perplexity'] = output_perps[best_batch_idx].item()
                    if args.unigram_output_constraint is not None:
                        run_metadata['output_unigram'] = torch.exp(output_unigram_lps[curr_toks[output_start:]]).mean().item()
                        run_metadata['max_output_unigram'] = torch.exp(output_unigram_lps[curr_toks[output_start:]].max()).item()
                        run_metadata['min_output_unigram'] = torch.exp(output_unigram_lps[curr_toks[output_start:]].min()).item()
                    if args.unigram_input_constraint is not None:
                        run_metadata['input_unigram'] = torch.exp(input_unigram_lps[curr_toks[:output_start]]).mean().item()
                        run_metadata['max_input_unigram'] = torch.exp(input_unigram_lps[curr_toks[:output_start]].max()).item()
                        run_metadata['min_input_unigram'] = torch.exp(input_unigram_lps[curr_toks[:output_start]].min()).item()
                    if fixed_output:
                        curr_toks = curr_toks[:-args.output_length]
                    return curr_toks, it, run_metadata
    # Failure case
    if args.lam_perp > 0:
        run_metadata['perplexity'] = None
        if args.unigram_output_constraint is not None:
            run_metadata['output_unigram'] = -1
        elif args.unigram_input_constraint is not None:
            run_metadata['input_unigram'] = -1
    return -1, -1, run_metadata
