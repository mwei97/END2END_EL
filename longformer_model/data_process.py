import os
import json
import torch
import numpy as np
from tqdm import tqdm

def read_dataset(fpath, split):
    fname = os.path.join(fpath, split)+'.jsonl'
    samples = []
    with open(fname) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]

def select_field_with_padding(data, key1, key2=None, pad_idx=-1):
    max_len = 0
    selected_list = []
    padding_mask = []
    for example in data:
        if key2 is None:
            selected_list.append(example[key1])
            max_len = max(max_len, len(example[key1]))
        else:
            selected_list.append(example[key1][key2])
            max_len = max(max_len, len(example[key1][key2]))
    for i, entry in enumerate(selected_list):
        # pad to max len
        pad_list = [1 for _ in range(len(entry))] + [0 for _ in range(max_len - len(entry))]
        assert len(pad_list) == max_len
        padding_mask.append(pad_list)
        if len(entry)==max_len:
            continue
        if isinstance(pad_idx, np.ndarray):
            selected_list[i] = np.vstack((entry, np.array([pad_idx for _ in range(max_len - len(entry))])))
        else:
            selected_list[i] += [pad_idx for _ in range(max_len - len(entry))]
        assert len(selected_list[i]) == max_len
    return selected_list, padding_mask

def get_context_representation_multiple_mentions(
    sample, tokenizer, max_context_length,
    input_key='tokenized_text_ids', mention_key='tokenized_mention_idxs',
    use_longformer=True
):
    mention_idxs = sample[mention_key]
    input_ids = sample[input_key]

    all_mention_spans_range = [mention_idxs[0][0], mention_idxs[-1][1]]
    max_seq_length = max_context_length if use_longformer else max_context_length-2
    #while all_mention_spans_range[1] - all_mention_spans_range[0] > max_context_length:
    while all_mention_spans_range[1] - all_mention_spans_range[0] > max_seq_length:
        if len(mention_idxs) == 1:
            # don't cut further
            #assert mention_idxs[0][1] - mention_idxs[0][0] > max_context_length
            assert mention_idxs[0][1] - mention_idxs[0][0] > max_seq_length
            # truncate mention
            #mention_idxs[0][1] = max_context_length + mention_idxs[0][0]
            mention_idxs[0][1] = max_seq_length + mention_idxs[0][0]
        else:
            # cut last mention
            mention_idxs = mention_idxs[:len(mention_idxs) - 1]
        all_mention_spans_range = [mention_idxs[0][0], mention_idxs[-1][1]]
    
    context_left = input_ids[:all_mention_spans_range[0]]
    all_mention_tokens = input_ids[all_mention_spans_range[0]:all_mention_spans_range[1]]
    context_right = input_ids[all_mention_spans_range[1]:]
    
    left_quota = (max_context_length - len(all_mention_tokens)) // 2
    if not use_longformer:
        left_quota -= 1
    right_quota = max_context_length - len(all_mention_tokens) - left_quota
    if not use_longformer:
        right_quota -= 2
    left_add = len(context_left)
    right_add = len(context_right)
    
    if left_add <= left_quota:  # tokens left to add <= quota ON THE LEFT
        if right_add > right_quota:  # add remaining quota to right quota
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:  # tokens left to add <= quota ON THE RIGHT
            left_quota += right_quota - right_add  # add remaining quota to left quota
    
    if left_quota <= 0:
        left_quota = -len(context_left)  # cut entire list (context_left = [])
    if right_quota <= 0:
        right_quota = 0  # cut entire list (context_right = [])
    input_ids_window = context_left[-left_quota:] + all_mention_tokens + context_right[:right_quota]
    
    #if len(input_ids) <= max_context_length:
    if len(input_ids) <= max_seq_length:
        try:
            # if length of original input_ids is sufficient to fit in max_seq_length, then none of the above
            # opeartions should be done
            assert input_ids == input_ids_window
        except:
            import pdb
            pdb.set_trace()
    else:
        assert input_ids != input_ids_window
        cut_from_left = len(context_left) - len(context_left[-left_quota:])
        if cut_from_left > 0:
            # must shift mention_idxs
            for c in range(len(mention_idxs)):
                mention_idxs[c] = [
                    mention_idxs[c][0] - cut_from_left, mention_idxs[c][1] - cut_from_left,
                ]
    
    if not use_longformer:
        # add [CLS] and [SEP] to the start and end respectively
        input_ids_window = [101] + input_ids_window + [102]
    tokens = tokenizer.convert_ids_to_tokens(input_ids_window)

    attention_mask = [1]*len(input_ids_window)

    padding = [0] * (max_context_length - len(input_ids_window))
    input_ids_window += padding
    attention_mask += padding
    assert len(input_ids_window) == max_context_length

    if not use_longformer:
        # +1 for CLS token
        mention_idxs = [[mention[0]+1, mention[1]+1] for mention in mention_idxs]
    
    return {
        'tokens':tokens,
        'ids': input_ids_window,
        'attention_mask': attention_mask,
        'mention_idxs': mention_idxs
    }

def get_ner_tag(
    context_tokens, max_context_length, end_tag=False,
    token_key='tokens', mention_key='mention_idxs',
    pad_id=-1
):
    """
    Given context tokens, return NER tags with global attention mask (all B/I tokens have global attention)
    """
    tokens = context_tokens[token_key]
    mention_idxs = context_tokens[mention_key] # inclusive bounds
    ## 1: O, 2: B, 3: I, 0: padding
    # 0: O, 1: B, 2: I, -1: padding
    #ner_tags = [1]*len(tokens) # initialize to be not entity (O)
    ner_tags = [0]*len(tokens) # initialize to be not entity(O)
    global_attention_mask = [0]*len(tokens) # initialize to local attention
    # b_tag = 2
    # i_tag = 3
    # e_tag = 4 if end_tag else 3
    b_tag = 1
    i_tag = 2
    e_tag = 3 if end_tag else 2
    for mention_idx in mention_idxs:
        b = mention_idx[0]
        e = mention_idx[1]
        ner_tags[b] = b_tag
        if e>b:
            ner_tags[b+1:e] = [i_tag]*(e-(b+1))
            ner_tags[e] = e_tag
        global_attention_mask[b:e+1] = [1]*(e+1-b)
        # if e==s:
        #     continue
        # elif e>s:
        #     ner_tags[b+1:e] = [i_tag]*(e-(b+1))
        #     ner_tags[e] = e_tag
    tag_paddings = [pad_id] * (max_context_length - len(tokens))
    ner_tags += tag_paddings
    attn_paddings = [0] * (max_context_length - len(tokens))
    global_attention_mask += attn_paddings

    return ner_tags, global_attention_mask

def process_mention_data(
    samples,
    tokenizer,
    max_context_length=512,
    silent=False,
    end_tag=False,
    is_biencoder=False,
    cand_enc_path=None,
    use_longformer=True
):
    processed_samples = []
    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    if is_biencoder:
        assert cand_enc_path is not None
        with open(cand_enc_path) as f:
            golden_cand_enc = json.load(f)

    for idx, sample in enumerate(iter_):
        # if no mentions in the document, skip
        if len(sample['mentions'])==0:
            continue

        context_tokens = get_context_representation_multiple_mentions(
            sample, tokenizer, max_context_length, use_longformer=use_longformer
        )

        for i in range(len(context_tokens["mention_idxs"])):
            context_tokens["mention_idxs"][i][1] -= 1  # make bounds inclusive

        ner_tags, global_attention_mask = get_ner_tag(context_tokens, max_context_length, end_tag=end_tag)
        context_tokens['ner_tags'] = ner_tags
        context_tokens['global_attention_mask'] = global_attention_mask

        # todo: delete/modify later
        if is_biencoder:
            sample_id = sample['id']
            num_mentions = len(context_tokens['mention_idxs'])
            cand_enc = np.array(golden_cand_enc[sample_id][:num_mentions])
            #assert len(cand_enc)==num_mentions
            #cand_enc = np.random.randn(num_mentions, 1024)
            context_tokens['cand_enc'] = cand_enc

        processed_samples.append(context_tokens)
    
    # (num_samples, max_context_length)
    context_vecs = torch.tensor(select_field(processed_samples, 'ids'), dtype=torch.long)#.to(device)
    ner_tag_vecs = torch.tensor(select_field(processed_samples, 'ner_tags'), dtype=torch.long)#.to(device)
    mask_vecs = torch.tensor(select_field(processed_samples, 'attention_mask'), dtype=torch.bool)#.to(device)
    global_attn_mask_vecs = torch.tensor(select_field(processed_samples, 'global_attention_mask'), dtype=torch.bool)#.to(device)
    if is_biencoder:
        cand_enc_vecs, cand_enc_mask = select_field_with_padding(processed_samples, 'cand_enc', pad_idx=np.zeros(1024)) # todo: dim as variable?
        # (num_samples, max_num_mentions, cand_enc_dim)
        cand_enc_vecs = torch.tensor(cand_enc_vecs, dtype=torch.float)
        cand_enc_mask = torch.tensor(cand_enc_mask, dtype=torch.bool)
        return (context_vecs, ner_tag_vecs, cand_enc_vecs, cand_enc_mask, mask_vecs, global_attn_mask_vecs)
    else:
        return (context_vecs, ner_tag_vecs, mask_vecs, global_attn_mask_vecs)

