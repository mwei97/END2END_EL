from tqdm import tqdm
import torch

def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]

def get_tokenized_text(sample, tokenizer, max_context_length):

    raw_sent = sample['text']
    raw_tags = sample['tags']
    raw_mask = sample['global_attn_mask']

    tokenized_sentence = []
    tags = []
    global_attn_mask = []

    # tokenize and preserve labels
    for word, label, mask in zip(raw_sent, raw_tags, raw_mask):

        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        tokenized_sentence.extend(tokenized_word)
        tags.extend([label] * n_subwords)
        global_attn_mask.extend([mask] * n_subwords)

    token_ids = tokenizer.convert_tokens_to_ids(tokenized_sentence)
    attn_mask = [1]*len(token_ids)

    if len(token_ids) > max_context_length:
        token_ids = token_ids[:max_context_length]
        tags = tags[:max_context_length]
        global_attn_mask = global_attn_mask[:max_context_length]
        attn_mask = attn_mask[:max_context_length]
    else:
        padding = [0] * (max_context_length - len(token_ids))
        tag_padding = ['O'] * (max_context_length - len(token_ids))
        token_ids += padding
        global_attn_mask += padding
        attn_mask += padding
        tags += tag_padding

    return {
        'ids': token_ids,
        'tags': tags,
        'attn_mask': attn_mask,
        'global_attn_mask': global_attn_mask
    }


def process_conll_data(
    samples,
    tokenizer,
    max_context_length=512,
    tag2id=None,
    silent=False
):
    processed_samples = []

    if tag2id is None:
        tag_types = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"]
        tag2id = {tag:i for i,tag in enumerate(tag_types)}
    
    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    for idx, sample in enumerate(iter_):
        # if no mentions in the document, skip
        if sum(sample['global_attn_mask'])==0:
            continue

        context_tokens = get_tokenized_text(
            sample, tokenizer, max_context_length
        )
        context_tokens['ner_tags'] = [tag2id[tag] for tag in context_tokens['tags']]

        processed_samples.append(context_tokens)

    context_vecs = torch.tensor(select_field(processed_samples, 'ids'), dtype=torch.long)
    ner_tag_vecs = torch.tensor(select_field(processed_samples, 'ner_tags'), dtype=torch.long)
    mask_vecs = torch.tensor(select_field(processed_samples, 'attn_mask'), dtype=torch.bool)
    global_attn_mask_vecs = torch.tensor(select_field(processed_samples, 'global_attn_mask'), dtype=torch.bool)

    return (context_vecs, ner_tag_vecs, mask_vecs, global_attn_mask_vecs)


