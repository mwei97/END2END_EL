import os
import argparse
import torch
import json

from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from entity_linker import LongEntityLinker
from data_process import read_dataset, process_mention_data
from params import EvalParser
import utils

def cand_set_eval(ranker, valid_dataloader, params, device, cand_set_enc, id2label):
    ranker.model.eval()
    y_true = []
    y_pred = []

    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)

        token_ids, tags, cand_enc, cand_enc_mask, label_ids, label_mask, attn_mask, global_attn_mask = batch

        with torch.no_grad():
            raw_ctxt_encoding = ranker.model.get_raw_ctxt_encoding(token_ids, attn_mask)
            ctxt_embeds = ranker.model.get_ctxt_embeds(raw_ctxt_encoding, tags)

        scores = ctxt_embeds.mm(cand_set_enc.t())

        true_labels = label_ids[label_mask].cpu().tolist()
        y_true.extend(true_labels)
        pred_inds = torch.argmax(scores, dim=1).cpu().tolist()
        pred_labels = [id2label[i].item() for i in pred_inds]
        y_pred.extend(pred_labels)
        assert len(y_true)==len(y_pred)
    
    acc, f1_macro, f1_micro = utils.get_metrics_result(y_true, y_pred)
    print(f'Accuracy: {acc:.4f}, F1 macro: {f1_macro:.4f}, F1 micro: {f1_micro:.4f}')

def kb_el_eval(ranker, valid_dataloader, params, device, all_cand_enc):
    ranker.model.eval()
    y_true = []
    y_pred = []

    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        assert params['is_biencoder']

        token_ids, tags, cand_enc, cand_enc_mask, label_ids, label_mask, attn_mask, global_attn_mask = batch

        with torch.no_grad():
            raw_ctxt_encoding = ranker.model.get_raw_ctxt_encoding(token_ids, attn_mask)
            ctxt_embeds = ranker.model.get_ctxt_embeds(raw_ctxt_encoding, tags)

        scores = ctxt_embeds.mm(all_cand_enc.t())

        true_labels = label_ids[label_mask].cpu().tolist()
        y_true.extend(true_labels)
        pred_labels = torch.argmax(scores, dim=1).cpu().tolist()
        y_pred.extend(pred_labels)
        assert len(y_true)==len(y_pred)

    acc, f1_macro, f1_micro = utils.get_metrics_result(y_true, y_pred)
    print(f'Accuracy: {acc:.4f}, F1 macro: {f1_macro:.4f}, F1 micro: {f1_micro:.4f}')


def main(params):
    assert params['model_path'] is not None
    model_path = params['model_path']

    # init model
    ranker = LongEntityLinker(params)
    tokenizer = ranker.tokenizer
    device = ranker.device
    
    model_name = params['model_name']
    checkpoint = torch.load(os.path.join(model_path, model_name), map_location=device)
    # load model
    ranker.model.load_state_dict(checkpoint['model_state_dict'])
    model = ranker.model
    # load optimizer
    optim = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])

    # load data
    eval_batch_size = params['eval_batch_size']
    valid_samples = read_dataset(params['data_path'], params['split'])
    if params['conll']:
        valid_tensor_data = process_conll_data(
            valid_samples,
            tokenizer,
            max_context_length=params['max_context_length'],
            silent=params['silent']
        )
    else:
        cand_enc_path = os.path.join(params['data_path'], f'{params["split"]}_enc.json')
        valid_tensor_data = process_mention_data(
            valid_samples,
            tokenizer,
            max_context_length=params['max_context_length'],
            silent=params['silent'],
            end_tag=params['end_tag'],
            is_biencoder=params['is_biencoder'],
            cand_enc_path=cand_enc_path,
            use_longformer=params['use_longformer']
        )
    valid_tensor_data = TensorDataset(*valid_tensor_data)
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    if params['silent']:
        iter_ = valid_dataloader
    else:
        iter_ = tqdm(valid_dataloader)

    model.eval()

    if params['cand_set_eval']:
        print('-----Start evaluating EL task in selected candidate set-----')
        cand_set_enc = torch.load(params['selected_set_path'], map_location=device)
        id2label = torch.load(params['id_to_label_path'], map_location=device)
        cand_set_eval(ranker, iter_, params, device, cand_set_enc, id2label)

    if params['kb_el_eval']:
        print('-----Start evaluating EL task in knowledge base-----')
        all_cand_enc = torch.load(params['all_cand_path'], map_location=device)
        kb_el_eval(ranker, iter_, params, device, all_cand_enc)


if __name__ == '__main__':
    parser = EvalParser()
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    params['use_longformer'] = not params['use_bert']
    params['use_golden_tags'] = True
    params['is_biencoder'] = True
    main(params)
