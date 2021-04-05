import os
import argparse
import torch

from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from longformer_encoder import LongEncoderRanker
from data_process import read_dataset, process_mention_data
from params import EvalParser
import utils

def ner_eval(ranker, valid_dataloader, params, device, pos_tag=1):
    ranker.model.eval()
    #pos_tag = params['pos_tag']
    y_true = []
    y_pred = []

    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        cand_enc = cand_enc_mask = None
        if params['is_biencoder']:
            token_ids, tags, cand_enc, cand_enc_mask, attn_mask, global_attn_mask = batch
        else:
            token_ids, tags, attn_mask, global_attn_mask = batch
        
        with torch.no_grad():
            _, tags_pred, _ = ranker(
                token_ids, attn_mask, global_attn_mask, tags,
                golden_cand_enc=cand_enc, golden_cand_mask=cand_enc_mask
            )

        y_true.extend(tags[attn_mask].cpu().tolist())
        y_pred.extend(tags_pred[attn_mask].cpu().tolist())
        assert len(y_true)==len(y_pred)

    acc, precision, recall, f1, f1_macro, f1_micro = utils.get_metrics_result(y_true, y_pred, pos_tag)

    # print result
    print(f'Accuracy: {acc:.4f}, F1 macro: {f1_macro:.4f}, F1 micro: {f1_micro:.4f}')
    print(f'Tag to investigate is {pos_tag}, metrics: precision {precision:.4f}, recall {recall:.4f}, F1 {f1:.4f}')


def in_batch_el_eval(ranker, valid_dataloader, params, device):
    ranker.model.eval()

    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        assert params['is_biencoder']

        total = corr = 0

        token_ids, tags, cand_enc, cand_enc_mask, attn_mask, global_attn_mask = batch

        with torch.no_grad():
            raw_ctxt_encoding = ranker.model.get_raw_ctxt_encoding(token_ids, attn_mask, global_attn_mask)
            # get ctxt embeds for golden b tags
            ctxt_embeds = ranker.model.get_ctxt_embeds(raw_ctxt_encoding, tags)
        
        # get similarity scores
        cand_enc = cand_enc[cand_enc_mask]
        scores = ctxt_embeds.mm(cand_enc.t())

        pred_labels = torch.argmax(scores, dim=1).cpu()
        true_labels = torch.LongTensor(torch.arange(scores.size(1)))
        total += pred_labels.size(0)
        corr += sum(true_labels==pred_labels).item()

    # do I need F1 scores for EL?
    acc = corr / total * 1.

    print(f'Test in batch accuracy for EL is: {acc:.4f}')


#def kb_el_eval():



def main(params):
    assert params['model_path'] is not None
    model_path = params['model_path']

    if params['in_batch_el_eval'] or params['kb_el_eval']:
        params['is_biencoder'] = True
    else:
        params['is_biencoder'] = False

    # load model
    ranker = LongEncoderRanker(params)
    tokenizer = ranker.tokenizer
    model = ranker.model
    # load optimizer
    optim = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])
    model_name = params.get('model_name')
    if model_name is None:
        model_name = 'last_epoch'
    checkpoint = torch.load(model_path+model_name)
    optim.load_state_dict(checkpoint['optimizer_state_dict'])

    device = ranker.device

    # load data
    eval_batch_size = params['eval_batch_size']
    valid_samples = read_dataset(params['data_path'], 'test')
    cand_enc_path = os.path.join(params['data_path'], 'test_enc.json')
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

    if params['ner_eval']:
        print('-----Start evaluating NER task-----')
        ner_eval(ranker, iter_, params, device)

    if params['in_batch_el_eval']:
        print('-----Start evaluating EL task in batch-----')
        in_batch_el_eval(ranker, iter_, params, device)

    if params['kb_el_eval']:
        print('-----Start evaluating EL task in knowledge base-----')
        kb_el_eval() # todo: add argument


if __name__ == '__main__':
    parser = EvalParser()
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
