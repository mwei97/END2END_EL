import os
import argparse
import torch

from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from longformer_encoder import LongEncoderRanker
from data_process import read_dataset, process_mention_data
from params import EvalParser
import utils

def ner_eval(ranker, valid_dataloader, params, device):
    if params['silent']:
        iter_ = valid_dataloader
    else:
        iter_ = tqdm(valid_dataloader)

    pos_tag = params['pos_tag']
    y_true = []
    y_pred = []

    for batch in iter_:
        batch = tuple(t.to(device) for t in batch)
        token_ids, tags, attn_mask, global_attn_mask = batch

        with torch.no_grad():
            loss, tags_pred, _ = ranker(token_ids, attn_mask, global_attn_mask, tags)

        y_true.extend(tags[attn_mask].cpu().tolist())
        y_pred.extend(tags_pred[attn_mask].cpu().tolist())
        assert len(y_true)==len(y_pred)

    acc, precision, recall, f1, f1_macro, f1_micro = utils.get_metrics_result(y_true, y_pred, pos_tag)

    # print result
    print(f'Test accuracy: {res[0]:.4f}, F1 macro: {res[4]:.4f}, F1 micro: {res[5]:.4f}')
    print(f'Tag to investigate is {pos_tag}, metrics: precision {res[1]:.4f}, recall {res[2]:.4f}, F1 {res[3]:.4f}')


def in_batch_el_eval():

def kb_el_eval():



def main(params):
    assert params['model_path'] is not None

    # load model
    ranker = LongEncoderRanker(params)
    tokenizer = ranker.tokenizer
    model = ranker.model
    # load optimizer
    optim = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])
    model_name = params.get('model_name', 'last_epoch')
    checkpoint = torch.load(model_path+model_name)
    optim.load_state_dict(checkpoint['optimizer_state_dict'])

    device = ranker.device

    # load data
    eval_batch_size = params['eval_batch_size']
    valid_samples = read_dataset(params['data_path'], 'test')
    #cand_enc_path = os.path.join(params['data_path'], 'test_enc.json')
    valid_tensor_data = process_mention_data(
        valid_samples,
        tokenizer,
        max_context_length=params['max_context_length'],
        silent=params['silent'],
        end_tag=params['end_tag'],
        #is_biencoder=params['is_biencoder'],
        is_biencoder=False,
        #cand_enc_path=cand_enc_path,
        cand_enc_path=None,
        use_longformer=params['use_longformer']
    )
    valid_tensor_data = TensorDataset(*valid_tensor_data)
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    model.eval()

    if params['ner_eval']:
        print('-----Start evaluating NER task-----')
        ner_eval() # todo: add arguments

    if params['in_batch_el_eval']:
        print('-----Start evaluating EL task in batch-----')
        in_batch_el_eval() # todo: add arguments

    if params['kb_el_eval']:
        print('-----Start evaluating EL task in knowledge base-----')
        kb_el_eval() # todo: add argument


if __name__ == '__main__':
    parser = EvalParser()
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)