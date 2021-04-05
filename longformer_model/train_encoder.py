import os
import argparse
import torch
import json
import sys
import io

from tqdm import tqdm, trange
from transformers import LongformerModel, LongformerTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from longformer_encoder import LongEncoderRanker
from data_process import read_dataset, process_mention_data
from params import Parser
import utils
from evaluate_encoder import ner_eval, in_batch_el_eval

#def evaluate(ranker, valid_dataloader, params, device, pad_id=0):
#def evaluate(ranker, valid_dataloader, params, device, pad_id=-1):
# def evaluate(ranker, valid_dataloader, params, device):
#     ranker.model.eval()

#     if params['silent']:
#         iter_ = valid_dataloader
#     else:
#         iter_ = tqdm(valid_dataloader)

#     b_tag = params['b_tag']
#     y_true = []
#     y_pred = []
    
#     for batch in iter_:
#         batch = tuple(t.to(device) for t in batch)

#         # todo: add cand_enc
#         token_ids = batch[0]
#         tags = batch[1]
#         attn_mask = batch[-2]
#         global_attn_mask = batch[-1]
#         if params['is_biencoder']:
#             cand_enc = batch[2]
#             cand_enc_mask = batch[3]

#         with torch.no_grad():
#             if params['is_biencoder']:
#                 loss, tags_pred, _ = ranker(
#                     token_ids, attn_mask, global_attn_mask, tags, b_tag=b_tag,
#                     golden_cand_enc=cand_enc, golden_cand_mask=cand_enc_mask
#                 )
#             else:
#                 loss, tags_pred, _ = ranker(token_ids, attn_mask, global_attn_mask, tags)

#         y_true.extend(tags[attn_mask].cpu().tolist())
#         y_pred.extend(tags_pred[attn_mask].cpu().tolist())
#         assert len(y_true)==len(y_pred)

#     acc, precision_b, recall_b, f1_b, f1_macro, f1_micro = utils.get_metrics_result(y_true, y_pred, b_tag)

#     return (acc, precision_b, recall_b, f1_b, f1_macro, f1_micro)


def main(params):
    train_batch_size = params['train_batch_size']
    eval_batch_size = params['eval_batch_size']
    model_output_path = params.get('output_path')
    if model_output_path is None:
        data_path = params['data_path'].split('/')[-2]
        #model_output_path = f'experiments/{data_path}_{params["train_batch_size"]}_{params["eval_batch_size"]}_{params["is_biencoder"]}_{params["not_use_golden_tags"]}/'
        model_used = 'long' if params['use_longformer'] else 'bert'
        model_output_path = f'experiments/{data_path}/{params["max_context_length"]}_{train_batch_size}_{eval_batch_size}_{model_used}_{params["is_biencoder"]}_{params["not_use_golden_tags"]}_{params["classifier"]}/'
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    print('Model saved to: ', model_output_path)

    # init model
    ranker = LongEncoderRanker(params)
    tokenizer = ranker.tokenizer
    model = ranker.model

    device = ranker.device

    start_epoch = 0

    optim = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    model_path = params.get('model_path', None)
    if model_path is not None:
        # load optim state
        model_name = params.get('model_name')
        checkpoint = torch.load(model_path+model_name)
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        # load last epoch
        with open(os.path.join(model_path, 'training_params.json')) as f:
            prev_params = json.load(f)
        start_epoch = prev_params['epochs']

    epochs = params['epochs']+start_epoch
    params['epochs'] = epochs
    b_tag = params['b_tag']

    # prepare data
    # load train and validate data
    train_samples = read_dataset(params['data_path'], 'train')
    valid_samples = read_dataset(params['data_path'], 'dev')

    # todo: delete later
    if params['debug']:
        train_samples = train_samples[:100]
        valid_samples = valid_samples[:50]

    cand_enc_path = os.path.join(params['data_path'], 'train_enc.json')
    train_tensor_data = process_mention_data(
        train_samples,
        tokenizer,
        max_context_length=params['max_context_length'],
        silent=params['silent'],
        end_tag=params['end_tag'],
        is_biencoder=params['is_biencoder'],
        cand_enc_path=cand_enc_path,
        use_longformer=params['use_longformer']
    )
    train_tensor_data = TensorDataset(*train_tensor_data)
    train_sampler = SequentialSampler(train_tensor_data)
    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
    )

    cand_enc_path = os.path.join(params['data_path'], 'dev_enc.json')
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

    with open(os.path.join(model_output_path, 'training_params.json'), 'w') as outf:
        json.dump(params, outf)

    model.train()

    for epoch in range(start_epoch, epochs):
        total = 0
        running_loss = 0.0

        if params['silent']:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader)

        for batch in iter_:
            #model.zero_grad()

            batch = tuple(t.to(device) for t in batch)
            
            # todo: add cand_enc
            token_ids = batch[0]
            tags = batch[1]
            attn_mask = batch[-2]
            global_attn_mask = batch[-1]
            cand_enc = cand_enc_mask = None
            if params['is_biencoder']:
                cand_enc = batch[2]
                cand_enc_mask = batch[3]
            loss, _, _ = ranker(
                token_ids, attn_mask, global_attn_mask, tags, 
                b_tag=b_tag,
                golden_cand_enc=cand_enc,
                golden_cand_mask=cand_enc_mask
            )
            
            # Perform backpropagation
            #(loss/token_ids.size(1)).backward()
            loss.backward()
            
            #optim.step()
            
            total += 1
            running_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), params['max_grad_norm']
            )
            optim.step()
            optim.zero_grad()

        # optim.step()
        # optim.zero_grad()

        # evaluate on valid_dataloader
        print (f'Epoch: {epoch} Epoch Loss: {running_loss/total:.4f}')
        if params['silent']:
            iter_ = valid_dataloader
        else:
            iter_ = tqdm(valid_dataloader)
        ner_eval(ranker, iter_, params, device)
        if params['is_biencoder']:
            in_batch_el_eval(ranker, iter_, params, device)

        model.train()

        epoch_output_folder_path = os.path.join(model_output_path, 'last_epoch')
        utils.save_state_dict(model, optim, epoch_output_folder_path)

if __name__ == '__main__':
    parser = Parser()
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
