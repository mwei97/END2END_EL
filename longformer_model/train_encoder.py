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

#def evaluate(ranker, valid_dataloader, params, device, pad_id=0):
def evaluate(ranker, valid_dataloader, params, device, pad_id=-1, start_id=1):
    ranker.model.eval()

    if params['silent']:
        iter_ = valid_dataloader
    else:
        iter_ = tqdm(valid_dataloader)

    correct = 0
    total = 0

    true_positive_start = 0
    predicted_positive_start = 0
    total_positive_start = 0
    #start_id = 2

    end_tag = params['end_tag']
    if end_tag:
        true_positive_end = 0
        predicted_positive_end = 0
        total_positive_end = 0
        #end_id = 4
        end_id = 3

    for batch in iter_:
        batch = tuple(t.to(device) for t in batch)

        # todo: add cand_enc
        token_ids = batch[0]
        tags = batch[1]
        attn_mask = batch[-2]
        global_attn_mask = batch[-1]
        if params['is_biencoder']:
            cand_enc = batch[2]
            cand_enc_mask = batch[3]

        with torch.no_grad():
            if params['is_biencoder']:
                loss, tags_pred, _ = ranker(token_ids, attn_mask, global_attn_mask, tags, cand_enc, cand_enc_mask)
            else:
                loss, tags_pred, _ = ranker(token_ids, attn_mask, global_attn_mask, tags)

            tags = tags.cpu()#.numpy()
            tags_pred = tags_pred.cpu()#.numpy()
            mask = tags.ne(pad_id)

            cor = (tags == tags_pred)[mask]
            correct += cor.float().sum().item()
            total += mask.float().sum().item()
            
            predicted_positive_start += (mask * tags_pred.eq(start_id)).float().sum().item()
            true_positive_start += (mask * tags.eq(start_id) * tags_pred.eq(start_id)).float().sum().item()
            total_positive_start += (mask * tags.eq(start_id)).float().sum().item()
            
            if end_tag:
                predicted_positive_end += (mask * tags_pred.eq(end_id)).float().sum().item()
                true_positive_end += (mask * tags.eq(end_id) * tags_pred.eq(end_id)).float().sum().item()
                total_positive_end += (mask * tags.eq(end_id)).float().sum().item()
            
    if end_tag:
        res = (correct/total, predicted_positive_start, true_positive_start, total_positive_start, predicted_positive_end, true_positive_end, total_positive_end)
    else:
        res = (correct/total, predicted_positive_start, true_positive_start, total_positive_start)

    return res


def main(params):
    model_output_path = params.get('output_path')
    if model_output_path is None:
        data_path = params['data_path'].split('/')[-2]
        model_output_path = f'experiments/{data_path}_{params["train_batch_size"]}_{params["eval_batch_size"]}_{params["is_biencoder"]}_{params["not_use_golden_tags"]}/'
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    print(model_output_path)

    # init model
    ranker = LongEncoderRanker(params)
    tokenizer = ranker.tokenizer
    model = ranker.model

    device = ranker.device

    train_batch_size = params['train_batch_size']
    eval_batch_size = params['eval_batch_size']

    # load train_data
    train_samples = read_dataset(params['data_path'], 'train')
    # load valid data
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
        cand_enc_path=cand_enc_path
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
        cand_enc_path=cand_enc_path
    )
    valid_tensor_data = TensorDataset(*valid_tensor_data)
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    utils.write_to_file(
        os.path.join(model_output_path, 'training_params.txt'), str(params)
    )

    model.train()

    # todo: optimizer
    optim = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    epochs = params['epochs']

    for epoch in range(epochs):
        total = 0
        running_loss = 0.0

        if params['silent']:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader)

        for batch in iter_:
            model.zero_grad()

            batch = tuple(t.to(device) for t in batch)
            
            # todo: add cand_enc
            token_ids = batch[0]
            tags = batch[1]
            attn_mask = batch[-2]
            global_attn_mask = batch[-1]
            if params['is_biencoder']:
                cand_enc = batch[2]
                cand_enc_mask = batch[3]
                loss, _, _ = ranker(token_ids, attn_mask, global_attn_mask, tags, cand_enc, cand_enc_mask)
            else:
                loss, _, _ = ranker(token_ids, attn_mask, global_attn_mask, tags)
            
            # Perform backpropagation
            (loss/token_ids.size(1)).backward()
            
            optim.step()
            
            total += 1
            running_loss += loss.item()

        if epoch%3==0:
            res = evaluate(ranker, valid_dataloader, params, device)
            print (f'Epoch: {epoch} Epoch Loss: {running_loss/total:.4f} Validation acc: {res[0]:.4f}')
            print(f'Pred start: {res[1]}, True start: {res[2]}, Total start: {res[3]}')
            model.train()
        # save model
        # epoch_output_folder_path = os.path.join(
        #     model_output_path, f'epoch_{epoch}'
        # )
        epoch_output_folder_path = os.path.join(model_output_path, 'last_epoch')
        utils.save_state_dict(model, optim, epoch_output_folder_path)

    res = evaluate(ranker, valid_dataloader, params, device)
    print (f'Epoch: {epoch} Epoch Loss: {running_loss/total:.4f} Validation acc: {res[0]:.4f}')
    print(f'Pred start: {res[1]}, True start: {res[2]}, Total start: {res[3]}')

if __name__ == '__main__':
    parser = Parser()
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
