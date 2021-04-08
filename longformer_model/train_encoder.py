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
from data_process_conll import process_conll_data
from params import Parser
import utils
from evaluate_encoder import ner_eval, in_batch_el_eval, cand_set_eval


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
    device = ranker.device

    start_epoch = 0
    
    model_path = params.get('model_path', None)
    if model_path is not None:
        model_name = params.get('model_name')
        checkpoint = torch.load(os.path.join(model_path, model_name), map_location=device)
        # load model state
        ranker.model.load_state_dict(checkpoint['model_state_dict'])
        model = ranker.model
        # load optim state
        optim = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        # load last epoch
        with open(os.path.join(model_path, 'training_params.json')) as f:
            prev_params = json.load(f)
        start_epoch = prev_params['epochs']
    else:
        model = ranker.model
        optim = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

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

    if params['conll']:
        train_tensor_data = process_conll_data(
            train_samples,
            tokenizer,
            max_context_length=params['max_context_length'],
            silent=params['silent']
        )

        valid_tensor_data = process_conll_data(
            valid_samples,
            tokenizer,
            max_context_length=params['max_context_length'],
            silent=params['silent']
        )
    else:
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
    
    train_tensor_data = TensorDataset(*train_tensor_data)
    if params['shuffle']:
        train_sampler = RandomSampler(train_tensor_data)
    else:
        train_sampler = SequentialSampler(train_tensor_data)
    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
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

            cand_enc = cand_enc_mask = label_ids = label_mask = None
            if params['is_biencoder']:
                token_ids, tags, cand_enc, cand_enc_mask, label_ids, label_mask, attn_mask, global_attn_mask = batch
            else:
                token_ids, tags, attn_mask, global_attn_mask = batch

            loss, _, _ = ranker(
                token_ids, attn_mask, global_attn_mask, tags, 
                b_tag=b_tag,
                golden_cand_enc=cand_enc,
                golden_cand_mask=cand_enc_mask,
                label_ids=label_ids,
                label_mask=label_mask
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
        if params['silent']:
            iter_ = valid_dataloader
        else:
            iter_ = tqdm(valid_dataloader)
        print (f'Epoch: {epoch} Epoch Loss: {running_loss/total:.4f}')
        ner_eval(ranker, iter_, params, device)
        if params['is_biencoder']:
            # in batch negative
            #in_batch_el_eval(ranker, iter_, params, device)
            # eval against all entities in train, dev, and test
            cand_set_enc = torch.load(params['selected_set_path'], map_location=device)
            id2label = torch.load(params['id_to_label_path'], map_location=device)
            cand_set_eval(ranker, iter_, params, device, cand_set_enc, id2label)

        model.train()

        epoch_output_folder_path = os.path.join(model_output_path, 'last_epoch')
        utils.save_state_dict(model, optim, epoch_output_folder_path)

if __name__ == '__main__':
    parser = Parser()
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
