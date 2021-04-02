import os
import argparse
import torch
import json
import sys
import io

from tqdm import tqdm, trange
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from bert_tagger import BertEncoderRanker
from params import Parser
import utils

def f1_score(true_pos, pred_pos, total_pos, epsilon=1e-7):
    precision = true_pos/(pred_pos+epsilon)
    recall = true_pos/(total_pos+epsilon)
    F1 = 2*(precision*recall) / (precision+recall+epsilon)
    return precision, recall, F1

def evaluate(ranker, valid_dataloader, params, device, pad_id=-1):
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
    start_id = 0 # id for B-PER

    for batch in iter_:
        batch = tuple(t.to(device) for t in batch)

        token_ids, attn_mask, tags = batch

        with torch.no_grad():
            loss, tags_pred, _ = ranker(token_ids, attn_mask, tags)

        tags = tags.cpu()#.numpy()
        tags_pred = tags_pred.cpu()#.numpy()
        mask = attn_mask.cpu()

        cor = (tags == tags_pred)[mask]
        correct += cor.float().sum().item()
        total += mask.float().sum().item()
        
        predicted_positive_start += (mask * tags_pred.eq(start_id)).float().sum().item()
        true_positive_start += (mask * tags.eq(start_id) * tags_pred.eq(start_id)).float().sum().item()
        total_positive_start += (mask * tags.eq(start_id)).float().sum().item()
    
    precision_start, recall_start, f1_start = f1_score(true_positive_start, predicted_positive_start, total_positive_start)
            
    res = {
        'acc': correct/total,
        'start_tag': [precision_start, recall_start, f1_start, predicted_positive_start, true_positive_start, total_positive_start],
    }

    return res


def main(params):
    train_batch_size = params['train_batch_size']
    eval_batch_size = params['eval_batch_size']
    model_output_path = params.get('output_path')
    if model_output_path is None:
        data_path = params['data_path'].split('/')[-2]
        #model_output_path = f'experiments/{data_path}_{params["train_batch_size"]}_{params["eval_batch_size"]}_{params["is_biencoder"]}_{params["not_use_golden_tags"]}/'
        model_used = 'long' if params['use_longformer'] else 'bert'
        model_output_path = f'experiments/{data_path}/{train_batch_size}_{eval_batch_size}_{model_used}_{params["is_biencoder"]}_{params["not_use_golden_tags"]}_{params["classifier"]}/'
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    print('Model saved to: ', model_output_path)

    # init model
    ranker = BertEncoderRanker(params)
    tokenizer = ranker.tokenizer
    model = ranker.model

    device = ranker.device

    optim = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    model_path = params.get('model_path', None)
    if model_path is not None:
        checkpoint = torch.load(model_path+'last_epoch')
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = params['epochs']
    #b_tag = params['b_tag']

    # prepare data
    train_dataloader, valid_dataloader = utils.load_and_prepare_data(
        tokenizer, train_batch_size, eval_batch_size, max_len=params['max_context_length']
    )

    utils.write_to_file(
        os.path.join(model_output_path, 'training_params.txt'), str(params)
    )

    model.train()

    for epoch in range(epochs):
        total = 0
        running_loss = 0.0

        if params['silent']:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader)

        for batch in iter_:
            #model.zero_grad()

            batch = tuple(t.to(device) for t in batch)
            token_ids, attn_mask, tags = batch

            loss, _, _ = ranker(
                token_ids, attn_mask, tags
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

        res = evaluate(ranker, valid_dataloader, params, device)
        print (f'Epoch: {epoch} Epoch Loss: {running_loss/total:.4f} Validation acc: {res["acc"]:.4f}')
        metrics = res['start_tag']
        print(f'Start tag metrics: precision {metrics[0]:.4f}, recall {metrics[1]:.4f}, F1 {metrics[2]:.4f}')
        print(f'Pred start: {metrics[3]}, True start: {metrics[4]}, Total start: {metrics[5]}')
        model.train()
        epoch_output_folder_path = os.path.join(model_output_path, 'last_epoch')
        utils.save_state_dict(model, optim, epoch_output_folder_path)

if __name__ == '__main__':
    parser = Parser()
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
