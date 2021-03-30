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

def f1_score(true_pos, pred_pos, total_pos, epsilon=1e-7):
    precision = true_pos/(pred_pos+epsilon)
    recall = true_pos/(total_pos+epsilon)
    #F1 = 2./(1./precision+1./recall+epsilon)
    F1 = 2*(precision*recall) / (precision+recall+epsilon)
    return precision, recall, F1

#def evaluate(ranker, valid_dataloader, params, device, pad_id=0):
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
    start_id = params['b_tag']
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
                loss, tags_pred, _ = ranker(
                    token_ids, attn_mask, global_attn_mask, tags, b_tag=start_id,
                    golden_cand_enc=cand_enc, golden_cand_mask=cand_enc_mask
                )
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
            precision_start, recall_start, f1_start = f1_score(true_positive_start, predicted_positive_start, total_positive_start)
            
            if end_tag:
                predicted_positive_end += (mask * tags_pred.eq(end_id)).float().sum().item()
                true_positive_end += (mask * tags.eq(end_id) * tags_pred.eq(end_id)).float().sum().item()
                total_positive_end += (mask * tags.eq(end_id)).float().sum().item()
                precision_end, recall_end, f1_end = f1_score(true_positive_end, predicted_positive_end, total_positive_end)
            
    res = {
        'acc': correct/total,
        'start_tag': [precision_start, recall_start, f1_start, predicted_positive_start, true_positive_start, total_positive_start],
    }
    if end_tag:
        res['end_tag'] = [precision_end, recall_end, f1_end, predicted_positive_end, true_positive_end, total_positive_end]

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
    ranker = LongEncoderRanker(params)
    tokenizer = ranker.tokenizer
    model = ranker.model

    device = ranker.device

    # todo: optimizer
    optim = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    model_path = params.get('model_path', None)
    if model_path is not None:
        checkpoint = torch.load(model_path)
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = params['epochs']
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

            torch.nn.utils.clip_grad_norm(
                model.parameters(), params['max_grad_norm']
            )
            optim.step()
            optim.zero_grad()

        # optim.step()
        # optim.zero_grad()

        #if epoch%3==0 or epoch==(epochs-1):
        res = evaluate(ranker, valid_dataloader, params, device)
        print (f'Epoch: {epoch} Epoch Loss: {running_loss/total:.4f} Validation acc: {res["acc"]:.4f}')
        metrics = res['start_tag']
        print(f'Start tag metrics: precision {metrics[0]:.4f}, recall {metrics[1]:.4f}, F1 {metrics[2]:.4f}')
        print(f'Pred start: {metrics[3]}, True start: {metrics[4]}, Total start: {metrics[5]}')
        model.train()
        # save model
        # epoch_output_folder_path = os.path.join(
        #     model_output_path, f'epoch_{epoch}'
        # )
        epoch_output_folder_path = os.path.join(model_output_path, 'last_epoch')
        utils.save_state_dict(model, optim, epoch_output_folder_path)

    # res = evaluate(ranker, valid_dataloader, params, device)
    # print (f'Epoch: {epoch} Epoch Loss: {running_loss/total:.4f} Validation acc: {res[0]:.4f}')
    # print(f'Pred start: {res[1]}, True start: {res[2]}, Total start: {res[3]}')

if __name__ == '__main__':
    parser = Parser()
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
