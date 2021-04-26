import os
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

def get_metrics_result(y_true, y_pred, b_tag=None):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    if b_tag is not None:
        precision_b = precision_score(y_true, y_pred, labels=[b_tag], average='micro')
        recall_b = recall_score(y_true, y_pred, labels=[b_tag], average='micro')
        f1_b = f1_score(y_true, y_pred, labels=[b_tag], average='micro')
        return (acc, precision_b, recall_b, f1_b, f1_macro, f1_micro)
    else:
        return (acc, f1_macro, f1_micro)

def write_to_file(path, string, mode="w"):
    with open(path, mode) as writer:
        writer.write(string)

def save_model(model, tokenizer, output_dir):
    """Saves the model and the tokenizer used in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

def save_state_dict(model, optimizer, output_dir):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_dir)

def get_mention_boundaries(pred_tag_ids, idx2tag=None):
    """
    Input:
        B-I-O tag indices
    Return:
        List of mention boundaries (inclusive)
    """
    if idx2tag is None:
        idx2tag = {0: 'O', 1: 'B', 2: 'I'}

    pred_tags = [idx2tag[i] for i in pred_tag_ids]
    mentions = []
    open_mention = False

    for i, tag in enumerate(pred_tags):
        if tag=='B':
            # ...BB...
            if open_mention:
                mentions[-1].extend([i-1])
                mentions.append([i])
            # start of new mention
            else:
                mentions.append([i])
                open_mention = True
        elif tag=='I':
            continue
        else:
            if open_mention:
                mentions[-1].extend([i-1])
                open_mention = False
            else:
                continue

    if len(mentions)>0:
        if len(mentions[-1])==1:
            mentions[-1].extend([i])

    return mentions
