import torch
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import os

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

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

class SentenceGetter(object):
    def __init__(self, data_path, tag2idx):

        """
        Constructs a list of lists for sentences and labels
        from the data_path passed to SentenceGetter.
        We can then access sentences using the .sents
        attribute, and labels using .labels.
        """

        with open(data_path) as f:
            if "ru" in data_path:
                txt = f.read().split("\n\n")
            else:
                txt = f.read().split("\n \n")

        self.sents_raw = [(sent.split("\n")) for sent in txt]
        self.sents = []
        self.labels = []

        for sent in self.sents_raw:
            tok_lab_pairs = [pair.split() for pair in sent]

            # Handles (very rare) formatting issue causing IndexErrors
            try:
                toks = [pair[0] for pair in tok_lab_pairs]
                labs = [pair[1] for pair in tok_lab_pairs]

                # In the Russian data, a few invalid labels such as '-' were produced
                # by the spaCy preprocessing. Because of that, we generate a mask to
                # check if there are any invalid labels in the sequence, and if there
                # are, we reindex `toks` and `labs` to exclude them.
                mask = [False if l not in tag2idx else True for l in labs]
                if any(mask):
                    toks = list(np.array(toks)[mask])
                    labs = list(np.array(labs)[mask])

            except IndexError:
                continue

            self.sents.append(toks)
            self.labels.append(labs)

        print(f"Constructed SentenceGetter with {len(self.sents)} examples.")

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):

    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


class BertDataset:
    def __init__(self, sg, tokenizer, max_len, tag2idx):

        """
        Takes care of the tokenization and ID-conversion steps
        for prepping data for BERT.
        Takes a SentenceGetter (sg) initialized on the data you
        want to use as argument.
        """

        pad_tok = tokenizer.vocab["[PAD]"]
        sep_tok = tokenizer.vocab["[SEP]"]
        o_lab = tag2idx["O"]

        # Tokenize the text into subwords in a label-preserving way
        tokenized_texts = [
            tokenize_and_preserve_labels(sent, labs, tokenizer)
            for sent, labs in zip(sg.sents, sg.labels)
        ]

        self.toks = [["[CLS]"] + text[0] for text in tokenized_texts]
        self.labs = [["O"] + text[1] for text in tokenized_texts]

        # Convert tokens to IDs
        self.input_ids = pad_sequences(
            [tokenizer.convert_tokens_to_ids(txt) for txt in self.toks],
            maxlen=max_len,
            dtype="long",
            truncating="post",
            padding="post",
        )

        # Convert tags to IDs
        self.tags = pad_sequences(
            [[tag2idx.get(l) for l in lab] for lab in self.labs],
            maxlen=max_len,
            value=tag2idx["O"],
            padding="post",
            dtype="long",
            truncating="post",
        )

        # Swaps out the final token-label pair for ([SEP], O)
        # for any sequences that reach the MAX_LEN
        for voc_ids, tag_ids in zip(self.input_ids, self.tags):
            if voc_ids[-1] == pad_tok:
                continue
            else:
                voc_ids[-1] = sep_tok
                tag_ids[-1] = o_lab

        # Place a mask (zero) over the padding tokens
        self.attn_masks = [[float(i > 0) for i in ii] for ii in self.input_ids]


def load_and_prepare_data(tokenizer, train_batch_size, eval_batch_size, max_len=75):

    label_types = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']
    tag2idx = {t: i for i, t in enumerate(label_types)}

    #tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    train_data_path = 'data/train_combined_std.txt'
    dev_data_path = 'data/dev_combined_std.txt'

    getter_train = SentenceGetter(train_data_path, tag2idx)
    getter_dev = SentenceGetter(dev_data_path, tag2idx)
    train = BertDataset(getter_train, tokenizer, max_len, tag2idx)
    dev = BertDataset(getter_dev, tokenizer, max_len, tag2idx)

    # Input IDs (tokens), tags (label IDs), attention masks
    tr_inputs = torch.tensor(train.input_ids)
    val_inputs = torch.tensor(dev.input_ids)
    tr_tags = torch.tensor(train.tags)
    val_tags = torch.tensor(dev.tags)
    tr_masks = torch.tensor(train.attn_masks)
    val_masks = torch.tensor(dev.attn_masks)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=train_batch_size
    )

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(
        valid_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    return train_dataloader, valid_dataloader


