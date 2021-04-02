import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertTagger(nn.Module):
    def __init__(self, output_dim, num_tags):
        super(BertTagger, self).__init__()
        self.classifier = nn.Linear(output_dim, num_tags)

    def forward(self, bert_output):
        logits = self.classifier(bert_output)
        tags = torch.argmax(logits, 2)
        return logits, tags

class BertEncoderModule(nn.Module):
    def __init__(self, params):
        super(BertEncoderModule, self).__init__()
        self.params = params
        self.ctxt_encoder = BertModel.from_pretrained('bert-base-cased')
        #self.NULL_IDX = 0
        output_dim = self.ctxt_encoder.embeddings.word_embeddings.weight.size(1)
        self.label_types = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']
        num_tags = len(self.label_types)
        #self.config = self.ctxt_encoder.config
        self.tagger = BertTagger(output_dim, num_tags)

    def get_raw_ctxt_encoding(
        self,
        token_idx_ctxt,
        mask_ctxt,
        global_attn_mask_ctxt=None
    ):
        outputs = self.ctxt_encoder(
            input_ids=token_idx_ctxt,
            token_type_ids=None,
            attention_mask=mask_ctxt
        )
        # (bsz, max_context_length, longformer_output_dim)
        try:
            raw_ctxt_encoding = outputs.last_hidden_state
        except:
            raw_ctxt_encoding = outputs[0]
        return raw_ctxt_encoding

    def get_ctxt_logits_tags(
        self,
        raw_ctxt_encoding=None,
        token_idx_ctxt=None,
        mask_ctxt=None,
        global_attn_mask_ctxt=None,
        
    ):
        if raw_ctxt_encoding is None:
            raw_ctxt_encoding = self.get_raw_ctxt_encoding(token_idx_ctxt, mask_ctxt)

        return self.tagger(raw_ctxt_encoding)


    def forward(
        self,
        token_idx_ctxt,
        mask_ctxt,
        global_attn_mask_ctxt=None
    ):
        """"""
        raw_ctxt_encoding = self.get_raw_ctxt_encoding(token_idx_ctxt, mask_ctxt)
        ctxt_logits, ctxt_tags = self.get_ctxt_logits_tags(raw_ctxt_encoding)
        ctxt_outs = {
            'ctxt_logits': ctxt_logits,
            'ctxt_tags': ctxt_tags
        }
        return ctxt_outs

class BertEncoderRanker(nn.Module):
    def __init__(self, params):
        super(BertEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count()
        self.label_types = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']
        self.num_tags = len(self.label_types)

        # init tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        #self.pad_id = -1
        # init model
        self.model = BertEncoderModule(self.params)
        model_path = params.get('model_path', None)
        if model_path is not None:
            checkpoint = torch.load(model_path+'last_epoch')
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)

    # todo: add mask (because pad_id = O = 8)
    def score_tagger(
        self,
        ctxt_logits,
        golden_tags
    ):
        loss_function = nn.CrossEntropyLoss(reduction='mean')
        tag_loss = loss_function(ctxt_logits.view(-1,self.num_tags), golden_tags.view(-1))

        return tag_loss


    def forward(
        self,
        token_idx_ctxt,
        mask_ctxt,
        golden_tags,
        global_attn_mask_ctxt=None
    ):
        ctxt_outs = self.model(
            token_idx_ctxt, mask_ctxt
        )
        ctxt_tags = ctxt_outs['ctxt_tags']
        ctxt_logits = ctxt_outs['ctxt_logits']
        # mask
        loss = self.score_tagger(ctxt_logits[mask_ctxt], golden_tags[mask_ctxt])

        return loss, ctxt_tags, ctxt_logits
