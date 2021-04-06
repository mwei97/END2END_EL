import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerTokenizer
from transformers import BertModel, BertTokenizer

class LongTagger(nn.Module):
    def __init__(self, longformer_output_dim, num_tags, classifier='linear'):
        super(LongTagger, self).__init__()
        if classifier=='linear':
            self.classifier = nn.Linear(longformer_output_dim, num_tags)
        elif classifier=='nonlinear': # todo: better nonlinear model?
            self.classifier = nn.Sequential(
                nn.Linear(longformer_output_dim, longformer_output_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(longformer_output_dim, num_tags)
            )
        else:
            raise NotImplementedError()
    
    def forward(self, transformer_output):
        # (bsz, max_context_length, num_tags)
        logits = self.classifier(transformer_output)
        # (bsz, max_context_length)
        tags = torch.argmax(logits, 2)
        return logits, tags

class LongEncoderModule(nn.Module):
    def __init__(self, params):
        super(LongEncoderModule, self).__init__()
        self.params = params
        if params['use_longformer']:
            self.ctxt_encoder = LongformerModel.from_pretrained('allenai/longformer-base-4096')
            longformer_output_dim = self.ctxt_encoder.embeddings.word_embeddings.weight.size(1)
            self.NULL_IDX = 0
        else:
            self.ctxt_encoder = BertModel.from_pretrained('bert-base-uncased')
            self.NULL_IDX = 0
            longformer_output_dim = self.ctxt_encoder.embeddings.word_embeddings.weight.size(1)
        #num_tags = 4 if not self.params['end_tag'] else 5
        #num_tags = 3 if not self.params['end_tag'] else 4
        num_tags = 9 if self.params['conll'] else 3
        self.config = self.ctxt_encoder.config
        self.tagger = LongTagger(longformer_output_dim, num_tags, self.params['classifier'])
        self.linear_compression = None
        if longformer_output_dim != self.params['cand_emb_dim']:
            self.linear_compression = nn.Linear(longformer_output_dim, self.params['cand_emb_dim'])

    def get_raw_ctxt_encoding(
        self,
        token_idx_ctxt,
        mask_ctxt,
        global_attn_mask_ctxt
    ):
        if self.params['use_longformer']:
            longformer_outputs = self.ctxt_encoder(
                token_idx_ctxt, attention_mask=mask_ctxt, global_attention_mask=global_attn_mask_ctxt
            )
        else:
            #token_idx_ctxt, segment_idx_ctxt, _ = to_bert_input(token_idx_ctxt, self.NULL_IDX)
            segment_idx_ctxt = token_idx_ctxt*0
            longformer_outputs = self.ctxt_encoder(
                input_ids=token_idx_ctxt, attention_mask=mask_ctxt, token_type_ids=segment_idx_ctxt, 
            )
        # (bsz, max_context_length, longformer_output_dim)
        try:
            raw_ctxt_encoding = longformer_outputs.last_hidden_state
        except:
            raw_ctxt_encoding = longformer_outputs[0]
        return raw_ctxt_encoding

    def get_ctxt_logits_tags(
        self,
        raw_ctxt_encoding=None,
        token_idx_ctxt=None,
        mask_ctxt=None,
        global_attn_mask_ctxt=None,
        
    ):
        if raw_ctxt_encoding is None:
            raw_ctxt_encoding = self.get_raw_ctxt_encoding(token_idx_ctxt, mask_ctxt, global_attn_mask_ctxt)

        return self.tagger(raw_ctxt_encoding)

    def get_ctxt_embeds(
        self,
        raw_ctxt_encoding,
        tags,
        golden_tags=None,
        b_tag=1
    ):
        """
            Get embeddings of B tags
            tags could be pred tags or golden tags
            If self.linear_compression, match embeddings to candidate entity embeds dimension 
        """
        #b_tag = 1 #2
        # (bsz, max_context_length)
        mask = (tags==b_tag)
        if torch.sum(mask).cpu().item()==0: # no pred b tag
            #mask = (tags==golden_tags)
            mask = (golden_tags==b_tag)
        # (num_b_tags, longformer_output_dim)
        ctxt_embeds = raw_ctxt_encoding[mask]
        if self.linear_compression is not None:
            # (num_b_tags, cand_emb_dim)
            ctxt_embeds = self.linear_compression(ctxt_embeds)
        return ctxt_embeds

    def forward(
        self,
        token_idx_ctxt,
        mask_ctxt,
        global_attn_mask_ctxt,
        is_biencoder=False,
        use_golden_tags=False,
        golden_tags=None,
        b_tag=1
    ):
        """"""
        raw_ctxt_encoding = self.get_raw_ctxt_encoding(token_idx_ctxt, mask_ctxt, global_attn_mask_ctxt)
        ctxt_logits, ctxt_tags = self.get_ctxt_logits_tags(raw_ctxt_encoding)
        ctxt_outs = {
            'ctxt_logits': ctxt_logits,
            'ctxt_tags': ctxt_tags
        }
        if is_biencoder:
            # use golden tags to get context embeddings
            if use_golden_tags:
                assert golden_tags is not None
                ctxt_embeds = self.get_ctxt_embeds(raw_ctxt_encoding, golden_tags, b_tag=b_tag)
            # use pred tags to get context embeddings
            else:
                ctxt_embeds = self.get_ctxt_embeds(raw_ctxt_encoding, ctxt_tags, golden_tags=golden_tags, b_tag=b_tag) # if pred no B tags, use golden_tags
            ctxt_outs['ctxt_embeds'] = ctxt_embeds
        return ctxt_outs


class LongEncoderRanker(nn.Module):
    def __init__(self, params):
        super(LongEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count() # todo
        #self.num_tags = 4 if not self.params['end_tag'] else 5
        #self.num_tags = 3 if not self.params['end_tag'] else 4
        self.num_tags = 9 if self.params['conll'] else 3
        self.is_biencoder = params['is_biencoder']
        self.use_golden_tags = not params['not_use_golden_tags']
        # init tokenizer
        if params['use_longformer']:
            self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.pad_id = 0
        self.pad_id = -1
        # init model
        self.model = LongEncoderModule(self.params)
        model_path = params.get('model_path', None)
        if model_path is not None:
            model_name = params.get('model_name')
            checkpoint = torch.load(model_path+model_name)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        # # set model parallel
        # self.data_parallel = params.get('data_parallel')
        # if self.data_parallel:
        #     self.model = nn.DataParallel(self.model)

    # todo: add mask
    def score_tagger(
        self,
        ctxt_logits,
        golden_tags
    ):
        #loss_function = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.pad_id)
        loss_function = nn.CrossEntropyLoss(reduction='mean')
        tag_loss = loss_function(ctxt_logits.view(-1,self.num_tags), golden_tags.view(-1))

        return tag_loss

    # # todo
    def score_candidate(
        self,
        golden_cand_enc,
        golden_cand_mask,
        ctxt_embeds,
        use_golden_tags=True,
        golden_tags=None,
        pred_tags=None
    ):
        loss_function = nn.CrossEntropyLoss(reduction='mean')

        if use_golden_tags:
            # (num_golden_entities, cand_emb_dim)
            golden_cand_enc = golden_cand_enc[golden_cand_mask]
            # (num_b_tags, num_golden_entities)
            scores = ctxt_embeds.mm(golden_cand_enc.t())
        else:
            cand_enc = torch.zeros(ctxt_embeds.size()).to(self.device)
            golden_cand_enc = golden_cand_enc[golden_cand_mask]
            golden_tags = golden_tags.view(-1)
            pred_tags = pred_tags.view(-1)

            enc_order = 0
            golden_enc_order = 0
            for i in range(golden_tags.size(0)):
                if pred_tags[i]==1:
                    if golden_tags[i]==1:
                        cand_enc[enc_order] = golden_cand_enc[golden_enc_order]
                        golden_enc_order += 1
                    enc_order += 1
                elif golden_tags[i]==1:
                    golden_enc_order += 1
                else:
                    continue
            scores = ctxt_embeds.mm(cand_enc.t())

        target = torch.LongTensor(torch.arange(scores.size(1))).to(self.device)
        cand_loss = loss_function(scores, target)
        return cand_loss, scores

            # # todo: modify loss when pred tag > golden & pred tag < golden
            # if scores.size(0)==scores.size(1):
            #     target = torch.LongTensor(torch.arange(scores.size(1))).to(self.device)
            #     cand_loss = loss_function(scores, target)
            # elif scores.size(0)<scores.size(1):
            #     target = torch.LongTensor(torch.arange(scores.size(0))).to(self.device)
            #     cand_loss = loss_function(scores, target)
            # else:
            #     target = torch.LongTensor(torch.arange(scores.size(1))).to(self.device)
            #     cand_loss = loss_function(scores[:scores.size(1)], target)

    def forward(
        self,
        token_idx_ctxt,
        mask_ctxt,
        global_attn_mask_ctxt,
        golden_tags,
        b_tag=1,
        golden_cand_enc=None,
        golden_cand_mask=None
    ):
        ctxt_outs = self.model(
            token_idx_ctxt, mask_ctxt, global_attn_mask_ctxt,
            is_biencoder=self.is_biencoder,
            use_golden_tags=self.use_golden_tags,
            golden_tags=golden_tags, b_tag=b_tag
        )
        ctxt_tags = ctxt_outs['ctxt_tags']
        ctxt_logits = ctxt_outs['ctxt_logits']
        loss = self.score_tagger(ctxt_logits[mask_ctxt], golden_tags[mask_ctxt])
        if self.is_biencoder:
            ctxt_embeds = ctxt_outs['ctxt_embeds']
            cand_loss, _ = self.score_candidate(
                golden_cand_enc, golden_cand_mask, ctxt_embeds,
                use_golden_tags=self.use_golden_tags,
                golden_tags=golden_tags,
                pred_tags=ctxt_tags
            )
            loss += cand_loss

        return loss, ctxt_tags, ctxt_logits

# def to_bert_input(token_idx, null_idx):
#     segment_idx = token_idx * 0
#     mask = token_idx != null_idx
#     # nullify elements in case self.NULL_IDX was not 0
#     token_idx = token_idx * mask.long()
#     return token_idx, segment_idx, mask

