import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerTokenizer

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
        self.ctxt_encoder = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        longformer_output_dim = self.ctxt_encoder.embeddings.word_embeddings.weight.size(1) # todo: confirm this line works
        #num_tags = 4 if not self.params['end_tag'] else 5
        num_tags = 3 if not self.params['end_tag'] else 4
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
        longformer_outputs = self.ctxt_encoder(
            token_idx_ctxt, attention_mask=mask_ctxt, global_attention_mask=global_attn_mask_ctxt
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
        tags
    ):
        """
            Get embeddings of B tags
            tags could be pred tags or golden tags
            If self.linear_compression, match embeddings to candidate entity embeds dimension 
        """
        #b_tag = 2
        b_tag = 1
        # (bsz, max_context_length)
        mask = (tags==b_tag)
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
        golden_tags=None
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
                ctxt_embeds = self.get_ctxt_embeds(raw_ctxt_encoding, golden_tags)
            # use pred tags to get context embeddings
            else:
                ctxt_embeds = self.get_ctxt_embeds(raw_ctxt_encoding, ctxt_tags)
            ctxt_outs['ctxt_embeds'] = ctxt_embeds
        return ctxt_outs


class LongEncoderRanker(nn.Module):
    def __init__(self, params):
        super(LongEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count() # todo
        #self.num_tags = 4 if not self.params['end_tag'] else 5
        self.num_tags = 3 if not self.params['end_tag'] else 4
        self.is_biencoder = params['is_biencoder']
        self.use_golden_tags = not params['not_use_golden_tags']
        # init tokenizer
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        #self.pad_id = 0
        self.pad_id = -1
        # init model
        self.model = LongEncoderModule(self.params)
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
        loss_function = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.pad_id)
        tag_loss = loss_function(ctxt_logits.view(-1,self.num_tags), golden_tags.view(-1))

        return tag_loss

    # # todo
    def score_candidate(
        self,
        golden_cand_enc,
        golden_cand_mask,
        ctxt_embeds,
        golden_tags=None
    ):
        # (num_golden_entities, cand_emb_dim)
        golden_cand_enc = golden_cand_enc[golden_cand_mask]
        # (num_pred_b_tags, num_golden_entities)
        scores = ctxt_embeds.mm(golden_cand_enc.t())

        loss_function = nn.CrossEntropyLoss(reduction='mean')
        # todo: modify loss when pred tag > golden & pred tag < golden
        if scores.size(0)==scores.size(1):
            target = torch.LongTensor(torch.arange(scores.size(1))).to(self.device)
            cand_loss = loss_function(scores, target)
        elif scores.size(0)<scores.size(1):
            target = torch.LongTensor(torch.arange(scores.size(0))).to(self.device)
            cand_loss = loss_function(scores, target)
        else:
            target = torch.LongTensor(torch.arange(scores.size(1))).to(self.device)
            cand_loss = loss_function(scores[:scores.size(1)], target)

        return cand_loss, scores

    def forward(
        self,
        token_idx_ctxt,
        mask_ctxt,
        global_attn_mask_ctxt,
        golden_tags,
        golden_cand_enc=None,
        golden_cand_mask=None
    ):
        ctxt_outs = self.model(
            token_idx_ctxt, mask_ctxt, global_attn_mask_ctxt,
            is_biencoder=self.is_biencoder,
            use_golden_tags=self.use_golden_tags,
            golden_tags=golden_tags
        )
        ctxt_tags = ctxt_outs['ctxt_tags']
        ctxt_logits = ctxt_outs['ctxt_logits']
        loss = self.score_tagger(ctxt_logits, golden_tags)
        if self.is_biencoder:
            ctxt_embeds = ctxt_outs['ctxt_embeds']
            cand_loss, _ = self.score_candidate(golden_cand_enc, golden_cand_mask, ctxt_embeds, golden_tags)
            loss += cand_loss

        return loss, ctxt_tags, ctxt_logits










