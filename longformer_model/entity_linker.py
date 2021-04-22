import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerTokenizer
from transformers import BertModel, BertTokenizer

class LongEntityLinkerModule(nn.Module):
    def __init__(self, params):
        super(LongEntityLinkerModule, self).__init__()
        self.params = params
        if params['use_longformer']:
            self.ctxt_encoder = LongformerModel.from_pretrained('allenai/longformer-base-4096')
            longformer_output_dim = self.ctxt_encoder.embeddings.word_embeddings.weight.size(1)
            self.NULL_IDX = 0
        else:
            self.ctxt_encoder = BertModel.from_pretrained('bert-base-uncased')
            self.NULL_IDX = 0
            longformer_output_dim = self.ctxt_encoder.embeddings.word_embeddings.weight.size(1)
        self.config = self.ctxt_encoder.config
        self.linear_compression = None
        if longformer_output_dim != self.params['cand_emb_dim']:
            self.linear_compression = nn.Linear(longformer_output_dim, self.params['cand_emb_dim'])

    def get_raw_ctxt_encoding(
        self,
        token_idx_ctxt,
        mask_ctxt
    ):
        if self.params['use_longformer']:
            longformer_outputs = self.ctxt_encoder(
                token_idx_ctxt, attention_mask=mask_ctxt
            )
        else:
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

    def get_ctxt_embeds(
        self,
        raw_ctxt_encoding,
        tags,
        b_tag=1
    ):
        """
            Get embeddings of B tags
            tags could be pred tags or golden tags
            If self.linear_compression, match embeddings to candidate entity embeds dimension 
        """
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
        golden_tags,
        b_tag=1
    ):
        """"""
        raw_ctxt_encoding = self.get_raw_ctxt_encoding(token_idx_ctxt, mask_ctxt)
        assert golden_tags is not None
        ctxt_embeds = self.get_ctxt_embeds(raw_ctxt_encoding, golden_tags, b_tag=b_tag)
        ctxt_outs = {'ctxt_embeds': ctxt_embeds}
        return ctxt_outs


class LongEntityLinker(nn.Module):
    def __init__(self, params):
        super(LongEntityLinker, self).__init__()
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count()
        self.use_golden_tags = params['use_golden_tags']
        # init tokenizer
        if params['use_longformer']:
            self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.pad_id = -1
        # init model
        self.model = LongEntityLinkerModule(self.params)
        self.model = self.model.to(self.device)

    def score_candidate(
        self,
        golden_cand_enc,
        golden_cand_mask,
        ctxt_embeds,
        use_golden_tags=True,
        label_ids=None,
        label_mask=None,
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

    def forward(
        self,
        token_idx_ctxt,
        mask_ctxt,
        golden_tags,
        b_tag=1,
        golden_cand_enc=None,
        golden_cand_mask=None,
        label_ids=None,
        label_mask=None
    ):
        ctxt_outs = self.model(
            token_idx_ctxt,
            mask_ctxt,
            golden_tags=golden_tags,
            b_tag=b_tag
        )
        ctxt_embeds = ctxt_outs['ctxt_embeds']
        cand_loss, _ = self.score_candidate(
            golden_cand_enc, golden_cand_mask, ctxt_embeds, 
            use_golden_tags=self.use_golden_tags,
            label_ids=label_ids,
            label_mask=label_mask,
            golden_tags=golden_tags,
            pred_tags=ctxt_tags
        )

        return cand_loss

