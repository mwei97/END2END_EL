import argparse

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_args()

    def add_args(self, args=None):
        parser = self.add_argument_group('Common Arguments')
        parser.add_argument(
            '--silent', default=False, action='store_true'
        )
        parser.add_argument(
            '--debug',
            default=False,
            action='store_true'
        )
        parser.add_argument(
            '--b_tag',
            default=1,
            type=int
        )
        parser.add_argument(
            '--end_tag', default=False, action='store_true'
        )
        parser.add_argument(
            '--classifier',
            default='linear',
            type=str
        )
        parser.add_argument(
            '--cand_emb_dim',
            default=1024,
            type=int
        )
        parser.add_argument(
            '--learning_rate',
            default=3e-5,
            type=float
        )
        parser.add_argument('--max_grad_norm', default=1.0, type=float)
        parser.add_argument(
            '--use_longformer', 
            default=False,
            action='store_true'
        )
        parser.add_argument(
            '--is_biencoder', default=False, action='store_true'
        )
        parser.add_argument(
            '--not_use_golden_tags', default=False, action='store_true')
        parser.add_argument(
            '--max_context_length',
            default=512,
            type=int
        )
        parser.add_argument(
            '--train_batch_size',
            default=16,
            type=int
        )
        parser.add_argument(
            '--eval_batch_size',
            default=16,
            type=int
        )
        parser.add_argument(
            '--shuffle',
            default=False,
            action='store_true'
        )
        parser.add_argument(
            '--epochs',
            default=3,
            type=int
        )
        parser.add_argument(
            '--data_path',
            default='../data/AIDA-YAGO2_longformer/tokenized',
            type=str
        )
        parser.add_argument(
            '--output_path',
            type=str,
            required=False
        )
        parser.add_argument(
            '--model_path',
            type=str,
            required=False
        )
        parser.add_argument(
            '--model_name',
            default='last_epoch',
            type=str,
            required=False
        )
        parser.add_argument(
            '--conll',
            default=False,
            action='store_true'
        )
        parser.add_argument(
            '--selected_set_path',
            default='../models/el_candidate_set.t7',
            type=str
        )
        parser.add_argument(
            '--id_to_label_path',
            default='../models/candidate_set_id2label.json',
            type=str
        )

class EvalParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_args()

    def add_args(self, args=None):
        parser = self.add_argument_group('Common Arguments')
        parser.add_argument(
            '--ner_eval',
            default=False,
            action='store_true'
        )
        parser.add_argument(
            '--in_batch_el_eval',
            default=False,
            action='store_true'
        )
        parser.add_argument(
            '--cand_set_eval',
            default=False,
            action='store_true'
        )
        parser.add_argument(
            '--kb_el_eval',
            default=False,
            action='store_true'
        )
        parser.add_argument(
            '--silent', default=False, action='store_true'
        )
        parser.add_argument(
            '--debug',
            default=False,
            action='store_true'
        )
        parser.add_argument(
            '--pos_tag',
            default=1,
            type=int
        )
        parser.add_argument(
            '--end_tag', default=False, action='store_true'
        )
        parser.add_argument(
            '--classifier',
            default='linear',
            type=str
        )
        parser.add_argument(
            '--cand_emb_dim',
            default=1024,
            type=int
        )
        parser.add_argument(
            '--learning_rate',
            default=3e-5,
            type=float
        )
        parser.add_argument('--max_grad_norm', default=1.0, type=float)
        parser.add_argument(
            '--use_longformer', 
            default=False,
            action='store_true'
        )
        # parser.add_argument(
        #     '--is_biencoder', default=False, action='store_true'
        # )
        parser.add_argument(
            '--not_use_golden_tags', default=False, action='store_true')
        parser.add_argument(
            '--max_context_length',
            default=512,
            type=int
        )
        parser.add_argument(
            '--eval_batch_size',
            default=16,
            type=int
        )
        parser.add_argument(
            '--data_path',
            default='../data/AIDA-YAGO2_longformer/tokenized',
            type=str
        )
        parser.add_argument(
            '--output_path',
            type=str,
            required=False
        )
        parser.add_argument(
            '--model_path',
            type=str,
            required=False
        )
        parser.add_argument(
            '--model_name',
            default='last_epoch',
            type=str,
            required=False
        )
        parser.add_argument(
            '--all_cand_path',
            default='../models/all_entities_large.t7',
            type=str
        )
        parser.add_argument(
            '--selected_set_path',
            default='../models/el_candidate_set.t7',
            type=str
        )
        parser.add_argument(
            '--id_to_label_path',
            default='../models/candidate_set_id2label.json',
            type=str
        )
        parser.add_argument(
            '--conll',
            default=False,
            action='store_true'
        )
