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
            default=1e-3,
            type=float
        )
        parser.add_argument(
            '--use_longformer', 
            default=True,
            type=bool
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
            default=8,
            type=int
        )
        parser.add_argument(
            '--eval_batch_size',
            default=4,
            type=int
        )
        parser.add_argument(
            '--epochs',
            default=3,
            type=int
        )
        parser.add_argument(
            '--data_path',
            default='END2END_EL/data/AIDA-YAGO2-wiki_content-NEW/tokenized',
            type=str
        )
        parser.add_argument(
            '--output_path',
            type=str,
            required=False)
        parser.add_argument(
            '--model_path',
            type=str,
            required=False)
