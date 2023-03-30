# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import json
import itertools
import logging
import os
import math
import sys

import torch
import numpy as np
import xmlrpc.client
import sacrebleu
import pandas as pd
from sklearn import preprocessing

from fairseq import metrics, options, utils, bleu
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)

from fairseq.tasks import FairseqTask, register_task
from fairseq.data.data_utils import collate_tokens


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset, eos=eos
    )


@register_task('translation')
class TranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenizer before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='if setting, we compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        
        # some choices: ['bleu', 'bertscore', 'bleurt', 'comet', 'bartscore', 'unite_ref', 'unite_src_ref'] ['bleurt&bertscore', 'bleu&comet', ...]  可单个，可组合
        parser.add_argument('--sample-metric', type=str, default=None,
                            help='metric to score samples')
        parser.add_argument('--sample-metric-weights', type=str, metavar='JSON', default=None,
                            help='weight of sample metrics')
        parser.add_argument('--sample-self', action='store_true',
                            help='add self to sampling list')
        parser.add_argument('--sample-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--combine-bleu-bleurt', action='store_true',
                            help='bleurt_score += bleu_score')

        parser.add_argument('--finetune-metric', action='store_true',
                            help='mrt & finetune_metrics at the same time')
        parser.add_argument('--finetune-metric-weights', type=str, metavar='JSON', default=None,
                            help='weight of finetune metrics')
        parser.add_argument('--finetune-metric-threshold', type=float, default=0.5,
                            help='finetune with samples < threshold')
        parser.add_argument('--finetune-metric-lr', type=float, default=3e-5,
                            help='finetune metric learning rate')
        parser.add_argument('--finetune-data-path', type=str, default=None,
                            help='finetune data save path')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_lang = args.source_lang
        self.tgt_lang = args.target_lang
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        import sys
        self.sys_modules = sys.modules
        self.sample_metric_list = []
        if getattr(args, 'sample_metric', False):
            self.sample_metric_list = args.sample_metric.split('#')
        self.scorer_dict = dict()
        if 'bleu' in self.sample_metric_list:
            self.scorer_dict['bleu'] = bleu.Scorer(pad=tgt_dict.pad_index, eos=tgt_dict.eos_index, unk=tgt_dict.eos_index)
        if 'bertscore' in self.sample_metric_list or getattr(args, 'finetune_metric', False):
            import sys
            sys.path.append("bert_score")
            from bert_score import BERTScorer
            self.scorer_dict['bertscore'] = BERTScorer(lang=args.target_lang, rescale_with_baseline=True)
        if 'bleurt' in self.sample_metric_list or getattr(args, 'finetune_metric', False):
            logger.info("init bluert rpc client")
            self.scorer_dict['bleurt'] = xmlrpc.client.ServerProxy('http://localhost:8888')
        if 'comet' in self.sample_metric_list or getattr(args, 'finetune_metric', False):
            import sys
            sys.path.append("COMET_mello")
            from comet import load_from_checkpoint
            comet_model_path = './COMET_mello/checkpoints/wmt20-comet-da.ckpt'
            self.scorer_dict['comet'] = load_from_checkpoint(comet_model_path)
        if 'bartscore' in self.sample_metric_list or getattr(args, 'finetune_metric', False):
            from BARTScore.bart_score import BARTScorer
            bart_scorer = BARTScorer(device=self.args.device_id, lang=self.src_lang+'-'+self.tgt_lang)
            #bart_scorer.load(path='../bart.pth')  # TODO 后续修改路径，成为超参
            self.scorer_dict['bartscore'] = bart_scorer
        if 'unite' in '#'.join(self.sample_metric_list) or getattr(args, 'finetune_metric', False):
            import sys
            sys.path.append('UniTE_mello')
            from unite_comet.models.regression.regression_metric import RegressionMetric
            model_prefix = 'UniTE-models/UniTE-MUP/'
            model_path = model_prefix + 'checkpoints/UniTE-MUP.ckpt'
            hparams_ref_path = model_prefix + 'hparams.ref.yaml'
            import yaml
            with open(hparams_ref_path) as yaml_file:
                hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
                if 'unite_src_ref' in '#'.join(self.sample_metric_list): hparams['input_segments'] = ['hyp', 'src', 'ref']
            self.scorer_dict['unite'] = RegressionMetric.load_from_checkpoint(model_path, **hparams)
            self.scorer_dict['unite'].eval()

        if getattr(args, 'sample_metric_weights', False):
            self.sample_weights = json.loads(getattr(args, 'sample_metric_weights', '{}') or '{}')
        if getattr(args, 'finetune_metric_weights', False):
            self.finetune_metric_weights = json.loads(getattr(self.args, 'finetune_metric_weights', '{}') or '{}')

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        if getattr(args, 'sample_metric', False):
            sample_args = json.loads(getattr(args, 'sample_bleu_args', '{}') or '{}')
            self.sample_generator = self.build_generator([model], Namespace(**sample_args))
        return model

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False, old_sample=None,
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        
        loss, sample_size, logging_output = criterion(model, sample, old_sample=old_sample)
        if ignore_grad:
            loss *= 0
        
        if self.args.sample_metric and 'comet' in self.args.sample_metric:
            torch.use_deterministic_algorithms(True, warn_only=True)
        optimizer.backward(loss)

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        # 这里调用是单卡
        # 节省资源
        # valid interval 控制多久
        
        if self.args.eval_bleu:
            bleu, metric_score = self._inference_with_bleu(self.sequence_generator, sample, model)
            # print(bleu)
            # print(metric_score)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
            if self.args.sample_metric:
                logging_output['_' + self.args.sample_metric + '_score'] = metric_score
            # if self.args.sample_metric == 'bertscore': logging_output['_bertscore_score'] = metric_score
            # elif self.args.sample_metric == 'bleurt': logging_output['_bleurt_score'] = metric_score
            # elif self.args.sample_metric == 'comet': logging_output['_comet_score'] = metric_score
            # elif self.args.sample_metric == 'bartscore': logging_output['_bartscore_score'] = metric_score
            # elif self.args.sample_metric == 'unite_ref': logging_output['_unite_ref_score'] = metric_score
            # elif self.args.sample_metric == 'unite_src_ref': logging_output['_unite_src_ref_score'] = metric_score
            # print(logging_output)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if self.args.eval_bleu:

            def sum_logs(key):
                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result
                # return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)

        metric_name = ""
        metric_name_whole = ""
        for name in logging_outputs[0].keys():
            if '_score' in name:
                metric_name_whole = name
                metric_name = '#'.join(self.sample_metric_list)
        if len(metric_name) and metric_name != 'bleu':
            metric_score = sum(log.get(metric_name_whole, 0) for log in logging_outputs) / len(logging_outputs) / self.args.distributed_world_size
            metrics.log_scalar(metric_name, metric_score)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False, src=False):
            
            if not src:
                s = self.tgt_dict.string(
                    toks.int().cpu(),
                    self.args.eval_bleu_remove_bpe,
                    escape_unk=escape_unk,
                )
            else:
                s = self.src_dict.string(
                    toks.int().cpu(),
                    self.args.eval_bleu_remove_bpe,
                    escape_unk=escape_unk,
                )
            
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        # print('ssssssssssssssssssssssssssssssample')
        # print(sample)
        # print(gen_out)
        hyps, refs, srcs = [], [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            # print('============hypo_str')
            # print(gen_out[i][0]['tokens'])
            # print(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
            srcs.append(decode(sample['net_input']['src_tokens'][i], escape_unk=True, src=True))

        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        elif self.args.target_lang in ['zh']:
            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize='zh')
        else:
            bleu = sacrebleu.corpus_bleu(hyps, [refs])

        # print("================= srcs, hyps, refs, ")
        # print(srcs)
        # print(hyps)
        # print(refs)  # list of sent of batch_size
        # metrics
        metric_score = 0
        if 'bleu' in self.sample_metric_list:
            if hasattr(self, 'sample_weights'):
                metric_score += bleu.score * self.sample_weights['bleu']
            else:
                metric_score += bleu.score
        if 'bertscore' in self.sample_metric_list:
            metric_scores = self.get_cost_bertscore(hyps, refs)
            metric_scores = [100. - score for score in metric_scores]
            if hasattr(self, 'sample_weights'):
                metric_score += sum(metric_scores) / len(metric_scores) * self.sample_weights['bertscore']
            else:
                metric_score += sum(metric_scores) / len(metric_scores)
        if 'bartscore' in self.sample_metric_list:
            if self.tgt_lang == 'en':
                metric_scores = self.get_cost_bartscore(hyps, refs, tgt_lang=self.tgt_lang)
            else:
                metric_scores = self.get_cost_bartscore(srcs, hyps, tgt_lang=self.tgt_lang)
            metric_scores = [-score for score in metric_scores]   # 从costs还原
            if hasattr(self, 'sample_weights'):
                metric_score += sum(metric_scores) / len(metric_scores) * self.sample_weights['bartscore']
            else:
                metric_score += sum(metric_scores) / len(metric_scores)
        if 'bleurt' in self.sample_metric_list:
            metric_scores = self.get_cost_bleurt(hyps, refs)
            metric_scores = [100. - score for score in metric_scores]
            print(metric_scores)  # 512  -8到几十不等, 20就非常稳定6.70
            if hasattr(self, 'sample_weights'):
                metric_score += sum(metric_scores) / len(metric_scores) * self.sample_weights['bleurt']
            else:
                metric_score += sum(metric_scores) / len(metric_scores)
        if 'comet' in self.sample_metric_list:
            metric_scores = self.get_cost_comet(hyps, refs, srcs)
            metric_scores = [100. - score for score in metric_scores]
            if hasattr(self, 'sample_weights'):
                metric_score += sum(metric_scores) / len(metric_scores) * self.sample_weights['comet']
            else:
                metric_score += sum(metric_scores) / len(metric_scores)
        if 'unite' in '#'.join(self.sample_metric_list):
            metric_scores = self.get_cost_unite(hyps, refs, srcs)
            metric_scores = [100. - score for score in metric_scores]
            if hasattr(self, 'sample_weights'):
                metric_score += sum(metric_scores) / len(metric_scores) * self.sample_weights['unite']
            else:
                metric_score += sum(metric_scores) / len(metric_scores)

        return bleu, metric_score

    def get_cost_bleu(self, hypo_tensor, ref_tensor):
        try:
            # hypo, ref = self.sacrebleu_tokenizer(hypo), self.sacrebleu_tokenizer(ref)
            # scores.append(1 - sacrebleu.sentence_bleu(hypo, [ref]).score / 100.0)

            # 普通bleu计算
            self.scorer_dict['bleu'].reset(one_init=True) # in case of 0.
            self.scorer_dict['bleu'].add(ref_tensor.type(torch.IntTensor), hypo_tensor.type(torch.IntTensor))
            cost = 100. - self.scorer_dict['bleu'].score()
            # sacrebleu计算
            # pass
        except Exception as e:
            # logger.error(f"error when using sacrebleu: {e}\n{hypo}\n{ref}")
            cost = 100.
        return cost

    def get_cost_bertscore(self, hyps, refs):
        # hyps / refs : list of sent
        try:
            P, R, F1_scores = self.scorer_dict['bertscore'].score(hyps, refs)   # list of float
            costs = [100. - score * 100 for score in F1_scores]
        except Exception as e:
            costs = [100.] * len(refs)
        return costs

    def get_cost_bartscore(self, hyps, refs, tgt_lang):
        try:
            if tgt_lang == 'en':
                recall = self.scorer_dict['bartscore'].score(hyps, refs, batch_size=8)
                precision = self.scorer_dict['bartscore'].score(refs, hyps, batch_size=8)
                recall = np.array(recall)
                precision = np.array(precision)
                F1_scores = (2 * np.multiply(precision, recall) / (precision + recall))
                costs = [-score for score in F1_scores]
            else:
                srcs, hyps = hyps, refs
                faithfulness = self.scorer_dict['bartscore'].score(srcs, hyps, batch_size=8)
                faithfulness = np.array(faithfulness)
                costs = [-score for score in faithfulness]

        except Exception as e:
            costs = [100.] * len(refs)
        return costs

    def get_cost_bleurt(self, hyps, refs):
        # hyps / refs : list of sent
        try:
            scores = self.scorer_dict['bleurt'].bleurt_score((hyps, refs), self.args.device_id)   # list of float
            costs = [100. - score * 100 for score in scores]
        except Exception as e:
            costs = [100.] * len(refs)
        return costs

    def get_cost_comet(self, hyps, refs, srcs):
        # if self.args.finetune_metric:
        #     import sys
        #     sys.path.append("COMET_mello")
        #     from comet import load_from_checkpoint_wo_hparams
        #     comet_model_path = './COMET_mello/checkpoints/wmt20-comet-da.ckpt'
        #     self.scorer_dict['comet'] = load_from_checkpoint_wo_hparams(comet_model_path)
        try:
            data = {'src': srcs, 'mt': hyps, 'ref': refs}
            data = [dict(zip(data, t)) for t in zip(*data.values())]
            seg_scores = self.scorer_dict['comet'].predict_mello(data, batch_size=8, device=self.args.device_id)   # 参数怎么设置？
            costs = [100. - score * 100 for score in seg_scores]
        except Exception as e:
            costs = [100.] * len(refs)
        return costs

    def get_cost_unite(self, hyps, refs, srcs):
        try:
            data = {'src': srcs, 'mt': hyps, 'ref': refs}
            data = [dict(zip(data, t)) for t in zip(*data.values())]
            seg_scores = self.scorer_dict['unite'].predict_mello(samples=data, batch_size=8, device=self.args.device_id) 
            costs = [100. - score * 100 for score in seg_scores]
        except Exception as e:
            costs = [100.] * len(refs)
        return costs

    def generate_sample(self, sample, model):
        
        def decode(toks, escape_unk=False, src=False):
            if not src:
                s = self.tgt_dict.string(
                    toks.int().cpu(),
                    self.args.eval_bleu_remove_bpe,
                    escape_unk=escape_unk,
                )
            else:
                s = self.src_dict.string(
                    toks.int().cpu(),
                    self.args.eval_bleu_remove_bpe,
                    escape_unk=escape_unk,
                )
            if self.tokenizer:
                # print("ssssssssself.tokenizer")
                # print(self)  # <fairseq.tasks.translation.TranslationTask object at 0x7fe8544da780>
                # print(self.tokenizer)  # <fairseq.data.encoders.moses_tokenizer.MosesTokenizer object at 0x7fe7ef367eb8>
                s = self.tokenizer.decode(s)
            return s

        def minmax(iterable):
            _min = min(iterable)
            res = [x - _min for x in iterable]
            _max = max(res)
            return [x / (1e-4 +_max) for x in res]

        gen_out = self.inference_step(self.sample_generator, [model], sample, None)
        srcs, hyps, refs, ids = [], [], [], []
        unique_hyps = set()
        new_input, new_target = [], []
        src_lengths, tgt_lengths = [], []
        costs = []

        for i in range(len(sample['target'])):   # batch_size
            gold_tokens = utils.strip_pad(sample['target'][i], self.tgt_dict.pad())
            ref = decode(gold_tokens, escape_unk=True)
            src_tokens = sample['net_input']['src_tokens'][i]
            src = decode(src_tokens, escape_unk=True, src=True)
            idx = sample['id'][i].item()
            src_length = sample['net_input']['src_lengths'][i].item()
            sub_costs = []
            sub_costs_bleu = []
            sub_tgt_lengths = []
            if self.args.sample_self: # add gold reference
                new_input.append(src_tokens)
                new_target.append(gold_tokens)
                src_lengths.append(src_length)
                hyps.append(ref)
                refs.append(ref)
                srcs.append(src)
                ids.append(idx)
                if 'bleu' in self.sample_metric_list:
                    sub_costs_bleu.append(self.get_cost_bleu(gold_tokens, gold_tokens))
                sub_tgt_lengths.append(len(gold_tokens))
            
            for j in range(len(gen_out[i])):   # beam size
                hypo_tokens = utils.strip_pad(gen_out[i][j]['tokens'], self.tgt_dict.pad())
                hypo = decode(hypo_tokens)
                # if hypo in unique_hyps or len(hypo.strip()) == 0 or hypo is None or ref is None or len(ref.strip()) == 0:
                #     continue
                unique_hyps.add(hypo)
                hyps.append(hypo)
                refs.append(ref)
                srcs.append(src)
                ids.append(idx)
                src_lengths.append(src_length)
                new_input.append(src_tokens)
                new_target.append(hypo_tokens)
                if 'bleu' in self.sample_metric_list:
                    sub_costs_bleu.append(self.get_cost_bleu(hypo_tokens, gold_tokens))
                sub_tgt_lengths.append(len(hypo_tokens))
            if 'bleu' in self.sample_metric_list:
                if hasattr(self, 'sample_weights'):
                    sub_costs.append([x * self.sample_weights['bleu'] for x in sub_costs_bleu])
                else:
                    sub_costs.append(sub_costs_bleu)
            
            if 'bertscore' in self.sample_metric_list:
                sub_costs_bertscore = self.get_cost_bertscore(hyps[-len(sub_tgt_lengths):], refs[-len(sub_tgt_lengths):])  # list of beam size
                if hasattr(self, 'sample_weights'):
                    sub_costs.append([x * self.sample_weights['bertscore'] for x in sub_costs_bertscore])
                else:
                    sub_costs.append(sub_costs_bertscore)
            if 'bartscore' in self.sample_metric_list:
                if self.tgt_lang == 'en':
                    sub_costs_bartscore = self.get_cost_bartscore(hyps[-len(sub_tgt_lengths):], refs[-len(sub_tgt_lengths):], tgt_lang=self.tgt_lang)
                else:
                    sub_costs_bartscore = self.get_cost_bartscore(srcs[-len(sub_tgt_lengths):], hyps[-len(sub_tgt_lengths):], tgt_lang=self.tgt_lang)
                if hasattr(self, 'sample_weights'):
                    sub_costs.append([x * self.sample_weights['bartscore'] for x in sub_costs_bartscore])
                else:
                    sub_costs.append(sub_costs_bartscore)
            if 'bleurt' in self.sample_metric_list:
                sub_costs_bleurt = self.get_cost_bleurt(hyps[-len(sub_tgt_lengths):], refs[-len(sub_tgt_lengths):])
                if hasattr(self, 'sample_weights'):
                    sub_costs.append([x * self.sample_weights['bleurt'] for x in sub_costs_bleurt])
                else:
                    sub_costs.append(sub_costs_bleurt)
            if 'comet' in self.sample_metric_list:
                sub_costs_comet = self.get_cost_comet(hyps[-len(sub_tgt_lengths):], refs[-len(sub_tgt_lengths):], srcs[-len(sub_tgt_lengths):])
                if hasattr(self, 'sample_weights'):
                    sub_costs.append([x * self.sample_weights['comet'] for x in sub_costs_comet])
                else:
                    sub_costs.append(sub_costs_comet)
            if 'unite' in '#'.join(self.sample_metric_list):
                sub_costs_unite = self.get_cost_unite(hyps[-len(sub_tgt_lengths):], refs[-len(sub_tgt_lengths):], srcs[-len(sub_tgt_lengths):])
                if hasattr(self, 'sample_weights'):
                    sub_costs.append([x * self.sample_weights['unite'] for x in sub_costs_unite])
                else:
                    sub_costs.append(sub_costs_unite)
            
            sub_costs = np.sum(sub_costs, axis=0).tolist()
            costs.append(minmax(sub_costs))
            tgt_lengths.append(sub_tgt_lengths)

        src_tokens = torch.stack(new_input)
        target = collate_tokens(new_target, self.tgt_dict.pad_index, self.tgt_dict.eos_index, False, False)
        prev_output_tokens = collate_tokens(new_target, self.tgt_dict.pad_index, self.tgt_dict.eos_index, False, True)

        batch = {
            'id': torch.LongTensor(ids),
            'nsentences': len(ids),
            'ntokens': sum([sum(sub_tgt_lengths) for sub_tgt_lengths in tgt_lengths]),
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': torch.LongTensor(src_lengths),
                'prev_output_tokens': prev_output_tokens,
            },
            'target': target,
            'tgt_lengths': torch.LongTensor(tgt_lengths),
            'costs': torch.FloatTensor(costs), # [bsz, sample_size]
        }

        return batch

    def generate_sample_for_finetune_metric(self, sample, model, \
            finetune_metric_threshold=10, save_path_bleurt=None):
        # 保存数据，微调指标
        
        def decode(toks, escape_unk=False, src=False):
            if not src:
                s = self.tgt_dict.string(
                    toks.int().cpu(),
                    self.args.eval_bleu_remove_bpe,
                    escape_unk=escape_unk,
                )
            else:
                s = self.src_dict.string(
                    toks.int().cpu(),
                    self.args.eval_bleu_remove_bpe,
                    escape_unk=escape_unk,
                )
            if self.tokenizer:
                # print("ssssssssself.tokenizer")
                # print(self)  # <fairseq.tasks.translation.TranslationTask object at 0x7fe8544da780>
                # print(self.tokenizer)  # <fairseq.data.encoders.moses_tokenizer.MosesTokenizer object at 0x7fe7ef367eb8>
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(self.sample_generator, [model], sample, None)
        srcs, hyps, refs, scores = [], [], [], []

        for i in range(len(sample['target'])):   # batch_size
            gold_tokens = utils.strip_pad(sample['target'][i], self.tgt_dict.pad())
            ref = decode(gold_tokens, escape_unk=True)
            src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i], self.src_dict.pad())
            src = decode(src_tokens, escape_unk=True, src=True)
            
            for j in range(len(gen_out[i])):   # beam size
                hypo_tokens = utils.strip_pad(gen_out[i][j]['tokens'], self.tgt_dict.pad())
                hypo = decode(hypo_tokens)
                # ================ 计算各种指标分数  ================ #
                score_bleu = sacrebleu.corpus_bleu([hypo], [[ref]]).score * self.finetune_metric_weights['bleu']
                P, R, F1_scores = self.scorer_dict['bertscore'].score([hypo], [ref])
                score_bertscore = F1_scores[0].item() * 100 * self.finetune_metric_weights['bertscore']
                if self.tgt_lang == 'en':
                    recall = self.scorer_dict['bartscore'].score([hypo], [ref])[0]
                    precision = self.scorer_dict['bartscore'].score([ref], [hypo])[0]
                    score_bartscore = (2 * precision * recall / (precision + recall)) * self.finetune_metric_weights['bartscore']
                else:
                    score_bartscore = self.scorer_dict['bartscore'].score([src], [hypo])[0] * self.finetune_metric_weights['bartscore']
                score_bleurt = self.scorer_dict['bleurt'].bleurt_score(([hypo], [ref]), self.args.device_id)[0] * 100 * self.finetune_metric_weights['bleurt']
                data = {'src': [src], 'mt': [hypo], 'ref': [ref]}
                data = [dict(zip(data, t)) for t in zip(*data.values())]
                score_comet = self.scorer_dict['comet'].predict_mello(data, device=self.args.device_id)[0] * 100 * self.finetune_metric_weights['comet']
                score_unite = self.scorer_dict['unite'].predict_mello(samples=data, device=self.args.device_id)[0] * 100 * self.finetune_metric_weights['unite']
                score_ensemble = score_bleu + score_bertscore + score_bartscore + score_bleurt + score_comet + score_unite

                srcs.append(src)
                hyps.append(hypo)
                refs.append(ref)
                scores.append(score_ensemble)

        data = {'src': srcs, 'mt': hyps, 'ref': refs, 'raw_score': scores}
        data = pd.DataFrame(data)
        raw_score = data.raw_score
        z_score = preprocessing.scale(raw_score)
        data['score'] = pd.DataFrame(z_score)   # 额，但是仅用一个batch数据算出来的raw score，未必准哎！
        # 筛选分值<threshold的
        if finetune_metric_threshold != -1:
            data = data.loc[data['raw_score'] < finetune_metric_threshold]
        
        # 保存微调bleurt需要的训练数据
        if save_path_bleurt:
            data_bleurt = data[['mt', 'ref', 'score']]
            data_bleurt.rename(columns={'mt':'candidate', 'ref':'reference'}, inplace=True)
            data_bleurt = data_bleurt.to_dict('records')
            with open(save_path_bleurt, 'w', encoding='utf-8') as fs:
                for d in data_bleurt:
                    fs.write(json.dumps(d, ensure_ascii=False) + '\n')
            # assert 1==2

        # 返回comet需要的训练数据
        data = data.to_dict('records')
        return data

    def update_metric(self): 
        if self.args.sample_metric == 'comet':
            import sys
            sys.path.append("COMET_mello")
            from comet import load_from_checkpoint
            comet_model_path = './COMET_mello/checkpoints/wmt20-comet-da.ckpt'
            self.scorer_dict['comet'] = load_from_checkpoint(comet_model_path, pretrained_model='./transformers/xlm-roberta-large-for-comet/')
        
        elif self.args.sample_metric == 'bleurt':
            # 重新加载bleurt的话就不能通过启服务的方式……
            self.scorer_dict['bleurt'] = xmlrpc.client.ServerProxy('http://localhost:8888')
