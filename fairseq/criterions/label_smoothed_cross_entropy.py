# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
# torch.autograd.set_detect_anomaly(True)

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def mrt_loss(lprobs, target, sample, epsilon, ignore_index=None):
    # 如何理解：costs是当前sample的loss，cost越大代表指标评分越差，也就是样本的loss越大
    # probs是当前sample的解码概率，用于给costs也就是loss做加权

    # print(lprobs.size()) # [batch_size*sample_size, seq_len, vocab_size]
    # print(target.size()) # [batch_size*sample_size, seq_len]
    costs = sample['costs'].type_as(lprobs)   # [batch_size, sample_size]
    bsz, sample_size = costs.shape
    tgt_lengths = sample['tgt_lengths'].type_as(lprobs) # [batch_size, sample_size] 解码句长度

    target = target.view(bsz, sample_size, -1, 1)  # [batch_size, sample_size, seq_len]
    lprobs = lprobs.view(bsz, sample_size, -1, lprobs.size(-1)) # [batch_size, sample_size, seq_len, vocab_size]
    scores = lprobs.gather(dim=-1, index=target)   # [batch_size, sample_size, seq_len, 1]
    # nll_loss = -lprobs.gather(dim=-1, index=target) # [batch_size, sample_size, seq_len, 1]
    # smooth_loss = -lprobs.sum(dim=-1, keepdim=True) # [batch_size, sample_size, seq_len, 1]

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        scores.masked_fill_(pad_mask, 0.)
        # nll_loss.masked_fill_(pad_mask, 0.)
        # smooth_loss.masked_fill_(pad_mask, 0.)

    scores = scores.squeeze(3)  # [batch_size, sample_size, seq_len]
    avg_scores = scores.sum(dim=2) / tgt_lengths # [batch_size, sample_size]
    probs = torch.nn.functional.softmax(avg_scores.exp_())
    # seq_loss = (probs * costs).sum() / bsz
    loss = (probs * costs).sum()

    # nll_loss = nll_loss.sum()
    # smooth_loss = smooth_loss.sum()
    # eps_i = epsilon / lprobs.size(-1)
    # tok_loss = ((1. - epsilon) * nll_loss + eps_i * smooth_loss) / sample['ntokens']

    # loss = 0.5 * seq_loss + 0.5 * tok_loss
    return loss, loss


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)  # 选出tgt位置
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)  # 求和
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, mrt_lambda):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.mrt_lambda = mrt_lambda

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--combine-mrt-nll', action='store_true', 
                            help='combine mrt nll loss')
        parser.add_argument('--mrt-lambda', default=1., type=float, 
                            help='mrt loss ratio')
        # fmt: on

    def forward(self, model, sample, reduce=True, old_sample=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])

        if old_sample is not None:   # 同时用mrt和nll loss指导训练
            net_output_nll = model(**old_sample['net_input'])
            self.sentence_avg = True
            loss_mrt, nll_loss = self.compute_loss_mrt(model, net_output, sample)
            loss_nll, nll_loss = self.compute_loss(model, net_output_nll, old_sample)
            loss_nll *= 0.01
            loss = loss_mrt * self.mrt_lambda + loss_nll * (1 - self.mrt_lambda)    # TODO
            sample_size = sample['costs'].size(0) if self.sentence_avg else sample['ntokens']
        elif 'costs' in sample:
            self.sentence_avg = True
            loss, nll_loss = self.compute_loss_mrt(model, net_output, sample)
            sample_size = sample['costs'].size(0) if self.sentence_avg else sample['ntokens']
            # sample_size = 1 # combined loss
        else:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    def compute_loss_mrt(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        lprobs = lprobs.to(target.device)
        loss, nll_loss = mrt_loss(
            lprobs, target, sample, self.eps, ignore_index=self.padding_idx,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
