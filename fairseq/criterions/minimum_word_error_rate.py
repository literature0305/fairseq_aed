# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def minimum_word_error_rate_loss(lprobs_positive, lprobs_negative, target_positive, target_negative, ignore_index=None):
    # get mask
    pad_mask_positive = target_positive.eq(ignore_index)
    pad_mask_negative = target_negative.eq(ignore_index)

    # masking
    lprobs_positive = lprobs_positive * (1-pad_mask_positive.to(lprobs_positive.dtype)).unsqueeze(-1)
    lprobs_negative = lprobs_negative * (1-pad_mask_negative.to(lprobs_negative.dtype)).unsqueeze(-1)
    # print('target_positive:', target_positive.size())
    # print('lprobs_positive:', lprobs_positive.size())
    # print('target_negative:', target_negative.size())
    # print('lprobs_negative:', lprobs_negative.size())
    # target_positive: torch.Size([2592])
    # lprobs_positive: torch.Size([2592, 300])
    # target_negative: torch.Size([32, 82])
    # lprobs_negative: torch.Size([32, 82, 300])
    # raise ValueError('here')
    # get NLL
    nll_loss_positive = -lprobs_positive.gather(dim=-1, index=target_positive.unsqueeze(-1)).squeeze(-1)
    nll_loss_negative = -lprobs_negative.gather(dim=-1, index=target_negative.unsqueeze(-1)).squeeze(-1)
    # nll_loss_positive = -lprobs_positive.gather(dim=-1, index=target_positive)
    # nll_loss_negative = -lprobs_negative.gather(dim=-1, index=target_negative)

    # contrastive loss v1
    # loss_contrastive = nll_loss_positive.sum() - nll_loss_negative.sum()

    # contrastive loss v2
    loss_mask = nll_loss_negative.sum(-1) < nll_loss_positive.view(len(nll_loss_negative), -1).sum(-1)
    loss_contrastive = nll_loss_negative.sum(-1) * loss_mask.to(nll_loss_negative.dtype).to(nll_loss_negative.device)
    loss_contrastive = loss_contrastive.sum()
    # print('nll_loss_positive:', nll_loss_positive.sum())
    # print('nll_loss_negative:', nll_loss_negative.sum())

    return loss_contrastive

def get_one_best_denom(prev_output_tokens, sample, padding_idx):
    # get BOS and EOS
    bos = int(prev_output_tokens[0][0].item())
    bos_str = str(bos) + ' '
    eos_str = ' ' + str(padding_idx)

    # get denominator term
    prev_output_tokens_denom_list = []
    max_sequence_len = len(prev_output_tokens[0])
    for i in range(prev_output_tokens.size(0)):
        # i: batch index
        prev_output_token_str = prev_output_tokens[i].tolist()
        prev_output_token_str = map(str, prev_output_token_str)
        prev_output_token_str = ' '.join(prev_output_token_str).replace(bos_str, '').replace(eos_str,'') # remove BOS and EOS
        there_is_no_denom = True
        for j in range(len(sample['hypos'][i])):
            # j: N-best hypothesis index
            prev_output_token_denom_str = sample['hypos'][i][j]['tokens'].tolist()
            prev_output_token_denom_str = map(str, prev_output_token_denom_str)
            prev_output_token_denom_str = ' '.join(prev_output_token_denom_str).replace(bos_str, '').replace(eos_str,'') # remove BOS and EOS
            if prev_output_token_str != prev_output_token_denom_str:
                # return 1-best hypothesis that is different from the true previous output tokens
                prev_output_tokens_denom_list.append(sample['hypos'][i][j]['tokens'])
                there_is_no_denom = False
                if max_sequence_len < len(sample['hypos'][i][j]['tokens']):
                    max_sequence_len = len(sample['hypos'][i][j]['tokens'])
                break
        if there_is_no_denom:
            # if there is no hypothesis that is different from the true previous output tokens return the true previous output tokens
            prev_output_tokens_denom_list.append(prev_output_tokens[i][1:])

    # previous output tokens (denominator term)
    prev_output_tokens_denom_target = torch.ones(len(prev_output_tokens), max_sequence_len + 1).to(prev_output_tokens.device, dtype=prev_output_tokens.dtype).fill_(padding_idx)
    prev_output_tokens_denom = torch.ones(len(prev_output_tokens), max_sequence_len + 1).to(prev_output_tokens.device, dtype=prev_output_tokens.dtype).fill_(padding_idx)
    prev_output_tokens_denom[:,0] = prev_output_tokens[:,0] # BOS

    assert len(prev_output_tokens_denom_list) == len(prev_output_tokens)

    for i in range(len(prev_output_tokens_denom_list)):
        length = len(prev_output_tokens_denom_list[i])
        prev_output_tokens_denom[i][1:length+1] = prev_output_tokens_denom_list[i]
        prev_output_tokens_denom_target[i][:length] = prev_output_tokens_denom_list[i]

    return prev_output_tokens_denom, prev_output_tokens_denom_target

@register_criterion(
    "minimum_word_error_rate", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class MinimumWordErrorRate(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # get beam search decoding results
        # if 'hypos' in sample:
        #     print('# sample:', len(sample['target']))
        #     print('hypos:',len(sample['hypos']), sample['hypos'])
        #     for i in range(len(sample['hypos'])):
        #         print('hypos[',i,']:', len(sample['hypos'][i]))
        #     raise ValueError('sample MWER:', sample)

        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        if 'hypos' in sample:
            denom_prev, denom_target = get_one_best_denom(prev_output_tokens, sample, self.padding_idx)

        # for i in range(len(prev_output_tokens)):
        #     print('prev_output_tokens:', prev_output_tokens[i])
        #     print('denom:', denom[i])
        # print('prev_output_tokens:', prev_output_tokens[0])
        # for hyp_tokens in sample['hypos'][0]:
        #     print('hyp_tokens:', hyp_tokens['tokens'])
        #     print('hyp_score:', hyp_tokens['score'])
        # print('hyp:',sample['hypos'][0])
        # for i in range(len(sample)):
        #     print('denom_target:', denom_target[i])
        #     print('target:', sample['target'][i])
        # raise ValueError('here')

        encoder_output = model.encoder(src_tokens, src_lengths)
        net_output = model.decoder(prev_output_tokens, encoder_output)
        if 'hypos' in sample:
            net_output_denom = model.decoder(denom_prev, encoder_output)
            loss_mwer = self.compute_mwer_loss(model, net_output, net_output_denom, sample, {'target': denom_target})
        else:
            loss_mwer = 0

        # net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        # print('loss:', loss, 'nll_loss:', nll_loss)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss + loss_mwer, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample, unfold=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        if unfold:
            return lprobs.view(-1, lprobs.size(-1)), target.view(-1)
        else:
            return lprobs, target

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss
    
    def compute_mwer_loss(self, model, net_output_true, net_output_hyp, sample_true, sample_hyp):
        lprobs_negative, target_negative = self.get_lprobs_and_target(model, net_output_hyp, sample_hyp, unfold=False)
        lprobs_positive, target_positive = self.get_lprobs_and_target(model, net_output_true, sample_true)
        loss = minimum_word_error_rate_loss(
            lprobs_positive,
            lprobs_negative,
            target_positive,
            target_negative,
            ignore_index=self.padding_idx,
        )
        return loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
