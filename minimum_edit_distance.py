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
import Levenshtein

@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    wer_threshold: float = field(
        default=0.3,
        metadata={"help": "threshold for sequence level training, 0 means no sequence level training"},
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


def get_one_best_denom(prev_output_tokens, sample, padding_idx):
    # get BOS and EOS
    bos = int(prev_output_tokens[0][0].item())
    bos_str = str(bos) + ' '
    eos_str = ' ' + str(padding_idx)

    # get denominator term
    prev_output_tokens_denom_list = []
    max_sequence_len = len(prev_output_tokens[0])
    num_no_dedum = 0
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
            
            if prev_output_token_str != prev_output_token_denom_str and Levenshtein.distance(prev_output_token_str, prev_output_token_denom_str) / len(prev_output_token_str) < self.wer_threshold:
                # return 1-best hypothesis that is different from the true previous output tokens
                prev_output_tokens_denom_list.append(sample['hypos'][i][j]['tokens'])
                there_is_no_denom = False
                if max_sequence_len < len(sample['hypos'][i][j]['tokens']):
                    max_sequence_len = len(sample['hypos'][i][j]['tokens'])
                break
        if there_is_no_denom:
            num_no_dedum += 1
            # if there is no hypothesis that is different from the true previous output tokens return the true previous output tokens
            prev_output_tokens_denom_list.append(prev_output_tokens[i][1:])

    if torch.randperm(1000)[0] == 0:
        print('num_no_dedum/len(prev):', num_no_dedum, '/', len(prev_output_tokens_denom_list))
    # previous output tokens (denominator term)
    prev_output_tokens_denom_target = torch.ones(len(prev_output_tokens), max_sequence_len + 1).to(prev_output_tokens.device, dtype=prev_output_tokens.dtype).fill_(padding_idx)
    prev_output_tokens_denom = torch.ones(len(prev_output_tokens), max_sequence_len + 1).to(prev_output_tokens.device, dtype=prev_output_tokens.dtype).fill_(padding_idx)
    prev_output_tokens_denom[:,0] = prev_output_tokens[:,0] # BOS

    assert len(prev_output_tokens_denom_list) == len(prev_output_tokens)

    for i in range(len(prev_output_tokens_denom_list)):
        length = len(prev_output_tokens_denom_list[i])
        prev_output_tokens_denom[i][1:length+1] = prev_output_tokens_denom_list[i]
        prev_output_tokens_denom_target[i][:length] = prev_output_tokens_denom_list[i]

    return prev_output_tokens_denom.to(prev_output_tokens.device), prev_output_tokens_denom_target.to(prev_output_tokens.device)


@register_criterion(
    "minimum_edit_distance", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class MinimumEditDistanceCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        wer_threshold,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.wer_threshold = wer_threshold
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        if 'hypos' in sample:
            loss_med = self
            denom_prev, denom_target = get_one_best_denom(prev_output_tokens, sample, self.padding_idx)

        encoder_output = model.encoder(src_tokens, src_lengths)
        net_output = model.decoder(prev_output_tokens, encoder_output)

        # Cross-entropy loss
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        # Minimum edit distance loss
        if 'hypos' in sample:
            net_output_denom = model.decoder(denom_prev, encoder_output)
            loss_med, nll_loss_med = self.compute_med_loss(model, net_output_denom, sample, {'target': denom_target}, denom_prev)

            if torch.randperm(1000)[0] == 0:
                print('loss_med:', loss_med)
                print('loss:', loss)
                print('nll_loss:', nll_loss)
        else:
            loss_med = 0

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
        return loss + loss_med, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def get_aligned_lprobs_and_target(self, lprobs, targets, denom_prev):
        len_targets = (targets != self.padding_idx).sum(-1)
        len_hypos = (denom_prev != self.padding_idx).sum(-1)
        lprobs_argmax = torch.argmax(lprobs.detach(),dim=-1)

        len_max = 0
        targets_after = []
        lprobs_after = []
        for idx_batch in range(len(lprobs)):
            target_list = targets[idx_batch].tolist()[:len_targets[idx_batch]]
            hypos_list = lprobs_argmax[idx_batch].tolist()[:len_hypos[idx_batch]]
            edit_log=Levenshtein.editops(target_list,hypos_list)
            lprob_after = list(range(len(hypos_list)))
            target_after = list(target_list)
            idx_plus_a = 0
            idx_plus_b = 0
            # lprob_after = list(idx_lprob)
            for ele in edit_log:
                err=ele[0]
                position_a = ele[1]
                position_b = ele[2]
                if err == 'insert':
                    if position_a+idx_plus_a < len(target_after):
                        target_after.insert(position_a+idx_plus_a, target_after[position_a+idx_plus_a])
                    else:
                        target_after.insert(position_a+idx_plus_a, target_after[-1])
                    idx_plus_a = idx_plus_a + 1
                elif err == 'delete':
                    if position_b+idx_plus_b < len(lprob_after):
                        lprob_after.insert(position_b+idx_plus_b, lprob_after[position_b+idx_plus_b])
                    else:
                        lprob_after.insert(position_b+idx_plus_b, lprob_after[-1])
                    idx_plus_b = idx_plus_b + 1
            # print('target_after:', target_after)
            # print('lprob_after:', lprob_after)
            # print('edit_log:', edit_log)
            # print('target_list:', target_list)
            # print('lprob_list:', hypos_list)
            # target_after: [4] lprob_after: []

            assert len(target_after) == len(lprob_after)

            if len(target_after) > len_max:
                len_max = len(target_after)
            targets_after.append(target_after)
            lprobs_after.append(lprob_after)
        
        targets_after_tensor = torch.ones(len(lprobs), len_max).to(targets.device, dtype=targets.dtype).fill_(self.padding_idx)
        lprobs_after_tensor = torch.zeros(len(lprobs), len_max, lprobs.size(-1)).to(lprobs.device, dtype=lprobs.dtype)
        denom_prev_after_tensor = torch.ones(len(lprobs), len_max).to(targets.device, dtype=targets.dtype).fill_(self.padding_idx)
        for idx_batch in range(len(lprobs)):
            targets_after_tensor[idx_batch][:len(targets_after[idx_batch])] = torch.tensor(targets_after[idx_batch]).to(targets.device).to(targets.dtype)
            lprobs_after_tensor[idx_batch][:len(lprobs_after[idx_batch])] = lprobs[idx_batch][lprobs_after[idx_batch]]            
            denom_prev_after_tensor[idx_batch][:len(lprobs_after[idx_batch])] = denom_prev[idx_batch][lprobs_after[idx_batch]]

        if torch.randperm(1000)[0] == 0:
            print('target_list:', target_list)
            print('hypos_list:', hypos_list)
            print('edit_log:', edit_log)
            print('prev_tokens_after', denom_prev_after_tensor[-1])
            print('target_after:', target_after)
            print('lprob_after:', torch.argmax(lprobs_after_tensor[-1], dim=-1))
            print('targets_after_tensor:', targets_after_tensor.size())
            print('lprobs_after_tensor:', lprobs_after_tensor.size())
    
        return targets_after_tensor, lprobs_after_tensor

    def compute_med_loss(self, model, net_output, sample, samples_denome, denom_prev, reduce=True):
        b, t, v = net_output[0].size()
        lprobs, target = self.get_lprobs_and_target(model, net_output, samples_denome)
        # print('net_output[0].size():', net_output[0].size(), lprobs.size(), target.size())
        aligned_target, aligned_lprobs = self.get_aligned_lprobs_and_target(lprobs.view(b, t, v), target.view(b, t), denom_prev)
        loss, nll_loss = label_smoothed_nll_loss(
            aligned_lprobs.view(-1, v),
            aligned_target.view(-1),
            0,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

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
