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
import editdistance

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
    nll_loss = -lprobs.gather(dim=-1, index=target.to(lprobs.device))
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index).to(lprobs.device)
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


def matching_word_error_distribution_loss(lprobs_positive, lprobs_negative, target_positive, target_negative, ignore_index=None, levenshtein_distance=None):
    # get mask
    pad_mask_positive = target_positive.eq(ignore_index)
    pad_mask_negative = target_negative.eq(ignore_index)

    # masking
    lprobs_positive = lprobs_positive * (1-pad_mask_positive.to(lprobs_positive.dtype).to(lprobs_positive.device)).unsqueeze(-1)
    lprobs_negative = lprobs_negative * (1-pad_mask_negative.to(lprobs_negative.dtype).to(lprobs_positive.device)).unsqueeze(-1)

    # Dimensions
    # target_positive: torch.Size([2592])
    # lprobs_positive: torch.Size([2592, 300])
    # target_negative: torch.Size([32, 82])
    # lprobs_negative: torch.Size([32, 82, 300])

    # get NLL
    nll_loss_positive = -lprobs_positive.gather(dim=-1, index=target_positive.to(lprobs_positive.device).unsqueeze(-1)).squeeze(-1)
    nll_loss_negative = -lprobs_negative.gather(dim=-1, index=target_negative.to(lprobs_positive.device).unsqueeze(-1)).squeeze(-1)

    # contrastive loss v1
    # loss_contrastive = nll_loss_positive.sum() - nll_loss_negative.sum()

    # contrastive loss v2 (margin loss)
    # loss_mask = nll_loss_negative.sum(-1) < nll_loss_positive.view(len(nll_loss_negative), -1).sum(-1)
    # loss_contrastive = nll_loss_negative.sum(-1) * loss_mask.to(nll_loss_negative.dtype).to(nll_loss_negative.device)
    # loss_contrastive = loss_contrastive.sum()

    # contrastive loss v3 (MWED)
    # temperature = 1.0
    # nll_loss_positive = nll_loss_positive.view(len(nll_loss_negative), -1).sum(-1) # B
    # nll_loss_negative = nll_loss_negative.sum(-1) # B
    # inferred_lev_distribution = torch.log_softmax(torch.cat([nll_loss_positive.unsqueeze(0), nll_loss_negative.unsqueeze(0)], dim=0), dim=0) # 2, B

    # true_lev_distribution = torch.zeros(len(levenshtein_distance)).to(nll_loss_negative.device).to(nll_loss_negative.dtype)
    # true_lev_distribution = torch.cat([true_lev_distribution.unsqueeze(0), -torch.tensor(levenshtein_distance).to(nll_loss_negative.device).to(nll_loss_negative.dtype).unsqueeze(0)], dim=0) # 2, B
    # true_lev_distribution = torch.softmax(true_lev_distribution * temperature, dim=0) # 2, B

    # loss_contrastive = - (true_lev_distribution * inferred_lev_distribution)[0].sum()

    # contrastive loss v3 (MWED v2)
    temperature = 1.0
    nll_loss_positive = nll_loss_positive.view(len(nll_loss_negative), -1).sum(-1) # B
    nll_loss_negative = nll_loss_negative.sum(-1) # B
    inferred_lev_distribution = torch.log_softmax(torch.cat([nll_loss_positive.unsqueeze(0), nll_loss_negative.unsqueeze(0)], dim=0), dim=0) # 2, B

    true_lev_distribution = torch.zeros(len(levenshtein_distance)).to(nll_loss_negative.device).to(nll_loss_negative.dtype)
    true_lev_distribution = torch.cat([true_lev_distribution.unsqueeze(0), -torch.tensor(levenshtein_distance).to(nll_loss_negative.device).to(nll_loss_negative.dtype).unsqueeze(0)], dim=0) # 2, B
    true_lev_distribution = torch.softmax(true_lev_distribution * temperature, dim=0) # 2, B

    loss_contrastive = - (true_lev_distribution * inferred_lev_distribution).sum()

    # contrastive loss v3 (MWED v3)
    if torch.randperm(1000)[0]==0:
        print('true_lev_distribution.size:', true_lev_distribution.size())
        print('inferred_lev_distribution.size:', inferred_lev_distribution.size())
        print('levenshtein_distance:', levenshtein_distance)
        print('true_lev_distribution:', true_lev_distribution)
        print('inferred_lev_distribution:', torch.exp(inferred_lev_distribution))
        print('loss_contrastive:', loss_contrastive)

    return loss_contrastive


def matching_word_error_distribution_loss_v2(lprobs_positive, lprobs_negative, target_positive, target_negative, ignore_index=None, levenshtein_distance=None):
    len_hyps = len(target_negative)
   
    # get mask
    pad_mask_positive = target_positive.eq(ignore_index)

    # masking
    lprobs_positive = lprobs_positive * (1-pad_mask_positive.to(lprobs_positive.dtype).to(lprobs_positive.device)).unsqueeze(-1)

    # get NLL
    nll_loss_positive = -lprobs_positive.gather(dim=-1, index=target_positive.to(lprobs_positive.device).unsqueeze(-1)).squeeze(-1)
    nll_loss_positive = nll_loss_positive.view(len(target_negative[0]), -1).sum(-1) # B

    # for denominator terms
    nll_losses = [nll_loss_positive.unsqueeze(0)]
    for j in range(len(lprobs_negative)):
        pad_mask_negative = target_negative[j].eq(ignore_index)
        lprobs_negative[j] = lprobs_negative[j] * (1-pad_mask_negative.to(lprobs_positive.dtype).to(lprobs_positive.device)).unsqueeze(-1)
        nll_loss_negative = -lprobs_negative[j].gather(dim=-1, index=target_negative[j].to(lprobs_positive.device).unsqueeze(-1)).squeeze(-1)
        nll_loss_negative = nll_loss_negative.sum(-1) # B
        nll_losses.append(nll_loss_negative.unsqueeze(0))

    inferred_lev_distribution = torch.log_softmax(torch.cat(nll_losses, dim=0), dim=0) # H+1, B

    # levenshtein_distance: B, H
    # normalize levenshtein_distance
    # levenshtein_distance_avg = torch.tensor(levenshtein_distance.to(nll_loss_negative.dtype)).mean(dim=1).unsqueeze(-1) # B, 1
    levenshtein_distance_avg = levenshtein_distance.to(lprobs_positive.dtype).mean(dim=1).unsqueeze(-1) # B, 1
    temperature = 5.0
    true_lev_distribution = torch.zeros(len(nll_loss_positive)).to(lprobs_positive.device).to(lprobs_positive.dtype) # B
    true_lev_distribution = torch.cat([true_lev_distribution.unsqueeze(-1), -levenshtein_distance.to(lprobs_positive.device).to(lprobs_positive.dtype)], dim=-1) # B, H+1
    true_lev_distribution = torch.softmax((true_lev_distribution - levenshtein_distance_avg) * temperature, dim=-1) # B, H+1

    loss_contrastive = - (true_lev_distribution * inferred_lev_distribution.transpose(0,1)).sum()

    if torch.randperm(2000)[0]==0:
        print('true_lev_distribution.size:', true_lev_distribution.size())
        print('inferred_lev_distribution.size:', inferred_lev_distribution.size())
        print('levenshtein_distance:', levenshtein_distance)
        print('true_lev_distribution:', true_lev_distribution)
        print('inferred_lev_distribution:', torch.exp(inferred_lev_distribution.transpose(0,1)))
        print('loss_contrastive:', loss_contrastive)
        print('temperature:', temperature)

    return loss_contrastive

def get_one_best_denom(prev_output_tokens, sample, padding_idx):
    # get BOS and EOS
    bos = int(prev_output_tokens[0][0].item())
    bos_str = str(bos) + ' '
    eos_str = ' ' + str(padding_idx)

    # get denominator term
    prev_output_tokens_denom_list = []
    max_sequence_len = len(prev_output_tokens[0])
    levenshtein_distance = []
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
                # Return 1-best hypothesis that is different from the true previous output tokens
                prev_output_tokens_denom_list.append(sample['hypos'][i][j]['tokens'])
                levenshtein_distance.append(editdistance.eval(prev_output_token_str, prev_output_token_denom_str))
                there_is_no_denom = False
                if max_sequence_len < len(sample['hypos'][i][j]['tokens']):
                    max_sequence_len = len(sample['hypos'][i][j]['tokens'])
                break
        if there_is_no_denom:
            # If there is no hypothesis that is different from the true previous output tokens return the true previous output tokens
            prev_output_tokens_denom_list.append(prev_output_tokens[i][1:])
            levenshtein_distance.append(0)

    # previous output tokens (denominator term)
    prev_output_tokens_denom_target = torch.ones(len(prev_output_tokens), max_sequence_len + 1).to(prev_output_tokens.device, dtype=prev_output_tokens.dtype).fill_(padding_idx)
    prev_output_tokens_denom = torch.ones(len(prev_output_tokens), max_sequence_len + 1).to(prev_output_tokens.device, dtype=prev_output_tokens.dtype).fill_(padding_idx)
    prev_output_tokens_denom[:,0] = prev_output_tokens[:,0] # BOS

    assert len(prev_output_tokens_denom_list) == len(prev_output_tokens) == len(levenshtein_distance)

    for i in range(len(prev_output_tokens_denom_list)):
        length = len(prev_output_tokens_denom_list[i])
        prev_output_tokens_denom[i][1:length+1] = prev_output_tokens_denom_list[i]
        prev_output_tokens_denom_target[i][:length] = prev_output_tokens_denom_list[i]

    return prev_output_tokens_denom, prev_output_tokens_denom_target, levenshtein_distance

def get_n_best_denom(prev_output_tokens, sample, padding_idx, num_hyps):
    # get BOS and EOS
    bos = int(prev_output_tokens[0][0].item())
    bos_str = str(bos) + ' '
    eos_str = ' ' + str(padding_idx)

    # get denominator terms
    prev_output_tokens_denom_list = []
    max_sequence_len = len(prev_output_tokens[0])
    for i in range(prev_output_tokens.size(0)):
        prev_output_tokens_denom_list_tmp = []
        # i: batch index
        prev_output_token_str = prev_output_tokens[i].tolist()
        prev_output_token_str = map(str, prev_output_token_str)
        prev_output_token_str = ' '.join(prev_output_token_str).replace(bos_str, '').replace(eos_str,'') # remove BOS and EOS
        for j in range(len(sample['hypos'][i])):
            prev_output_tokens_denom_dict = dict()
            # j: N-best hypothesis index
            prev_output_token_denom_str = sample['hypos'][i][j]['tokens'].tolist()
            prev_output_token_denom_str = map(str, prev_output_token_denom_str)
            prev_output_token_denom_str = ' '.join(prev_output_token_denom_str).replace(bos_str, '').replace(eos_str,'') # remove BOS and EOS
            prev_output_tokens_denom_dict['tokens'] = sample['hypos'][i][j]['tokens']
            prev_output_tokens_denom_dict['levenshtein_distance'] = editdistance.eval(prev_output_token_str, prev_output_token_denom_str) / max(len(prev_output_token_str), len(prev_output_token_denom_str)) * (num_hyps + 1)
            if max_sequence_len < len(sample['hypos'][i][j]['tokens']):
                max_sequence_len = len(sample['hypos'][i][j]['tokens'])
            prev_output_tokens_denom_list_tmp.append(prev_output_tokens_denom_dict)
        prev_output_tokens_denom_list.append(prev_output_tokens_denom_list_tmp)

    prev_output_tokens_denom_target_list_final = []
    prev_output_tokens_denom_list_final = []
    prev_output_tokens_denom_leven_final = torch.zeros((len(prev_output_tokens), num_hyps)).to(sample["net_input"]["src_tokens"].device).to(sample["net_input"]["src_tokens"].dtype)
    for j in range(num_hyps):
        prev_output_tokens_denom_target = torch.ones(len(prev_output_tokens), max_sequence_len + 1).to(prev_output_tokens.device, dtype=prev_output_tokens.dtype).fill_(padding_idx)
        prev_output_tokens_denom = torch.ones(len(prev_output_tokens), max_sequence_len + 1).to(prev_output_tokens.device, dtype=prev_output_tokens.dtype).fill_(padding_idx)
        prev_output_tokens_denom[:,0] = prev_output_tokens[:,0] # BOS
        # prev_output_tokens_denom_leven = torch.zeros(len(prev_output_tokens)).to(prev_output_tokens.device, dtype=torch.long)
        for i in range(prev_output_tokens.size(0)):
            length = len(prev_output_tokens_denom_list[i][j]['tokens'])
            prev_output_tokens_denom[i][1:length+1] = prev_output_tokens_denom_list[i][j]['tokens']
            prev_output_tokens_denom_target[i][:length] = prev_output_tokens_denom_list[i][j]['tokens']
            prev_output_tokens_denom_leven_final[i][j] = prev_output_tokens_denom_list[i][j]['levenshtein_distance']
        prev_output_tokens_denom_target_list_final.append(prev_output_tokens_denom_target)
        prev_output_tokens_denom_list_final.append(prev_output_tokens_denom)

    return prev_output_tokens_denom_list_final, prev_output_tokens_denom_target_list_final, prev_output_tokens_denom_leven_final


@register_criterion(
    "matching_word_error_distribution", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class MatchingWordErrorDistribution(FairseqCriterion):
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

        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        if 'hypos' in sample:
            len_hyps = len(sample['hypos'][0])
            denom_prev, denom_target, levenshtein_distance = get_n_best_denom(prev_output_tokens, sample, self.padding_idx, len_hyps)
            for j in range(len_hyps):
                denom_prev[j] = denom_prev[j].to(prev_output_tokens.device)
                denom_target[j] = denom_target[j].to(prev_output_tokens.device)

        encoder_output = model.encoder(src_tokens, src_lengths)
        net_output = model.decoder(prev_output_tokens, encoder_output)
        if 'hypos' in sample:
            net_output_denoms = []
            net_output_targets = []
            for j in range(len_hyps):
                net_output_denoms.append(model.decoder(denom_prev[j], encoder_output))
                net_output_targets.append({'target': denom_target[j]})
            loss_mwed = self.compute_mwed_loss_v2(model, net_output, net_output_denoms, sample, net_output_targets, levenshtein_distance)
        else:
            loss_mwed = 0

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
        return loss + loss_mwed, sample_size, logging_output

    def get_unnormalized_lprobs_and_target(self, model, net_output, sample, unfold=True):
        lprobs = net_output[0] # model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        # raise ValueError('net_output:', net_output[0].size(), 'lprobs:', lprobs.size()) # ('net_output:', torch.Size([8, 103, 300]), 'lprobs:', torch.Size([8, 103, 300]))
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        if unfold:
            return lprobs.view(-1, lprobs.size(-1)), target.view(-1)
        else:
            return lprobs, target

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
    
    def compute_mwed_loss(self, model, net_output_true, net_output_denom, sample_true, sample_denom, levenshtein_distance=None):
        lprobs_negative, target_negative = self.get_lprobs_and_target(model, net_output_denom, sample_denom, unfold=False)
        lprobs_positive, target_positive = self.get_lprobs_and_target(model, net_output_true, sample_true)
        loss = matching_word_error_distribution_loss(
            lprobs_positive,
            lprobs_negative,
            target_positive,
            target_negative,
            ignore_index=self.padding_idx,
            levenshtein_distance=levenshtein_distance,
        )
        return loss


    def compute_mwed_loss_v2(self, model, net_output_true, net_output_denom, sample_true, sample_denom, levenshtein_distance=None):
        # size
        # net_output_true: B, T, C
        # net_output_denom: H, B, T, C
        # sample_true: B
        # sample_denom: H, B
        # levenshtein_distance: B, H

        lprobs_positive, target_positive = self.get_unnormalized_lprobs_and_target(model, net_output_true, sample_true)

        lprobs_negatives = []
        targets_negatives = []
        for j in range(len(net_output_denom)):
            lprobs_negative, target_negative = self.get_unnormalized_lprobs_and_target(model, net_output_denom[j], sample_denom[j], unfold=False)
            lprobs_negatives.append(lprobs_negative)
            targets_negatives.append(target_negative)
        loss = matching_word_error_distribution_loss_v2(
            lprobs_positive,
            lprobs_negatives,
            target_positive,
            targets_negatives,
            ignore_index=self.padding_idx,
            levenshtein_distance=levenshtein_distance,
        )
        return loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx).to(lprobs.device)
        target = target.to(lprobs.device)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask).to(lprobs.device))
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
