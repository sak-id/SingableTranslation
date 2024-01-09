import math
import os
import re
import warnings
from typing import Optional, Union, Callable, List, Iterable, Dict, Any, Tuple
import json

import torch
import random
import torch.nn as nn

import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.autograd import Variable
from transformers import MBartForConditionalGeneration, MBartModel, MBartConfig, MBart50TokenizerFast, BeamScorer, \
    LogitsProcessorList, StoppingCriteriaList, LogitsProcessor
from transformers import MT5Model, MT5Config, MT5TokenizerFast, MT5ForConditionalGeneration # for MT5
from transformers.file_utils import ModelOutput
from transformers.generation_stopping_criteria import validate_stopping_criteria
from transformers.generation_utils import BeamSearchOutput, GreedySearchOutput, SampleOutput, BeamSampleOutput, \
    BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutput, \
    Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.models.mbart.modeling_mbart import MBartDecoder, _expand_mask, logger, shift_tokens_right
from transformers.pytorch_utils import torch_int_div

rhyme_bias = 1e-8

def main():
    # play_tokenizer()

    mbart = MBartForCGCharEmbLen.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    # mbart, info = MBartLc3ForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt",
    #                                                                output_loading_info=True)
    # print(mbart.config)

    play_mbart_emb(mbart)


def get_model(model_class_name, model_name_or_path, config, cache_dir):
    print(model_class_name)
    cls = eval(model_class_name)
    model = cls.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=cache_dir,
    )
    return model

def get_model_online(model_class_name, model_name_or_path):
    print(model_class_name)
    cls = eval(model_class_name)
    model = cls.from_pretrained(model_name_or_path)
    return model


from pynvml import nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(3)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def shift_tokens_right_prefix_21(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    Without changing the position of prefix
    Prefix length: 21
    """
    n = 21
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 0:n] = input_ids[:, 0:n].clone()
    shifted_input_ids[:, n + 1:] = input_ids[:, n:-1].clone()
    shifted_input_ids[:, n] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def shift_tokens_right_prefix_20(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    Without changing the position of prefix
    Prefix length: 20
    """
    n = 20
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 0:n] = input_ids[:, 0:n].clone()
    shifted_input_ids[:, n + 1:] = input_ids[:, n:-1].clone()
    shifted_input_ids[:, n] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def shift_tokens_right_prefix_n(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    Without changing the position of prefix
    Prefix length: n
    """
    n = 2
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 0:n] = input_ids[:, 0:n].clone()
    shifted_input_ids[:, n + 1:] = input_ids[:, n:-1].clone()
    shifted_input_ids[:, n] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def shift_tokens_right_prefix_2(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    Without changing the position of prefix
    Prefix length: 2
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 0:2] = input_ids[:, 0:2].clone()
    shifted_input_ids[:, 3:] = input_ids[:, 2:-1].clone()
    shifted_input_ids[:, 2] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def shift_tokens_right_prefix_1(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    Without changing the position of prefix
    Prefix length: 1 (for controlling length/rhyme only)
    For this function, input are shifted 2 tokens actually
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape) # 既存のtensor: input_idsと同じshapeの0で初期化されたtensorを作成
    shifted_input_ids[:, 0:1] = input_ids[:, 0:1].clone() # cloneはtensorの微分できる形でのコピー
    shifted_input_ids[:, 2:] = input_ids[:, 1:-1].clone()
    shifted_input_ids[:, 1] = decoder_start_token_id
    #0列目はinput_idsの0列目、1列目はdecoder_start_token_id、2列目以降はinput_idsの1列目以降

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        data = f.read()
        data = json.loads(data)
    return data


def play_tokenizer():
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    print(tokenizer.lang_code_to_id["zh_CN"])  # 250025
    print(tokenizer.lang_code_to_id["en_XX"])  # 250004


def play_mbart_inference(model):
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    device = 'cuda'
    model.to(device)
    print('load complete')
    text = [
        "There's only one song left for you",
        # 'Get me off the streets of this city',
        # 'You only left one kiss for me',
        # "You're laughing so brightly",
        # 'Keeps me in my head for the rest of my life',
        # "How come I'm still stuck and you're long gone",
        # "You're laughing so brightly"
    ]
    # model.to(device)
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = 'zh_CN'
    encoded_input = tokenizer(text, return_tensors="pt", padding=True).to(
        device)  # shape: [bs, max_input_seq_len]  [7, 15]
    desired_length = torch.tensor([6, 5, 4, 5, 6, 7, 3], dtype=torch.int64, device=device).unsqueeze(1)
    generated_tokens = model.generate(**encoded_input,
                                      max_length=48,
                                      forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"])  # zh_CN, en_XX
    # print(tokenizer.convert_ids_to_tokens(2))
    for line in generated_tokens:
        print([tokenizer.decode(i) for i in line])
    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))


def play_mbart_emb(model):
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    device = 'cuda'
    model.to(device)
    print('load complete')
    text = [
        "There's only one song left for you",
        'Get me off the streets of this city',
        # 'You only left one kiss for me',
        # "You're laughing so brightly",
        # 'Keeps me in my head for the rest of my life',
        # "How come I'm still stuck and you're long gone",
        # "You're laughing so brightly"
    ]
    # model.to(device)
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = 'zh_CN'
    encoded_input = tokenizer(text, return_tensors="pt", padding=True)['input_ids'].to(
        device)  # shape: [bs, max_input_seq_len]  [7, 15]
    print(encoded_input)
    print(encoded_input.shape)
    constraints = ['7654321', '54321']

    # Convert constraint to tensor
    constraint_tensor = model.convert_constraint_to_tensor(encoded_input, constraints)
    print(constraint_tensor)
    constraint_tensor = model.shift_constraints(constraint_tensor, num_shift=2)
    print(constraint_tensor)
    # desired_length = torch.tensor([6, 5, 4, 5, 6, 7, 3], dtype=torch.int64, device=device).unsqueeze(1)
    # generated_tokens = model.generate(**encoded_input,
    #                                   max_length=48,
    #                                   forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"])  # zh_CN, en_XX
    # # print(tokenizer.convert_ids_to_tokens(2))
    # for line in generated_tokens:
    #     print([tokenizer.decode(i) for i in line])
    # print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))


def play_mbart_training(model):
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    device = 'cuda'
    model.to(device)
    print('load complete')
    text = [
        "There's only one song left for you",
        # 'Get me off the streets of this city',
        # 'You only left one kiss for me',
        # "You're laughing so brightly",
        # 'Keeps me in my head for the rest of my life',
        # "How come I'm still stuck and you're long gone",
        # "You're laughing so brightly"
    ]
    # model.to(device)
    tokenizer.src_lang = "en_XX"
    # tokenizer.src_lang = 'zh_CN'
    tokenizer.tgt_lang = 'zh_CN'
    encoded_input = tokenizer(text, return_tensors="pt", padding=True).to(
        device)  # shape: [bs, max_input_seq_len]  [7, 15]
    desired_length = torch.tensor([6, 5, 4, 5, 6, 7, 3], dtype=torch.int64, device=device).unsqueeze(1)
    generated_tokens = model.generate(**encoded_input,
                                      max_length=48,
                                      forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"],
                                      desired_length=desired_length)  # zh_CN, en_XX
    # print(tokenizer.convert_ids_to_tokens(2))
    for line in generated_tokens:
        print([tokenizer.decode(i) for i in line])
    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))


# Modified for not using bad_word_list by ieda
class MBartForConditionalGenerationCharLevel(MBartForConditionalGeneration):
    def __init__(self, config: MBartConfig):
        super().__init__(config)

        # Read bad word list
        import os
        self.bad_word_list = read_json(os.path.join(os.path.dirname(__file__), '../tokenizers/misc/bad_word_list.json'))

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        # Mask out bad (multi-char) words
        lm_logits[:, :, self.bad_word_list] = torch.tensor(float('-inf'), device=lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


    def beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None, #listは受け付けないよう変更されている
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            # Finish of beam search step.
            # print(beam_next_tokens)

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        # print()

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]


class MBartForConditionalGenerationCharLevelBiasedFirst(MBartForConditionalGenerationCharLevel):

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "tgt_"]  # add tgt_ for biased decoding
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    def beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        # ------ Rhyme dic for biased decoding code here ------
        rhyme_dic = read_json(os.path.dirname(__file__)+'/../tokenizers/misc/rhyme_type_dic.json')
        # ------                           ------

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # Debug

            # 检查这个句子已经生成的单词
            # exit(100)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            # ------ Biased decoding code here ------
            tgt_rhyme = [str(i) for i in model_kwargs['tgt_rhymes'].tolist()]
            tgt_ch_ids = []
            for i in tgt_rhyme:
                for j in range(5):
                    tgt_ch_ids.append(rhyme_dic[i])
            if cur_len == 3: # when generating the first token,
                for i in range(next_token_scores.shape[0]):
                    next_token_scores[i][tgt_ch_ids[i]] *= rhyme_bias
            # ------                           ------

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            # Finish of beam search step.
            # print(beam_next_tokens)

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        # print()

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

class MBartForConditionalGenerationCharLevelBiasedLast(MBartForConditionalGenerationCharLevel):
    '''
    Model class that implement beam search for rhyme biased decoder (left to right order)
    '''

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "tgt_"]  # add tgt_ for biased decoding
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    def beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        # ------ Rhyme dic for biased decoding code here ------
        rhyme_dic = read_json(os.path.dirname(__file__)+'/../tokenizers/misc/rhyme_type_dic.json')
        # ------                           ------

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # Debug

            # 检查这个句子已经生成的单词
            # exit(100)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            # ------ Biased decoding code here ------
            tgt_rhyme = [str(i) for i in model_kwargs['tgt_rhymes'].tolist()]
            # Because target length of every sentence is different, we have to make the bs=1
            assert len(tgt_rhyme) == 1
            tgt_rhyme = tgt_rhyme[0]
            tgt_ch_ids = []
            # for i in tgt_rhyme:
            for j in range(5):
                tgt_ch_ids.append(rhyme_dic[tgt_rhyme])
            tgt_len = model_kwargs['tgt_length'].item()
            # print(type(tgt_len), tgt_len)
            # exit(10)
            if cur_len == 3 + tgt_len - 1: # when generating the last token,
                for i in range(next_token_scores.shape[0]):
                    next_token_scores[i][tgt_ch_ids[i]] *= 1e-8
            # ------                           ------

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            # Finish of beam search step.
            # print(beam_next_tokens)

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        # print()

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

class MBartForConditionalGenerationCharLevelBiasedBdr(MBartForConditionalGenerationCharLevel):
    '''
    Model class that implement beam search for rhyme biased decoder (left to right order)
    '''

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "tgt_", "bdr_pos"]  # add tgt_ for biased decoding
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    def beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        # ------ Rhyme dic for biased decoding code here ------
        emit_ts = torch.load(os.path.dirname(__file__)+'/../../utils_common/emit_ts.pt').to('cuda')
        bdr_pos = model_kwargs['bdr_pos']
        # Because target length of every sentence is different, we have to make the bs=1
        assert len(bdr_pos) == 1
        bdr_pos = bdr_pos[0]
        # ------                           ------

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # Debug

            # 检查这个句子已经生成的单词
            # exit(100)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            # ------ Biased decoding code here ------
            # This model does not have rhyme control
            # When cur_len == 2 (<bos>, _, ), it is generating the first token
            # when generating i'th ( i from 0) token, cur_len = 2 + i
            # when generating j'th (j from 1) token, cur_len = 1 + j
            # when generating (j+1)th token, cur_len = 2 + j
            print('cut_len = ', cur_len)
            if cur_len - 2 not in bdr_pos: # when generating token after the position of bdr_1
                for i in range(next_token_scores.shape[0]): # for each beam
                    next_token_scores[i] += emit_ts
            else:
                pass
            # ------                           ------

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            # Finish of beam search step.
            # print(beam_next_tokens)

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        # print()

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]


class MBartEmbLenModel(MBartModel):
    def __init__(self, config):
        super().__init__(config)
        self.constraint_emb = nn.Embedding(30, config.d_model)  # , padding_idx=0)   but Stress emb need padding token


class MBartEmbLenForCGChar(MBartForConditionalGeneration):
    '''
    mBart for embedding length control
    '''

    def __init__(self, config: MBartConfig):
        # Code from MBartForConditionalGeneration, but change the model class
        super(MBartForConditionalGeneration, self).__init__(config)
        self.model = MBartEmbLenModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.post_init()

        # Read bad word list
        import os
        self.bad_word_list = read_json(os.path.join(os.path.dirname(__file__), '../tokenizers/misc/bad_word_list.json'))

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        lm_logits[:, :, self.bad_word_list] = torch.tensor(float('-inf'),
                                                           device=lm_logits.device)  # Mask out bad (multi-char) words

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    # cnt = 0
    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        '''
        Function called inside each loop of beam search. Might need modification.
        '''

        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # print(past)
        # print(decoder_input_ids.shape)

        # # debug
        # self.cnt += 1
        # print(decoder_input_ids, kwargs['emb_ids'][:, :1])

        # if decoder_input_ids.shape[1] == 2:
        #     model_kwargs['emb_ids'][:, 1:]

        # Compute embeddings
        inputs_embeds = self.model.decoder.embed_tokens(
            decoder_input_ids) * self.model.decoder.embed_scale  # [bs * beam_size, 1, d_model]
        # print(kwargs['emb_ids'])
        constraint_embeds = self.model.constraint_emb(kwargs['emb_ids'][:, :1])  # # [bs, ref_len -> 1, d_model]
        # print(kwargs['emb_ids'])
        # print(kwargs['emb_ids'][:, :1])
        # kwargs['emb_ids'] = kwargs['emb_ids'][:, 1:]
        # print(kwargs['emb_ids'])
        # print('but')
        # print(kwargs['emb_ids'])
        # exit(10)

        # if self.cnt == 3:
        #     exit(10)

        constraint_embeds = self.repeat_on_first_dim(
            constraint_embeds,
            expand_size=int(inputs_embeds.shape[0] / constraint_embeds.shape[0])
        )
        decoder_input_embedding = inputs_embeds + constraint_embeds

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            # "decoder_input_ids": decoder_input_ids,
            "decoder_inputs_embeds": decoder_input_embedding,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        '''
        This function has been modified so that the emb_ids in kwargs is excluded from encoder's kwargs
        '''
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", 'emb_ids']  # exclude emb_ids from encoder kwargs
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    def _update_model_kwargs_for_generation(
            self, outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        '''
        This function has been modified so that emb_ids will be updated
        '''
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        # Update embedding ids
        # print(model_kwargs['emb_ids'].shape)
        if model_kwargs['emb_ids'].shape[1] > 1:
            model_kwargs['emb_ids'] = model_kwargs['emb_ids'][:, 1:]
        else:
            model_kwargs['emb_ids'] = torch.zeros_like(model_kwargs['emb_ids'])

        return model_kwargs

    def repeat_on_first_dim(self, tensor, expand_size):
        '''
        Repeat a tensor on the first dim. Prepare for beam search.
        '''
        expanded_return_idx = (
            torch.arange(tensor.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(tensor.device)
        )
        tensor = tensor.index_select(0, expanded_return_idx)
        return tensor

    def prepare_decoder_inputs_embeds_for_training(self, decoder_input_ids, constraints):
        '''
        Compute input embedding given input ids and constraint.
        decoder_input_ids: float tensor, [bs, reference_length]
        constraint: a list of strings
        '''
        # print('train prepare func called')
        # print_gpu_utilization()
        assert decoder_input_ids.shape[0] == constraints.shape[0] or \
               decoder_input_ids.shape[0] % constraints.shape[0] == 0

        # Compute embeddings
        inputs_embeds = self.model.decoder.embed_tokens(
            decoder_input_ids) * self.model.decoder.embed_scale  # [bs, ref_len, d_model]
        constraints = F.pad(constraints, pad=[1, 0, 0, 0])
        constraint_embeds = self.model.constraint_emb(constraints)  # # [bs, ref_len, d_model]

        # print_gpu_utilization()
        if constraint_embeds.shape[1] < inputs_embeds.shape[1]:
            dif = inputs_embeds.shape[1] - constraint_embeds.shape[1]
            constraint_embeds = F.pad(constraint_embeds, pad=(0, 0, 0, dif))
        elif constraint_embeds.shape[1] > inputs_embeds.shape[1]:
            dif = constraint_embeds.shape[1] - inputs_embeds.shape[1]
            constraint_embeds = constraint_embeds[:, :-dif, :]
        ret = inputs_embeds + constraint_embeds

        return ret


class MBartEmbRhyModel(MBartModel):
    def __init__(self, config):
        super().__init__(config)
        self.constraint_emb = nn.Embedding(20, config.d_model, padding_idx=0)  # but Stress emb need padding token


class MBartEmbRhyForCGChar(MBartForConditionalGeneration):
    '''
    mBart for embedding length control
    '''

    def __init__(self, config: MBartConfig):
        # Code from MBartForConditionalGeneration, but change the model class
        super(MBartForConditionalGeneration, self).__init__(config)
        self.model = MBartEmbRhyModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.post_init()

        # Read bad word list
        import os
        self.bad_word_list = read_json(os.path.join(os.path.dirname(__file__), '../tokenizers/misc/bad_word_list.json'))

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        lm_logits[:, :, self.bad_word_list] = torch.tensor(float('-inf'),
                                                           device=lm_logits.device)  # Mask out bad (multi-char) words

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        '''
        Function called inside each loop of beam search. Might need modification.
        '''

        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # print(past)
        # print(decoder_input_ids.shape)

        # Compute embeddings
        inputs_embeds = self.model.decoder.embed_tokens(
            decoder_input_ids) * self.model.decoder.embed_scale  # [bs * beam_size, 1, d_model]
        # print(kwargs['emb_ids'].shape)
        constraint_embeds = self.model.constraint_emb(kwargs['emb_ids'])  # # [bs, ref_len -> 1, d_model]

        constraint_embeds = self.repeat_on_first_dim(
            constraint_embeds,
            expand_size=int(inputs_embeds.shape[0] / constraint_embeds.shape[0])
        )
        decoder_input_embedding = inputs_embeds + constraint_embeds

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            # "decoder_input_ids": decoder_input_ids,
            "decoder_inputs_embeds": decoder_input_embedding,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        '''
        This function has been modified so that the emb_ids in kwargs is excluded from encoder's kwargs
        '''
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", 'emb_ids']  # exclude emb_ids from encoder kwargs
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    def repeat_on_first_dim(self, tensor, expand_size):
        '''
        Repeat a tensor on the first dim. Prepare for beam search.
        '''
        expanded_return_idx = (
            torch.arange(tensor.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(tensor.device)
        )
        tensor = tensor.index_select(0, expanded_return_idx)
        return tensor

    def prepare_decoder_inputs_embeds_for_training(self, decoder_input_ids, constraints):
        '''
        Compute input embedding given input ids and constraint.
        decoder_input_ids: float tensor, [bs, reference_length]
        constraint: a list of strings
        '''
        # print('train prepare func called')
        # print_gpu_utilization()
        assert decoder_input_ids.shape[0] == constraints.shape[0] or \
               decoder_input_ids.shape[0] % constraints.shape[0] == 0

        # Compute embeddings
        inputs_embeds = self.model.decoder.embed_tokens(
            decoder_input_ids) * self.model.decoder.embed_scale  # [bs, ref_len, d_model]
        constraint_embeds = self.model.constraint_emb(constraints)  # # [bs, 1, d_model]

        # print_gpu_utilization()
        # constraint_
        # if constraint_embeds.shape[1] < inputs_embeds.shape[1]:
        #     dif = inputs_embeds.shape[1] - constraint_embeds.shape[1]
        #     constraint_embeds = F.pad(constraint_embeds, pad=(0, 0, 0, dif))
        # elif constraint_embeds.shape[1] > inputs_embeds.shape[1]:
        #     dif = constraint_embeds.shape[1] - inputs_embeds.shape[1]
        #     constraint_embeds = constraint_embeds[:, :-dif, :]
        ret = inputs_embeds + constraint_embeds

        return ret


class PositionalEncodingWithDesiredLength(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=108):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # [d_model/2, ]
        # div_term = [ 1 / （10000 ** 2i/d_model)   for i in range(0, d_model, 2) ]
        # div_term = 1 / (10000 ** (torch.arange(0., d_model, 2) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0, desired_length=None):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.pe.device
        )  # an array containing one or multiple elements, representing the index of tokens
        # print(generated_ratio.shape)
        ret = torch.tensor(self.pe[desired_length], device=self.pe.device, requires_grad=False)
        # shape: [len(position), d_model]
        return ret


class ForcedBOSTokenLogitsProcessorPrefixDecoder(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces the specified token as the first generated token.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id: int):
        self.bos_token_id = bos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len == 3: #ここだけ違う　元々は1
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != self.bos_token_id]] = -float("inf")
            scores[:, self.bos_token_id] = 0
        return scores


class ForcedBOSTokenLogitsProcessorPrefixDecoderLength(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces the specified token as the first generated token.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id: int):
        self.bos_token_id = bos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len == 2:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != self.bos_token_id]] = -float("inf")
            scores[:, self.bos_token_id] = 0
        return scores


class ForcedBOSTokenLogitsProcessorPrefixDecoderN(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces the specified token as the first generated token.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id: int, prefix_length):
        self.bos_token_id = bos_token_id
        self.prefix_length = prefix_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len == self.prefix_length + 1:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != self.bos_token_id]] = -float("inf")
            scores[:, self.bos_token_id] = 0
        return scores


if __name__ == '__main__':
    main()
