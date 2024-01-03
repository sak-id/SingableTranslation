import itertools
import json
import linecache
import math
import os
import random
import sys
import pickle
import socket
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

import git
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from torch import nn
from torch.utils.data import Dataset, Sampler

from .sentence_splitter import add_newline_to_end_of_each_sentence
from transformers import BartTokenizer, EvalPrediction, PreTrainedTokenizer, T5Tokenizer
from transformers.file_utils import cached_property
from transformers.models.bart.modeling_bart import shift_tokens_right
from datasets import load_metric
# import evaluate

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utils_common.utils import TextCorrupterEn, TextCorrupterCh

try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False

def trim_batch(
        input_ids,
        pad_token_id,
        attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

class AbstractSeq2SeqDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            **dataset_kwargs
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")
        if os.path.exists(self.len_file): #もし既存のlenファイルがあれば
            self.src_lens = pickle_load(self.len_file) #pickle化したファイルを読み込む
            self.used_char_len = False
        else: #なければ作る
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert not self.used_char_len, "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")


# class LegacySeq2SeqDataset(AbstractSeq2SeqDataset):
class Seq2SeqDataset(AbstractSeq2SeqDataset): # MBartForConditionalGeneration 動作確認
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Call tokenizer on src and tgt_lines"""
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        source_inputs = self.encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = self.encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": target_ids,
        }

    def encode_line(self, tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
        """Only used by LegacyDataset"""
        return tokenizer(
            [line],
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            **self.dataset_kwargs,
        )

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": y,
        }
        return batch

class Seq2SeqDatasetJaPrefixEncoderLength(AbstractSeq2SeqDataset):
    """
    Read constraints file when preparing data, append it to the beginning of input text
    Dataset class for encoder prompt
    TODO: add code Read rhyme constraints to batch, but doesn't add to input as prefix
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            type_path="train",
            n_obs=None,
            prefix="",
            constraint_type='reference',
            **dataset_kwargs
    ):
        super().__init__(tokenizer,
                         data_dir,
                         max_source_length,
                         max_target_length,
                         type_path,
                         n_obs,
                         prefix,
                         **dataset_kwargs)
        t = Path(data_dir).joinpath('constraints').joinpath(constraint_type).joinpath(type_path + ".target")
        print(t)
        assert t.exists()
        self.tgt_cons_file = t

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        constraint_line = linecache.getline(str(self.tgt_cons_file), index).rstrip('\n')
        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"
        assert constraint_line, f'empty constraint line for index {index}'
        length, rhyme = [int(i) for i in constraint_line.split('\t')]
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1, 'tgt_len': length, 'tgt_rhyme': rhyme}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""

        # Code in Mbart50TokenizerFast
        kwargs = self.dataset_kwargs.copy()
        # print('kwargs:', kwargs)
        src_lang = kwargs.pop('src_lang')
        tgt_lang = kwargs.pop('tgt_lang')
        src_texts = [x["src_texts"] for x in batch]
        tgt_texts = [x["tgt_texts"] for x in batch]
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        # Code in PreTrainedTokenizerFast
        max_length = self.max_source_length
        max_target_length = self.max_target_length
        padding = kwargs.pop('padding') if 'padding' in kwargs else 'longest'
        return_tensors = "pt"
        truncation = kwargs.pop('truncation') if 'truncation' in kwargs else True

        # Process src_texts
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        model_inputs = self.tokenizer(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        # print(model_inputs.keys()) # 'input_ids', 'attention_mask'
        # print(model_inputs)
        assert tgt_texts != None

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )  # Tensor: [BS, max_seq_len_in_batch] device: cpu
        labels = labels['input_ids']
        model_inputs["labels"] = labels

        # Process format and rhyme constraints
        tgt_lens = ['len_{}'.format(x["tgt_len"]) for x in batch]
        # tgt_rhymes = ['rhy_{}'.format(x["tgt_rhyme"]) for x in batch]
        t1 = self.tokenizer(
            tgt_lens,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=1,
            padding=False,
            truncation=True,
        )
        # t2 = self.tokenizer(
        #     tgt_rhymes,
        #     add_special_tokens=False,
        #     return_tensors=return_tensors,
        #     max_length=1,
        #     padding=False,
        #     truncation=True,
        # )
        tgt_lens = t1['input_ids']
        # tgt_rhymes = t2['input_ids']
        attn_len = t1['attention_mask']
        # attn_rhy = t2['attention_mask']
        model_inputs['tgt_lens'] = torch.tensor([x['tgt_len'] for x in batch], dtype=torch.long)
        model_inputs['tgt_rhymes'] = torch.tensor([x['tgt_rhyme'] for x in batch], dtype=torch.long)

        # Concat length and rhyme constraints with target ids
        input_ids = torch.cat((tgt_lens, model_inputs['input_ids']), dim=1)
        attention_mask = torch.cat((attn_len, model_inputs['attention_mask']), dim=1)
        model_inputs["input_ids"] = input_ids
        model_inputs['attention_mask'] = attention_mask

        # Save data to batch_encoding
        batch_encoding = model_inputs.data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding
