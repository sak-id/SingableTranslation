import os
import re
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../'))

import nltk
import json
import cmudict
# import datasets
import numpy as np
import spacy
import torch
import jieba
import random

import jieba.posseg as pseg
from tqdm import tqdm
from pypinyin import lazy_pinyin, Style, pinyin
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

ls = os.listdir
jpath = os.path.join

# from LM.ngram.language_model import LanguageModel


def main():
    # t = FluencyCaculator.compute_perplexity(['你在南方的艳阳里'])
    # print(t)
    pass


def dic_update(dic, v):
    if v not in dic:
        dic[v] = 1
    else:
        dic[v] += 1


def sort_dic_by_value(dic, reverse=False):
    t = list(dic.items())
    t = [(b, a) for a, b in t]
    t.sort(reverse=reverse)
    t = [(b, a) for a, b in t]
    t = dict(t)
    return t


def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        data = f.read()
        data = json.loads(data)
    return data


def save_json(data, path, sort=False):
    with open(path, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, indent=4, sort_keys=sort, ensure_ascii=False))


def print_json(data):
    print(json.dumps(data, indent=4, ensure_ascii=False))

# For Japanese
class BoundaryUtil:
    def __init__(self):
        self.nlp = spacy.load('ja_ginza')
        self.small_word = ["ャ","ュ","ョ","ァ","ィ","ゥ","ェ","ォ"]

    def get_all_boundary(self, s):
        '''
        Get all word boundaries from one sentence
        '''
        doc = self.nlp(s)

        # Construct prompt: 1 is ending syllable, 0 is other
        ret = []
        for token in doc:
            # print(token.text,token.morph.get("Reading")[0])
            num_of_syllable = max(len(token.morph.get("Reading")[0]),1) # set this valuable to be minimum 1 for code robustness
            for ch in token.morph.get("Reading")[0]:
                if ch in self.small_word:
                    num_of_syllable -= 1
            assert num_of_syllable >= 0
            word_prompt = ['0' for i in range(num_of_syllable)]
            word_prompt[-1] = '1'
            ret += word_prompt
        if ret:
            ret[-1] = '0' # last syllable is not a boundary
        return ''.join(ret)


        # words = list(jieba.cut(s)) #中国語の形態素解析

        # # Construct prompt: 1 is ending syllable, 0 is other
        # ret = []
        # for i in range(len(words)):
        #     word = words[i]
        #     num_of_syllable = max(len(word.strip()), 1)  # set this valuable to be minimum 1 for code robustness
        #     word_prompt = ['0' for i in range(num_of_syllable)]
        #     if i != len(words) - 1:
        #         word_prompt[-1] = '1'
        #     ret += word_prompt
        # return ''.join(ret)

    def get_group_boundary(self, s):
        '''
        Get the (pseudo) melody boundary prompt for a given sentence,
        Which are randomly sampled from real word boundaries.
        s: a sentence in target language (chinese)
        return: a binary string representing the neccasary boundaries
        '''
        # words = list(jieba.cut(s)) #中国語の形態素解析
        doc = self.nlp(s)

        n = None  # number of boudaries
        p = random.uniform(0, 10)
        if p < 2:
            n = 0
        elif p < 6:
            n = 1
        elif p < 9:
            n = 2
        else:
            n = 3
        n = min(len(doc) - 1, n)
        # print('n=',n)

        # Sample the index of ending word of the n group, from [0, |words|-2]
        indices = set(random.sample(range(len(doc) - 1), n))

        # Construct prompt: 1 is ending syllable, 0 is other
        ret = []
        for i,token in enumerate(doc):
            num_of_syllable = len(token.morph.get("Reading")[0])
            for ch in token.morph.get("Reading")[0]:
                if ch in self.small_word:
                    num_of_syllable -= 1
            assert num_of_syllable >= 0, "Error in line: {}".format(s)
            word_prompt = ['0' for i in range(num_of_syllable)]
            if i in indices:
                word_prompt[-1] = '1'
            ret += word_prompt
        return ''.join(ret)


        # # Sample number of boundaries, distribution function: 2:4:3:1
        # n = None  # number of boudaries
        # p = random.uniform(0, 10)
        # if p < 2:
        #     n = 0
        # elif p < 6:
        #     n = 1
        # elif p < 9:
        #     n = 2
        # else:
        #     n = 3
        # n = min(len(words) - 1, n)
        # # print('n=',n)

        # # Sample the index of ending word of the n group, from [0, |words|-2]
        # # print(len(words), n)
        # indices = set(random.sample(range(len(words) - 1), n))

        # # Construct prompt: 1 is ending syllable, 0 is other
        # ret = []
        # for i in range(len(words)):
        #     word = words[i]
        #     num_of_syllable = len(word.strip())
        #     word_prompt = ['0' for i in range(num_of_syllable)]
        #     if i in indices:
        #         word_prompt[-1] = '1'
        #     ret += word_prompt
        # return ''.join(ret)

class BoundaryUtilEn:
    def __init__(self):
        pass

    def sample_boundaries(self, s, n=None):
        '''
        Get the (pseudo) melody boundary prompt for a given english sentence,
        Which are randomly sampled from real word boundaries.
        s: a sentence in source language (English)
        n: number of boundaries
        return: a binary string representing the necessary boundaries
        '''
        words = SyllableCounter.count_syllable_sentence(s, return_list=True)

        # Sample number of boundaries, distribution function: 2:4:3:1
        if n == None:
            p = random.uniform(0, 10)
            if p < 2:
                n = 0
            elif p < 6:
                n = 1
            elif p < 9:
                n = 2
            else:
                n = 3
            n = min(len(words) - 1, n)

        # Sample the index of ending word of the n group, from [0, |words|-2]
        # print(len(words), n)
        indices = set(random.sample(range(len(words) - 1), n))

        # Construct prompt: 1 is ending syllable, 0 is other
        ret = []
        for i in range(len(words)):
            # word = words[i]
            num_of_syllable = words[i]
            word_prompt = ['0' for i in range(num_of_syllable)]
            if i in indices:
                word_prompt[-1] = '1'
            ret += word_prompt
        return ''.join(ret)


class FluencyCaculator:
    unigram_probs = read_json(os.path.join(os.path.dirname(__file__), 'unigram_prob.json'))

    @classmethod
    def compute_slor(cls, sentence_list):
        '''
        @param
        sentence_list: a list of strings. The string should have been stripped.
        '''
        lens = [len(line) for line in sentence_list]
        lm_probs, perp = cls.compute_lm_probability(sentence_list)
        uni_probs = cls.compute_unigram_prob(sentence_list)
        slor = [(np.log(i) - np.log(j)) / k for i, j, k in zip(lm_probs, uni_probs, lens)]
        return slor, perp, lm_probs, uni_probs

    @classmethod
    def normalize_to_0_and_1(cls, slors, e_min, e_max):
        ret = [(e - e_min) / (e_max - e_min) for e in slors]
        return ret

    @classmethod
    def compute_unigram_prob(cls, sentence_list):
        '''
        Compute unigram probability of sentences
        '''

        def compute_unigram_prob_for_sentence(sentence):
            tot_prob = 1
            for ch in sentence:
                prob = cls.unigram_probs[ch] if ch in cls.unigram_probs else 1e-9
                tot_prob *= prob
            return tot_prob

        probabilities = [compute_unigram_prob_for_sentence(s) for s in sentence_list]
        return probabilities

    @classmethod
    def compute_lm_probability(cls, sentence_list):
        '''
        Compute LM probability of sentences
        '''
        lens = [len(line) for line in sentence_list]
        perplexities = cls.compute_perplexity(sentence_list)
        probabilities = [cls.perplexity_to_probability(pp, l) for pp, l in zip(perplexities, lens)]
        return probabilities, perplexities

    @classmethod
    def compute_perplexity(cls, sentence_list):
        '''
        Compute perplexity of sentences
        '''
        # from evaluate import load
        perplexity = MyPerplexity()
        results = perplexity.compute(predictions=sentence_list,
                                     model_id='uer/gpt2-chinese-lyric',
                                     add_start_token=True,
                                     batch_size=512)['perplexities']
        return results

    @classmethod
    def perplexity_to_probability(cls, pp, len):
        '''
        Convert perplexity of a sentence to its probability
        '''
        return 1 / pp ** len


# class MyPerplexity():

#     def compute(self, predictions, model_id, batch_size: int = 16, add_start_token: bool = True, device=None):

#         if device is not None:
#             assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
#             if device == "gpu":
#                 device = "cuda"
#         else:
#             device = "cuda" if torch.cuda.is_available() else "cpu"

#         model = AutoModelForCausalLM.from_pretrained(model_id)
#         model = model.to(device)

#         tokenizer = AutoTokenizer.from_pretrained(model_id)
#         tokenizer.bos_token_id = 101

#         # if batch_size > 1 (which generally leads to padding being required), and
#         # if there is not an already assigned pad_token, assign an existing
#         # special token to also be the padding token
#         if tokenizer.pad_token is None and batch_size > 1:
#             existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
#             # check that the model already has at least one special token defined
#             assert (
#                     len(existing_special_tokens) > 0
#             ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
#             # assign one of the special tokens to also be the pad token
#             tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

#         if add_start_token:
#             # leave room for <BOS> token to be added:
#             assert (
#                     tokenizer.bos_token is not None
#             ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
#             max_tokenized_len = model.config.max_length - 1
#         else:
#             max_tokenized_len = model.config.max_length

#         encodings = tokenizer(
#             predictions,
#             add_special_tokens=False,
#             padding=True,
#             truncation=True,
#             max_length=max_tokenized_len,
#             return_tensors="pt",
#             return_attention_mask=True,
#         ).to(device)

#         encoded_texts = encodings["input_ids"]
#         attn_masks = encodings["attention_mask"]

#         # check that each input is long enough:
#         if add_start_token:
#             assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
#         else:
#             assert torch.all(
#                 torch.ge(attn_masks.sum(1), 2)
#             ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

#         ppls = []
#         loss_fct = CrossEntropyLoss(reduction="none")

#         for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
#             end_index = min(start_index + batch_size, len(encoded_texts))
#             encoded_batch = encoded_texts[start_index:end_index]
#             attn_mask = attn_masks[start_index:end_index]

#             if add_start_token:
#                 bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
#                 encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
#                 attn_mask = torch.cat(
#                     [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
#                 )

#             labels = encoded_batch

#             with torch.no_grad():
#                 out_logits = model(encoded_batch, attention_mask=attn_mask).logits

#             shift_logits = out_logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

#             perplexity_batch = torch.exp(
#                 (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
#                 / shift_attention_mask_batch.sum(1)
#             )

#             ppls += perplexity_batch.tolist()

#         return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


# for Japanese
class RhymeCaculator:
    rhyme_6_dic = {
        1:["ア","カ","サ","タ","ナ","ハ","マ","ヤ","ラ","ワ","ァ","ャ","ガ","ザ","ダ","バ","パ"], #ア行 "a"
        2:["イ","キ","シ","チ","ニ","ヒ","ミ","リ","ィ","ギ","ジ","ヂ","ビ","ピ"], #イ行 "i"
        3:["ウ","ク","ス","ツ","ヌ","フ","ム","ユ","ル","ゥ","ュ","グ","ズ","ヅ","ブ","プ"], #ウ行 "u"
        4:["エ","ケ","セ","テ","ネ","ヘ","メ","レ","ェ","ゲ","ゼ","デ","ベ","ペ"], #エ行 "e"
        5:["オ","コ","ソ","ト","ノ","ホ","モ","ヨ","ロ","ヲ","ォ","ョ","ゴ","ゾ","ド","ボ","ポ"], #オ行 "o"
        6:["ン"], #ン行 "n"
    }

    # Get reverse dict of rhyme_6_dic
    t = {}
    for k in rhyme_6_dic:
        for v in rhyme_6_dic[k]:
            t[v] = k
    rhyme_6_dic_reverse = t

    special_alph = {"ッ", "ー"}

    nlp = spacy.load('ja_ginza')

    @classmethod
    def get_rhyme_type(cls, ch: str):
        if not is_katakana(ch):
            return 15

        try:
            ret = cls.rhyme_6_dic_reverse[ch] # ret = rhyme type in rhyme_14_dic
        except:
            import traceback
            traceback.print_exc()
            print(ch)
            exit(10)
        return ret

    @classmethod
    def get_rhyme_type_of_line(cls, line):
        '''
        Get the rhyme type of the end-line character of a line a Japanese texts
        The line can end with \n
        '''
        doc = cls.nlp(line)

        if len(line) == 0:
            return 7
        
        yomi = "".join([doc[-i].morph.get("Reading")[0] for i in range(min(2,len(doc)),0,-1)]) #ラストの最大二単語の読みを取得
        while yomi and yomi[-1] in cls.special_alph: #最後の文字が特殊文字なら削除
            yomi = yomi[:-1]
        if len(yomi)>0:
            ch = yomi[-1]
        else:
            print('Error line found!')
            ch = 'ア'
        # yomi = ""
        # for sent in doc.sents:
        #     for token in sent:
        #         if token.morph.get("Reading") == []:
        #             break
        #         yomi += token.morph.get("Reading")[0]
        # ch = yomi[-1]
        # if ch in cls.special_alph:
        #     ch = yomi[-2]
        return cls.get_rhyme_type(ch)

# For Japanese
class RhymeUtil:
    def __init__(self):
        self.rhyme_6_dic = {
            1:["ア","カ","サ","タ","ナ","ハ","マ","ヤ","ラ","ワ","ァ","ャ","ガ","ザ","ダ","バ","パ"], #ア行 "a"
            2:["イ","キ","シ","チ","ニ","ヒ","ミ","リ","ィ","ギ","ジ","ヂ","ビ","ピ"], #イ行 "i"
            3:["ウ","ク","ス","ツ","ヌ","フ","ム","ユ","ル","ゥ","ュ","グ","ズ","ヅ","ブ","プ"], #ウ行 "u"
            4:["エ","ケ","セ","テ","ネ","ヘ","メ","レ","ェ","ゲ","ゼ","デ","ベ","ペ"], #エ行 "e"
            5:["オ","コ","ソ","ト","ノ","ホ","モ","ヨ","ロ","ヲ","ォ","ョ","ゴ","ゾ","ド","ボ","ポ"], #オ行 "o"
            6:["ン"], #ン行 "n"
        }
        # Get reverse dict of rhyme_14_dic
        t = {}
        for k in self.rhyme_6_dic:
            for v in self.rhyme_6_dic[k]:
                t[v] = k
        self.rhyme_6_dic_reverse = t

        self.special_alph = {"ッ", "ー"}

        self.nlp = spacy.load('ja_ginza')

    def get_rhyme(self, ch: str, heteronym=True):
        '''
        ch: the character
        return: an integer in [1, 14] indicating the rhyme type
        ONLY compatible for obtaining rhyme dict
        '''
        if not is_katakana(ch):
            return 7
        return ch

    def get_rhyme_type(self, ch: str):
        '''
        ch: the character
        return: an integer in [1, 14] indicating the rhyme type
        '''
        if not is_katakana(ch):
            return 7
        try:
            ret = self.rhyme_6_dic_reverse[ch] # ret = rhyme type in rhyme_14_dic
        except:
            import traceback
            traceback.print_exc()
            print(ch)
            exit(10)
        return ret

    def get_rhyme_type_of_line(self, line):
        '''
        Get the rhyme type of the end-line character of a line a Japanese texts
        The line can end with \n
        '''
        doc = self.nlp(line.strip("\n"))

        if len(line) == 0:
            print('Empty line found!')
            ch = 'ア' #啊
        else:
            assert len(doc) > 0
            yomi = ""
            try: #ラストの最大二単語の読みを取得
                for i in range(min(2,len(doc)),0,-1):
                    if doc[-i].morph.get("Reading"):
                        yomi += doc[-i].morph.get("Reading")[0]
            except:
                print("Error in line: ", line)
                exit(10)
            while yomi and yomi[-1] in self.special_alph: #最後の文字が特殊文字なら削除
                yomi = yomi[:-1]
            if len(yomi)>0:
                ch = yomi[-1]
            else:
                print('Error line found!')
                ch = 'ア'
            # yomi = ""
            # for sent in doc.sents:
            #     for token in sent:
            #         if token.morph.get("Reading") == []:
            #             break
            #         yomi += token.morph.get("Reading")[0]
            # ch = yomi[-1]
            # if ch in self.special_alph:
            #     ch = yomi[-2]
        return self.get_rhyme_type(ch)


# class PerpCaculator:
#     def __init__(self):
#         self.model = torch.load(jpath(os.path.dirname(__file__), '../LM/ngram/models/4-gram-full.pt'))
#
#     def __call__(self, text):
#         return self.model.perplexity(text)


# class FluencyCaculator2:
#     unigram_probs = read_json(os.path.join(os.path.dirname(__file__), 'unigram_prob.json'))
#     perplexity = PerpCaculator()
#
#     @classmethod
#     def compute_slor(cls, sentence_list):
#         '''
#         @param
#         sentence_list: a list of strings. The string should have been stripped.
#         '''
#         lens = [len(line) for line in sentence_list]
#         lm_probs, perp = cls.compute_lm_probability(sentence_list)
#         uni_probs = cls.compute_unigram_prob(sentence_list)
#         slor = [(np.log(i) - np.log(j)) / k for i, j, k in zip(lm_probs, uni_probs, lens)]
#         return slor, perp, lm_probs, uni_probs
#
#     @classmethod
#     def normalize_to_0_and_1(cls, slors, e_min, e_max):
#         ret = [(e - e_min) / (e_max - e_min) for e in slors]
#         return ret
#
#     @classmethod
#     def compute_unigram_prob(cls, sentence_list):
#         '''
#         Compute unigram probability of sentences
#         '''
#
#         def compute_unigram_prob_for_sentence(sentence):
#             tot_prob = 1
#             for ch in sentence:
#                 prob = cls.unigram_probs[ch] if ch in cls.unigram_probs else 1e-9
#                 tot_prob *= prob
#             return tot_prob
#
#         probabilities = [compute_unigram_prob_for_sentence(s) for s in sentence_list]
#         return probabilities
#
#     @classmethod
#     def compute_lm_probability(cls, sentence_list):
#         '''
#         Compute LM probability of sentences
#         '''
#         lens = [len(line) for line in sentence_list]
#         perplexities = cls.compute_perplexity(sentence_list)
#         probabilities = [cls.perplexity_to_probability(pp, l) for pp, l in zip(perplexities, lens)]
#         return probabilities, perplexities
#
#     @classmethod
#     def compute_perplexity(cls, sentence_list):
#         '''
#         Compute perplexity of sentences
#         '''
#         # from evaluate import load
#
#         results = [cls.perplexity(s) for s in sentence_list]
#         return results
#
#     @classmethod
#     def perplexity_to_probability(cls, pp, len):
#         '''
#         Convert perplexity of a sentence to its probability
#         '''
#         return 1 / pp ** len

class TextCorrupterEn:
    @classmethod
    def corrupt_sentence(cls, s):
        '''
        s: the English sentence to be corrupted
        return: the corrupted sentence
        '''
        word_list = s.strip().split(' ')
        s_len = len(word_list)  # determine the sentence length
        lam = 1 if s_len <= 3 else 2.5  # determine lambda value
        mask_span = min(np.random.poisson(lam=lam), s_len - 1) # lambdaのポワソン分布から　二つ目の引数はsize
        if mask_span > 0:  # determine whether to corrupt the sentence
            possible_mask_pos = s_len - mask_span + 1
            mask_start_index = np.random.randint(possible_mask_pos)  # start index of the mask
            mask_end_index = mask_start_index + mask_span - 1  # end index of the mask
            assert 0 <= mask_start_index <= mask_end_index <= s_len - 1
            t1 = word_list[:mask_start_index]
            t2 = word_list[mask_end_index + 1:]
            ret = '{} <mask> {}'.format(' '.join(t1), ' '.join(t2))
            return ret
        else:
            return s

    @classmethod
    def corrupt_sentence_list(cls, l):
        '''
        prerequisite: no \n in the sentences in the l
        '''
        ret = [cls.corrupt_sentence(s) for s in l]
        return ret


class TextCorrupterCh:
    @classmethod
    def corrupt_sentence(cls, s):
        '''
        s: the Chinese sentence to be corrupted
        return: the corrupted sentence
        '''
        s_len = len(s.strip())  # determine the sentence length
        lam = 1 if s_len <= 3 else 3  # determine lambda value
        mask_span = min(np.random.poisson(lam=lam), s_len - 1)
        if mask_span > 0:  # determine whether to corrupt the sentence
            possible_mask_pos = s_len - mask_span + 1
            mask_start_index = np.random.randint(possible_mask_pos)  # start index of the mask
            mask_end_index = mask_start_index + mask_span - 1  # end index of the mask
            assert 0 <= mask_start_index <= mask_end_index <= s_len - 1
            t1 = s[:mask_start_index]
            t2 = s[mask_end_index + 1:]
            ret = '{}<mask>{}'.format(t1, t2)
            return ret
        else:
            return s

    @classmethod
    def corrupt_sentence_list(cls, l):
        '''
        prerequisite: no \n in the sentences in the l
        '''
        ret = [cls.corrupt_sentence(s) for s in l]
        return ret


class PosSeg:
    '''
    Segmentation and Part-of-speech tagging
    '''
    nlp = spacy.load('ja_ginza')

    def __init__(self):
        # jiebaでの前置詞、接続詞、助動詞、感嘆詞、ステータスワードの種類
        # self.unstressed_type = ['p', 'c', 'u', 'e', 'z']  # 介词、连词、助词、叹词、状态词

        # In Japnese we don't stress on 接置詞、助動詞、接続詞、等位接続詞、従属接続詞、限定詞、間投詞、助詞、接頭辞、接尾辞、記号、補助記号、空白、数詞、代名詞、句読点
        self.unstressed_type = ['ADP', 'AUX', 'CONJ', 'CCONJ', 'SCONJ', 'DET', 'INTJ', 'PART', 'PRON', 'PUNCT', 'SYM', 'X', 'SPACE', 'NUM']
        self.small_word = ["ャ","ュ","ョ","ァ","ィ","ゥ","ェ","ォ"]

    def pseg(self, s):
        '''
        Part-of-speech segmentation for a string (sentence)
        '''
        # 品詞のflagをつける。returnは[(word,flag)]*len(s)
        # return pseg.cut(s)

        doc = self.nlp(s)
        #日本語版は [(number of syllables, pos tag)]*len(s))]に変える
        tagged_words = []
        for token in doc:
            num_syllables = 0
            for ch in token.morph.get("Reading")[0]:
                if not ch in self.small_word: # ”ッ"以外の小文字はsyllableにカウントしない
                    num_syllables += 1
            tagged_words.append((num_syllables, token.pos_))
        return tagged_words

    def get_stress_constraint(self, s):
        '''
        Obtain stress constraint for a target Japanese sentence (from a label), by doing pos seg.
        s: sentence to be processed
        return: a string of '0' and '1'. 1 represent stressed words. 0 represent unstressed words.
            The length of the return string is equal to the number of syllabus in the sentence.
        '''
        tagged_words = self.pseg(s) # add pos tag
        # print(list(tagged_words))
        ret = []
        for len_word, tag in tagged_words:
            # print(word, tag)
            if tag in self.unstressed_type:
                tags = ['0' for i in range(len_word)] # 0 for unstressed type of word
            else:
                tags = ['1' for i in range(len_word)] # 1 for stressed type of word
            ret += tags
        return ''.join(ret)

    # def get_stress_constraint(self, s):
    #     '''
    #     Obtain stress constraint for a target Chinese sentence (from a label), by doing pos seg.
    #     s: sentence to be processed
    #     return: a string of '0' and '1'. 1 represent stressed words. 0 represent unstressed words.
    #         The length of the return string is equal to the number of syllabus in the sentence.
    #     '''
    #     tagged_words = self.pseg(s) # add pos tag
    #     # print(list(tagged_words))
    #     ret = []
    #     for word, tag in tagged_words:
    #         # print(word, tag)
    #         if tag[0] in self.unstressed_type:
    #             tags = ['0' for i in range(len(word))] # 0 for unstressed type of word
    #         else:
    #             tags = ['1' for i in range(len(word))] # 1 for stressed type of word
    #         # if word[-1] in ['着', '的', '了', '地', '得', '这', '那']:
    #         #     # some verb ends with these characters; we treat these as z
    #         #     # we don't treat "这" and "那" as 代词
    #         #     tags[-1] = '0'
    #         if len(word) == 1 and word[0] in ['这', '那']: # means "this" and "that"
    #             tags[0] = '0'
    #         ret += tags
    #     return ''.join(ret)

    def get_stress_pattern_list(self, s):
        '''
        Receive a Japanese sentence (from a generation output) string as input,
        Return a list representing the stress pattern.
        '''
        constraint_str = self.get_stress_constraint(s)
        ret = [int(i) for i in list(constraint_str)] # iは元々'0'or'1'。intに変換
        return ret


def calculate_acc(out, tgt):
    '''
    Calculate the ratio of same elements of two lists
    '''
    assert len(out) == len(tgt)
    cnt_same = 0
    for i in range(len(out)):
        if out[i] == tgt[i]:
            cnt_same += 1
    return cnt_same / len(out)


def calculate_acc_2d(out, tgt):
    '''
    Calculate the ratio of same elements of two lists of lists
    '''
    # print(len(out), len(tgt))
    assert len(out) == len(tgt)
    cnt_same = 0
    cnt_tot = 0
    for i in range(len(out)):
        for j in range(min(len(out[i]), len(tgt[i]))):
            cnt_tot += 1
            if out[i][j] == tgt[i][j]:
                cnt_same += 1
    return cnt_same / cnt_tot


def plot_dic(dic):
    '''
    Plot a dictionary, x-axis is key, y-axis is value
    '''
    import matplotlib.pyplot as plt
    t = list(dic.items())
    t.sort()
    x = list(zip(*t))[0]
    y = list(zip(*t))[1]
    plt.bar(x, y)
    plt.show()


def naive_syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

# For English
class SyllableCounter:
    d = cmudict.dict()
    special_word = read_json(jpath(os.path.dirname(__file__), 'special_word_syllables.json'))
    error_word = set()

    @classmethod
    def count_syllable_word(cls, word):
        word = word.strip(" :!?\"")

        if len(word) == 0:
            return 0

        if word in ["'ll", "n't", "'d", ",", "'ve", "'m", "'re", "'s"]:
            return 0

        if len(word) > 2 and word[-2:] == "'s":
            word = word[:-2]
        if word in cls.special_word:
            return cls.special_word[word]
        word = word.lower()
        t = [len(list(y for y in x if y[-1].isdigit())) for x in cls.d[word]]

        if len(t) >= 1:
            ret = t[0]
        else:
            # print(word)
            ret = naive_syllable_count(word)

            # if test == True:
            #     cls.error_word.add(word)
            #     ret = 1
            # else:
            #     print('Exception:', word)
            #     raise Exception
        return ret  # 可能会有多种读音，暂时取第一个

    @classmethod
    def count_syllable_sentence(cls, sentence, test=False, return_list=False):
        try:
            # words = sentence.strip().split(' ')  # this will lead to word combined with punctuation
            # sentence = re.sub("'(\w)+", " ", sentence)
            # words = re.findall(r'\w+', sentence) # I've
            words = nltk.word_tokenize(sentence)
            # print(words)
            # words = [i if i not in cls.special_word else cls.special_word[i] for i in words]
            # print(words)
            # words = [i for i in words if re.search(r'[\w]', i) != None]
            # print(words)
            word_syllables = [cls.count_syllable_word(word.lower().strip()) for word in words]
            # print(word_syllables)

            if return_list == True:  # return results in a list of integers
                ret_t = word_syllables
                ret = []
                for i in ret_t:
                    if i != 0:
                        ret.append(i)
                if len(ret) == 0:
                    ret = [1]
                return ret
            else:
                ret = sum(word_syllables)
                if ret == 0:
                    print('Zero syllable sentence detected!', sentence)
                    ret = 1
                return ret
        except Exception as e:
            raise e
            import traceback
            traceback.print_exc()
            print('The exception sentence:')
            print(sentence)
            exit(10086)

    @classmethod
    def count_syllable_sentence_batch(cls, batch):
        syllables = [cls.count_syllable_sentence(i) for i in batch]
        return syllables

# For Japanese
class SyllableCounterJA:
    nlp = spacy.load('ja_ginza')
    @classmethod
    def count_syllable_sentence(cls, sentence, test=False, return_list=False):
        small_word = ["ャ","ュ","ョ","ァ","ィ","ゥ","ェ","ォ"]
        try:
            # words = nltk.word_tokenize(sentence) # word毎に区切る
            # word_syllables = [cls.count_syllable_word(word.lower().strip()) for word in words]
            # # print(word_syllables)

            doc = cls.nlp(sentence)
            word_syllables = []
            for token in doc:
                if token.pos_ == 'PUNCT':
                    continue
                num_syllables = 0
                # print(token.morph.get("Reading"))
                if token.morph.get("Reading") == []:
                    continue
                for ch in token.morph.get("Reading")[0]:
                    if not ch in small_word:
                        num_syllables += 1
                word_syllables.append(num_syllables)
            
            # for sent in doc.sents:
            #     for token in sent:
            #         if token.pos_ == 'PUNCT':
            #             continue
            #         for item in token.morph.get("Reading"):
            #             word_syllables.append(len(item))

            if return_list == True:  # return results in a list of integers
                ret_t = word_syllables
                ret = []
                for i in ret_t:
                    if i != 0:
                        ret.append(i)
                if len(ret) == 0:
                    ret = [1]
                return ret
            else:
                ret = sum(word_syllables)
                if ret == 0:
                    print('Zero syllable sentence detected!', sentence)
                    ret = 1
                return ret
        except Exception as e:
            raise e
            import traceback
            traceback.print_exc()
            print('The exception sentence:')
            print(sentence)
            exit(10086)

    @classmethod
    def count_syllable_sentence_batch(cls, batch):
        syllables = [cls.count_syllable_sentence(i) for i in batch]
        return syllables


class StressUtilEn:
    '''
    Obtain stress descriptor from english sentence
    '''

    def __init__(self):
        self.unstressed_type = ['CC', 'DT', 'IN', 'TO', 'UH']  # 连词、限定词、介词、to、叹词

    def get_stress_from_sentence(self, s):
        '''
        Get stress descriptor from a sentence
        s: str, a sentence in english
        '''
        s = s.strip(" :!?\"")
        text = nltk.word_tokenize(s) # separate words and punctuation
        text = nltk.pos_tag(text)  # word with its pos tag
        # print(text)
        res = []
        for w in text:
            word, pos = w[0], w[1]
            num_syl = SyllableCounter.count_syllable_word(word) # test=False
            is_stress = 0 if pos in self.unstressed_type else 1
            res += [str(is_stress) for i in range(num_syl)]
        res = ''.join(res)
        # print(res)
        return res


def is_katakana(ch):
    if '\u30A1' <= ch <= '\u30FF':
        return True
    else:
        return False


if __name__ == '__main__':
    main()
