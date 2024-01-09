'''
Modify original mbart tokenizer for Chinese
1. Remove word that have multiple characters
2. Add out-of-vocabulary word
(Step 2 is probably not going to remove_multi_character_word because embedding dimension have to be changed)
'''
"""
Modified for Japanese
"""
import os
import sys
import re
from transformers import MBart50TokenizerFast
from tqdm import tqdm
sys.path.insert(1, os.path.normpath(os.path.join(sys.path[0], '../')))
sys.path.insert(2, os.path.normpath(os.path.join(sys.path[0], '../../')))
# sys.path.insert(1, os.path.join(sys.path[0], '../../'))
from utils.utils import *
print("library imported")

# TOKENIZER_NAME = './mbart_tokenizer_fast_ja_prefix_lrs_notsep' # tokenizer name for saving
TOKENIZER_NAME = './mbart_tokenizer_fast_ja_add_tokens_notsep'
DEFAULT_NAME = './mbart_tokenizer_fast'

def _main():
    pass
    tokenizer_load(TOKENIZER_NAME)
    exit()

def procedures():
    # 一文字ずつ分割する場合はこちら
    # ./mbart_tokenizer_fast_ja_prefix_lrs
    remove_multi_character_word() #done
    check_modification()
    obtain_bad_word_ids() #done
    obtain_bad_word_list() #done

    add_special_tokens_for_prefix()

def ja_procedures():
    #　一文字ずつ分割しなくてOKならこちら
    # ./mbart_tokenizer_fast_ja_prefix_lrs_notsep　とかにしたい
    tokenizer_save()
    tokenizer_load()

    add_special_tokens_for_prefix()


def add_special_tokens_for_prefix():
    '''
    Add prefix tokens, 36 in total
    For the implementation of PoeLM method
    '''
    prefix_tokens = ['<pref>', '</pref>']

    num_len_token = 20
    num_rhyme_token = 6
    len_tokens = ['len_{}'.format(i) for i in range(1, num_len_token + 1)]
    rhyme_tokens = ['rhy_{}'.format(i) for i in range(0, num_rhyme_token + 1)] # 0 refers to no constraints
    stress_tokens = ['str_{}'.format(i) for i in [0,1]]
    doc_tokens = ['<brk>']
    """
    # result of special tokens: 
    ['<s>', '</s>', '<unk>', '<pad>', '<mask>', '<pref>', '</pref>', 
    'len_1', 'len_2', 'len_3', 'len_4', 'len_5', 'len_6', 'len_7', 'len_8', 'len_9', 'len_10', 'len_11', 'len_12', 'len_13', 'len_14', 'len_15', 'len_16', 'len_17', 'len_18', 'len_19', 'len_20', 
    'rhy_0', 'rhy_1', 'rhy_2', 'rhy_3', 'rhy_4', 'rhy_5', 'rhy_6', 'str_0', 'str_1', '<brk>']
    """

    tokenizer = MBart50TokenizerFast.from_pretrained('./mbart_tokenizer_fast')
    print("tokenizer loaded")
    # additional_special_tokens: tokens in this list will not be split by the tokenizer
    tokenizer.add_special_tokens({
        'additional_special_tokens': prefix_tokens + len_tokens + rhyme_tokens + stress_tokens + doc_tokens
    })

    tokenizer.save_pretrained(TOKENIZER_NAME, legacy_format=False) # only saved for json file


def tokenizer_save():
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    tokenizer.save_pretrained('./mbart_tokenizer_fast', legacy_format=False)
    # tokenizer.save_pretrained(TOKENIZER_NAME, legacy_format=False)


def tokenizer_load(tokenizer_name):
    tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_name) #'./mbart_tokenizer_fast'
    print(tokenizer.tokenize('夢が落ちてく場所をずっとずっと探してた'))
    print(tokenizer('夢が落ちてく場所をずっとずっと探してた'))
# ['▁', '夢', 'が', '落ち', 'て', 'く', '場所', 'を', 'ずっと', 'ずっと', '探', 'してた']
# {'input_ids': [250004, 6, 19101, 281, 67004, 3014, 2928, 21961, 251, 79824, 79824, 15362, 176539, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}


def play_mbart_tokenizer():
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    print(tokenizer.lang_code_to_id["ja_XX"])  # 250012
    print(tokenizer.lang_code_to_id["en_XX"])  # 250004

"""
def play_tokenizer_training():
    tokenizer_old = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    tokenizer_old.src_lang = "en_XX"
    tokenizer_old.tgt_lang = 'zh_CN'
    # example = '让我们荡起双桨'
    # tokenizer_old.tgt_lang = 'zh_CN'
    # print(tokenizer_old.src_lang, tokenizer_old.tgt_lang)
    # with tokenizer_old.as_target_tokenizer():
    #     print(tokenizer_old._convert_id_to_token(6))
    #     tokens = tokenizer_old(example)
    #     print(tokens)

    chars = ['我', '是', '谁', '哈', '嘻']

    with open('corpus.txt', encoding='utf8') as f:
        text = f.readlines()
    text = [i.strip() for i in text]

    def get_training_corpus(text):
        bs = 2
        for start_idx in range(0, len(text), bs):
            samples = text[start_idx: start_idx + bs]
            yield samples

    text_iter = get_training_corpus(text)

    tokenizer_new = tokenizer_old.train_new_from_iterator(text_iterator=text_iter, vocab_size=100,
                                                          # initial_alphabet=chars,
                                                          # max_piece_length=1
                                                          )
    # with tokenizer_old.as_target_tokenizer():

    # with tokenizer_new.as_target_tokenizer():
    print(tokenizer_new.tokenize('我是谁'))
    print(tokenizer_new('我是谁'))
    save_json(tokenizer_new.vocab, './vocab_new.json')

    # from tokenizers import models, trainers, Tokenizer, pre_tokenizers
    # tokenizer = Tokenizer(models.BPE())
    # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # print(tokenizer.model)
    # print(tokenizer.pre_tokenizer.pre_tokenize_str('我 是 谁'))
    # tokenizer.model = models.BPE()
    # tokenizer = tokenizer_old.train_new_from_iterator(training_corpus, 52000)
"""
"""
def obtain_zh_character_dic():
    '''
    Obtain all the appeared chinese character and frequency in the corpus
    Save to misc/dic_sorted.json
    '''
    # dic = {}
    # data = read_json('../../Dataset/data_full/v6/lyrics_v6_bt_clean.json')
    # for id in tqdm(data):
    #     lyric_ch = data[id]['ch']
    #     for line in lyric_ch:
    #         for ch in line:
    #             if ch not in dic:
    #                 dic[ch] = 1
    #             else:
    #                 dic[ch] +=1
    # save_json(dic, 'misc/dic_raw.json')

    dic = read_json('misc/dic_raw.json')
    x = dic
    y = {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}
    save_json(y, 'misc/dic_sorted.json')
"""

def remove_multi_character_word():
    # we have to remove multi-character word from original Mbart tokenizer, because we count the number of character in the word
    '''
    Deactivate some tokens in the tokenizer (ch part only) by substituting the entry to 'DEACTIVATED_TOKEN'
    Below are some types of tokens need to be deactivated:
        - multi-char word
        - entry that start with '▁' like '▁我'
    '''
    deactivated_label = 'DEACTIVATED_TOKEN'
    tk = read_json('./mbart_tokenizer_fast/tokenizer.json')
    vocab = tk['model']['vocab']
    #pattern_ch = '[\u4e00-\u9fff]'
    pattern_ja = '[\u3041-\u309F\u30A0-\u30FF\u4E00-\u9FFF]'
    pattern_katakana = '[\u30A0-\u30FF]'
    deactivated_tokens = []
    ok_tokens = []
    for i in range(len(vocab)):
        entry = vocab[i]
        if re.search(pattern_ja, entry[0]):  # If contain Japanese character
            if len(entry[0])>1 or not re.fullmatch(pattern_katakana, entry[0]): # if multi-char word or not katakana
                deactivated_tokens.append(entry[0])
                entry[0] = deactivated_label
            else:
                ok_tokens.append(entry[0])
    print(*ok_tokens)
    save_json(tk, './mbart_tokenizer_fast/tokenizer.json') # need change
    save_json(deactivated_tokens, './misc/deactivated_tokens.json')


def check_modification():
    tokenizer = MBart50TokenizerFast.from_pretrained('./mbart_tokenizer_fast_ja_prefix_lrs')
    tokenizer.src_lang = 'en_XX'
    tokenizer.tgt_lang = 'ja_XX'
    check_tokenize(tokenizer)


def check_tokenize(tokenizer):
    '''
    Check how will the modified tokenizer perform
    '''
    t = tokenizer.get_vocab()
    print(len(t))

    with tokenizer.as_target_tokenizer():
        example = '夢が落ちてく場所をずっとずっと探してた'
        print(tokenizer(example))
        print(tokenizer.convert_ids_to_tokens(tokenizer(example)['input_ids']))

        example = '日本史は苦手'
        print(tokenizer.tokenize(example))
        print(tokenizer(example))
    # print(tokenizer.all_special_tokens)
    # tokenizer.save_pretrained('./save_test')
"""
237347
{'input_ids': [250012, 6, 19101, 281, 6656, 25116, 3014, 2928, 5287, 1493, 251, 21862, 16974, 610, 21862, 16974, 610, 15362, 1373, 3014, 2427, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
['ja_XX', '▁', '夢', 'が', '落', 'ち', 'て', 'く', '場', '所', 'を', 'ず', 'っ', 'と', 'ず', 'っ', 'と', '探', 'し', 'て', 'た', '</s>']
['▁', '日', '本', '史', 'は', '苦', '手']
{'input_ids': [250012, 6, 635, 1516, 19312, 342, 11705, 1634, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""
def obtain_bad_word_ids():
    '''
    Add all multi-char Chinese word to a bad word list for mBart generation
    The result is in format of [[6, id], ... ]
    '''
    tokenizer = MBart50TokenizerFast.from_pretrained('./mbart_tokenizer_fast')
    bad_words = read_json('./misc/deactivated_tokens.json')
    bad_word_ids = []
    # for i in bad_words:
    t = tokenizer(bad_words, add_special_tokens=False).input_ids
    # print(i, t)

    save_json(t, './misc/bad_word_ids.json')


def obtain_bad_word_list():
    '''
    Convert bad_word_ids to a list of bad word ids for convenient generation
    '''
    ids = read_json('./misc/bad_word_ids.json')
    res = []
    for entry in ids:
        res.append(entry[-1])
    res.sort()
    save_json(res, './misc/bad_word_list.json')


if __name__ == '__main__':
    _main()
