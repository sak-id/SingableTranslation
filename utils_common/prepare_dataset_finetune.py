'''
Split songs into paragraphs
Generate constraints for input
'''

import os
import sys
import random

DATA_MODE = "data_bt" # data_bt | data_parallel
# CONSTRAINT_TYPE = "reference" # reference | random | source

DATA_DIR = '/raid/ieda/trans_jaen_dataset/Dataset/data_sources/{}/full/'.format(DATA_MODE)
# DATA_DIR = '/raid/ieda/trans_jaen_dataset/Dataset/data_sources/{}/'.format(DATA_MODE) # 実験のために一部のデータで処理を確認
OUT_DIR = '/raid/ieda/trans_jaen_dataset/Dataset/datasets/{}/full'.format(DATA_MODE) 
# CONST_OUT_DIR = '/raid/ieda/trans_jaen_dataset/Dataset/datasets/{}/constraints/{}/'.format(DATA_MODE, CONSTRAINT_TYPE)
CONST_OUT_DIR = os.path.join(OUT_DIR, 'constraints')

def create_if_not_exist(p):
    if not os.path.exists(p):
        os.mkdir(p)

assert os.path.exists(DATA_DIR)
create_if_not_exist(OUT_DIR)
assert os.path.exists(CONST_OUT_DIR)

# sys.path.insert(1, os.path.join(sys.path[0], '../../../'))

from utils import (
    ls, jpath,
    # SyllableCounterJA,
    BoundaryUtilEn,
    # RhymeUtil, # for Ja
    PosSeg, # for Ja
    SyllableCounter,
    BoundaryUtil, # for Ja
    StressUtilEn,
)
from tqdm import tqdm
import spacy

nlp = spacy.load("ja_ginza")
small_word = ["ャ","ュ","ョ","ァ","ィ","ゥ","ェ","ォ"]

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


error_log = []

def _main():
    print("Start procedures")
    train_procedures()
    print("Finish train procedures")
    val_procedures()
    print("Finish val procedures")
    test_procedures()
    print("Finish test procedures")
    write_error_log()

def train_procedures():
    train_source = load_save_text('train.source')
    train_target = load_save_text('train.target')
    train_kana = translate2kana(train_target, 'train')
    reference_rhylen(texts=train_kana, mode='train') # create train.target
    # reference_boundary(texts=train_target, mode='train') # create train_boundary.target
    # reference_stress(texts=train_target, mode='train') # create train_stress.target
    random_rhylen(texts=train_target, mode='train') # create /random/train.target

def val_procedures():
    val_source = load_save_text('val.source')
    val_target = load_save_text('val.target')
    val_kana = translate2kana(val_target, 'val')
    reference_rhylen(texts=val_kana, mode='val') # create /reference/val.target
    # reference_boundary(texts=val_target, mode='val') # create /reference/val_boundary.target
    # reference_stress(texts=val_target, mode='val') # create /reference/val_stress.target
    random_rhylen(texts=val_target, mode='val') # create /random/val.target
    source_rhylen(texts=val_source, mode='val') # create /source/val.target
    # source_boundary(texts=val_source, mode='val') # create /source/val_boundary.target
    # source_stress(texts=val_source, mode='val') # create /source/val_stress.target

def test_procedures():
    test_source = load_save_text('test.source')
    test_target = load_save_text('test.target')
    test_kana = translate2kana(test_target, 'test')
    reference_rhylen(texts=test_kana, mode='test') # create /reference/test.target
    # reference_boundary(texts=test_target, mode='test') # create /reference/test_boundary.target
    # reference_stress(texts=test_target, mode='test') # create /reference/test_stress.target
    random_rhylen(texts=test_target, mode='test') # create /random/test.target
    source_rhylen(texts=test_source, mode='test') # create /source/test.target
    # source_boundary(texts=test_source, mode='test') # create /source/test_boundary.target
    # source_stress(texts=test_source, mode='test') # create /source/test_stress.target

def load_save_text(file_name): # load texts from dataset from "data_source", save to "datasets"
    # load texts from dataset
    input_path = os.path.join(DATA_DIR, file_name)
    assert os.path.exists(input_path)
    with open(input_path, 'r', encoding='utf8') as f:
        texts = f.readlines()

    # save texts to directory "datasets"
    out_path = jpath(OUT_DIR, file_name)
    if os.path.exists(out_path):
        return texts
    with open(out_path, 'w', encoding='utf8') as f:
        f.writelines(texts)
    return texts

def translate2kana(texts,mode):
    out_path = jpath(OUT_DIR, mode + '.kana')
    if os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf8') as f:
            return f.readlines()
    # create kana lines
    kana_lines = []
    for sentence in tqdm(texts):
        try:
            doc = nlp(sentence)
            line = ""
            for token in doc:
                if token.pos_ in ['PUNCT', 'SYM', 'SPACE', 'X']:
                    continue
                # print(token.morph.get("Reading"))
                if token.morph.get("Reading") == []:
                    continue
                line += token.morph.get("Reading")[0]
            kana_lines.append(line + "\n")
            # print(line)
        except Exception as e:
            print(e)
            exit()
    with open(out_path, "w") as fo:
        fo.writelines(kana_lines)
    return kana_lines

def calc_syllables(kana_line):
    length = 0
    for ch in list(kana_line.strip()):
        if not ch in small_word:
            length += 1
    return length

def calc_rhyme_type(kana_line):
    last_ch = kana_line.strip()[-1]
    if last_ch in special_alph:
        last_ch = kana_line.strip()[-2]
    return rhyme_6_dic_reverse[last_ch]

def reference_rhylen(texts, mode):
    out_path = jpath(CONST_OUT_DIR, "reference", mode + '.target')
    # Create rhyme and length constraints
    # rhyme_util = RhymeUtil()
    consts_rhy_len = []
    for i,line in enumerate(tqdm(texts)):
        try:
            # length = SyllableCounterJA.count_syllable_sentence(line.strip())
            length = calc_syllables(line)
            if length > 20:
                error_log.append("Error in line {} of {}: {}".format(i, mode, line))
                length = 20
        except Exception as e:
            length = random.randint(3, 20)
            error_log.append("Error in line {} of {}: {}".format(i, mode, line))
        try:
            # rhyme = rhyme_util.get_rhyme_type_of_line(line.strip())
            rhyme = calc_rhyme_type(line)
        except Exception as e:
            rhyme = random.randint(1, 6)
            error_log.append("Error in line {} of {}: {}".format(i, mode, line))
        consts_rhy_len.append('{}\t{}'.format(length, rhyme))
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines([i + '\n' for i in consts_rhy_len])
    return

def reference_boundary(texts, mode):
    out_path = jpath(CONST_OUT_DIR, "reference", mode + '_boundary.target')
    # Create boundary constraints
    boundary_util = BoundaryUtil()
    consts_bdr = []
    for line in tqdm(texts):
        stress = boundary_util.get_group_boundary(line.strip()) #n=?
        consts_bdr.append(stress)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines([i + '\n' for i in consts_bdr])
    return

def reference_stress(texts, mode):
    out_path = jpath(CONST_OUT_DIR, "reference", mode + '_stress.target')
    # Create stress constraints
    consts_str = []
    stress_util = PosSeg()
    for line in tqdm(texts):
        stress = stress_util.get_stress_constraint(line.strip())
        consts_str.append(stress)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines([i + '\n' for i in consts_str])
    return

def random_rhylen(texts, mode):
    out_path = jpath(CONST_OUT_DIR, "random", mode + '.target')
    # randomly generate rhyme and length constraints
    consts_rhy_len = []
    for i in tqdm(range(len(texts))):
        length = random.randint(3, 20)
        rhyme = random.randint(1, 6)
        consts_rhy_len.append('{}\t{}'.format(length, rhyme))
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines([i + '\n' for i in consts_rhy_len])
    return

def source_rhylen(texts,mode):
    # Create rhyme and length constraints
    # rhyme is randomly generated (English sentence cannot decide Japanese rhyme type)
    out_path = jpath(CONST_OUT_DIR, "source", mode + '.target')
    syllable_util = SyllableCounter()
    consts_rhy_len = []
    for i,line in enumerate(tqdm(texts)):
        try:
            length = syllable_util.count_syllable_sentence(line.strip())
            # length = calc_syllables(line)
            # if length > 20:
            #     error_log.append("Error in line {} of {}: {}".format(i, mode, line))
            #     length = 20
        except Exception as e:
            length = random.randint(3, 20)
            error_log.append("Error in line {} of {}: {}".format(i, mode, line))
        rhyme = random.randint(1, 6)
        consts_rhy_len.append('{}\t{}'.format(length, rhyme))
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines([i + '\n' for i in consts_rhy_len])
    return


def source_boundary(texts,mode):
    out_path = jpath(CONST_OUT_DIR, "source", mode + '_boundary.target')
    # Create boundary constraints
    boundary_util = BoundaryUtilEn()
    consts_bdr = []
    for line in tqdm(texts):
        stress = boundary_util.sample_boundaries(line.strip()) #n=?
        consts_bdr.append(stress)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines([i + '\n' for i in consts_bdr])
    return

def source_stress(texts,mode):
    out_path = jpath(CONST_OUT_DIR, "source", mode + '_stress.target')
    # Create stress constraints
    consts_str = []
    stress_util = StressUtilEn()
    for line in tqdm(texts):
        stress = stress_util.get_stress_from_sentence(line.strip())
        consts_str.append(stress)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines([i + '\n' for i in consts_str])

def write_error_log():
    if not error_log:
        return
    with open(os.path.join(OUT_DIR, 'error_log.txt'), 'w', encoding='utf8') as f:
        f.writelines(error_log)
    return

if __name__ == '__main__':
    _main()
