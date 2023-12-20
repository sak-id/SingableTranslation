'''
Split songs into paragraphs
Generate constraints for input
'''

import os
import sys

DATA_MODE = "data_bt" # "data_bt" or "data_parallel"
CONSTRAINT_TYPE = "reference" # "reference" or "random" or "source"

DATA_DIR = '/raid/ieda/trans_jaen_dataset/Dataset/data_sources/{}/'.format(DATA_MODE)
OUT_DIR = '/raid/ieda/trans_jaen_dataset/Dataset/datasets/{}/'.format(DATA_MODE)
CONST_OUT_DIR = '/raid/ieda/trans_jaen_dataset/Dataset/datasets/{}/constraints2/{}/'.format(DATA_MODE, CONSTRAINT_TYPE)


def create_if_not_exist(p):
    if not os.path.exists(p):
        os.mkdir(p)

assert os.path.exists(DATA_DIR)
create_if_not_exist(OUT_DIR)
create_if_not_exist(CONST_OUT_DIR)

sys.path.insert(1, os.path.join(sys.path[0], '../../../'))

from utils_common.utils import ls, jpath, SyllableCounterJA, BoundaryUtilEn, RhymeUtil
from tqdm import tqdm

def _main():
    print("Start procedures")
    _procedures()


def _procedures():
    obtain_constraints(mode="train")
    print("Finish training dataset")
    obtain_constraints(mode="val")
    print("Finish validation dataset")
    obtain_constraints(mode="test")
    print("Finish testing dataset")

def obtain_constraints(mode):
    '''
    Obtain constraints for each of the paragraphs
    Target length: number of syllable in source text
    Target rhyme: rhy_0
    Target boundary: [bdr_0, bdr_0, ...]
    '''
    print("{}.source is processing...".format(mode))
    # for source (English)
    input_path = os.path.join(DATA_DIR, mode + '.source')
    assert os.path.exists(input_path)
    with open(input_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    
    # # Save source to directory "datasets"
    # out_path = jpath(OUT_DIR, mode + '.source')
    # with open(out_path, 'w', encoding='utf8') as f:
    #     f.writelines(lines)

    # Create boundary constraints
    boundary_util = BoundaryUtilEn()
    consts_bdr = []
    for line in lines:
        stress = boundary_util.sample_boundaries(line.strip()) #n=?
        consts_bdr.append(stress)
    
    # Save constraint of bdr
    out_path = jpath(CONST_OUT_DIR, mode + '_boundary.target') # save boundary constraints to ".target_boundary"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines([i + '\n' for i in consts_bdr])


    print("{}.target is processing...".format(mode))
    # for target (Japanese)
    input_path = os.path.join(DATA_DIR, mode + '.target')
    assert os.path.exists(input_path)
    with open(input_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    
    # # Save target to directory "datasets"
    # out_path = jpath(OUT_DIR, mode + '.target')
    # with open(out_path, 'w', encoding='utf8') as f:
    #     f.writelines(lines)

    # Create rhyme and length constraints
    rhyme_util = RhymeUtil()
    consts_rhy_len = []
    cnt = 0
    for line in lines:
        cnt += 1
        length = SyllableCounterJA.count_syllable_sentence(line.strip())
        rhyme = rhyme_util.get_rhyme_type_of_line(line.strip())
        consts_rhy_len.append('{}\t{}'.format(length, rhyme))
        if cnt % 100 == 0:
            print("Processed {} lines".format(cnt))
    
    # Save constraint of rhy and len
    out_path = jpath(CONST_OUT_DIR, mode + '.target') # save rhyme and length constraints to ".target"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines([i + '\n' for i in consts_rhy_len])
    


if __name__ == '__main__':
    _main()
