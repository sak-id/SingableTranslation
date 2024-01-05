# Description: Translate Japanese to Kana
# Run after prepare_dataset_finetune.py
# no need to create constraint file, it is already created in prepare_dataset_finetune.py
import spacy
from tqdm import tqdm

MODE = "data_bt" # data_bt | data_parallel

dataset_dir = "/raid/ieda/trans_jaen_dataset/Dataset/datasets/" + MODE + "/"

def main():
    print("Creating train.kana")
    process("train")
    print("Creating val.kana")
    process("val")
    print("Creating test.kana")
    process("test")
    print("Done")

nlp = spacy.load("ja_ginza")
    
def translate2kana(sentence):
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
        return line + "\n"
    except Exception as e:
        print(e)
        exit()

def process(mode):
    input_path = dataset_dir + mode + ".target"
    with open(input_path, "r") as fi:
        lines = fi.readlines()

    # create kana lines
    kana_lines = []
    for line in tqdm(lines):
        doc = nlp(line)
        kana_lines.append(translate2kana(line))
    
    # save
    output_path = dataset_dir + mode + ".kana"
    with open(output_path, "w") as fo:
        fo.writelines(kana_lines)

if __name__ == "__main__":
    main()