import fasttext
from fasttext import load_model
import os
import re
from pathlib import Path
import pandas as pd
import csv, sys

csv.field_size_limit(sys.maxsize)


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--in_file', type=str, required=False, default='../data/INPI/new_extraction/output/inpi_new_final.csv')
parser.add_argument('--section_type', 
                    type=str, 
                    required=True,
                    choices={"title","abstract","description","claims"}, 
                    action = "append",
                    help="Patent sections to train the model.")

parser.add_argument('--dim', type=int, default=300)


args = parser.parse_args()

if __name__ == '__main__':

    # create output directory
    output_path = os.path.join('./output' ,'-'.join(['_'.join(args.section_type), 'Dim_' + str(args.dim)]))
    output_path = Path(output_path)

    if not output_path.exists():
        try:
            output_path.mkdir(parents=True)
            print(f"Created output directory {output_path}")
        except FileExistsError:
            print(f"{output_path} already exists!")

    corpus_file = os.path.join(output_path, 'train.cor')
    if not Path(corpus_file).exists():
        # corpus pre-processing
        df = pd.read_csv(args.in_file, dtype=str, engine="python")
        dict_target_secs = {'title': 'title', 'abstract': 'abs', 'claims': 'claims', 'description': 'desc'}
        target_sections = [dict_target_secs[s] for s in args.section_type]

        for sec in target_sections:
            df.loc[:,sec] = df[sec].apply(str)
        df.loc[:,'text'] = df[target_sections].apply('\n'.join, axis=1)
        lines = df['text'].tolist()

        # tokenize
        lines = [[token for token in re.split(r' |\.|\;|\,', l) if token != ''] for l in lines]
        print('Number of lines:', len(lines))

        # save corpus 
        with open(corpus_file, 'w') as out_f:
            out_f.write('\n'.join([' '.join(l) for l in lines]))


    model_file = os.path.join(output_path, 'model_ft.bin')
    if not Path(model_file).exists():
        # train the model
        model = fasttext.train_unsupervised(corpus_file, minn=2, maxn=5, dim=args.dim)
        # save model
        model.save_model(model_file)


    # save model as vec
    model = load_model(model_file)

    # get all words from model
    words = model.get_words()
    print('Number of tokens:', len(words))

    vec_file = os.path.join(output_path, 'model_ft.vec')
    with open(vec_file, 'w') as file_out:
    # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(model.get_dimension()) + "\n")

        # line by line, you append vectors to VEC file
        for w in words:
            v = model.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr+'\n')
            except:
                pass
