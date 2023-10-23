from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
import re
from ast import literal_eval
import bz2
from tqdm import tqdm
import json

# https://huggingface.co/Norod78/hebrew-gpt_neo-small?text=%D7%A2%D7%95%D7%93+%D7%91%D7%99%D7%9E%D7%99+%D7%A7%D7%93%D7%9D


# https://metatext.io/datasets/cc100-hebrew
# http://data.statmt.org/cc-100/he.txt.xz
def load_cc100(raw_data_dir: Path) -> DatasetDict:
    root_dir = raw_data_dir / 'cc100-Hebrew'
    partition = load_dataset('text', name='cc100-Hebrew', data_files=[str(root_dir / 'he.txt')])
    return partition


# https://huggingface.co/datasets/HeNLP/HeDC4
def load_he_dc4() -> DatasetDict:
    # https://arxiv.org/pdf/2304.11077.pdf: HeRo: RoBERTa and Longformer Hebrew Language Models
    # @article{shalumov2023hero,
    #       title={HeRo: RoBERTa and Longformer Hebrew Language Models},
    #       author={Vitaly Shalumov and Harel Haskey},
    #       year={2023},
    #       journal={arXiv:2304.11077},
    # }
    # https://huggingface.co/HeNLP/HeRo
    # Combined the mC4 and the OSCAR 22.01 datasets for pre-training
    # Cleaned data using: https://gitlab.com/yhavinga/c4nlpreproc
    partition = load_dataset('HeNLP/HeDC4')
    return partition


# https://github.com/bdar-lab/heb_architecture_corpus
def load_heb_architecture_corpus(raw_data_dir: Path) -> DatasetDict:
    root_dir = raw_data_dir / 'heb_architecture_corpus'
    partition = load_dataset('text', name='heb_architecture_corpus', data_dir=str(root_dir / 'txt'))
    positions = [i for i, x in enumerate(partition['train']) if x['text'] == 'תקציר']
    article_num_arr = np.zeros(len(partition['train']), dtype=int)
    for i, (j, k) in enumerate(zip(positions[:-1], positions[1:])):
        article_num_arr[j:k] = i + 1
    return partition


# https://www.kaggle.com/datasets/guybarash/hebrew-songs-lyrics
def load_heb_songs_lyrics(raw_data_dir: Path) -> DatasetDict:
    root_dir = raw_data_dir / 'heb-songs-lyrics'
    partition = load_dataset('csv', name='heb-songs-lyrics', data_files=[str(root_dir / 'kaggle.csv')])
    texts = [' '.join(literal_eval(x['songs'])) for x in partition['train'].select(range(100))]
    return partition


# https://mega.nz/folder/CodSSA4R#4INvMes-56m_WUi7jQMbJQ
# https://huggingface.co/Norod78/hebrew-gpt_neo-small
def load_hebrew_text_corpus(raw_data_dir: Path) -> DatasetDict:
    root_dir = raw_data_dir / 'HebrewTextCorpus'
    partition = load_dataset('text', name='HebrewTextCorpus', data_dir=str(root_dir / 'HebrewTextCorpus' / 'text'))
    positions = [0] + [i+1 for i, x in enumerate(partition['train']) if x['text'].endswith('<|endoftext|>')]
    article_num_arr = np.zeros(len(partition['train']), dtype=int)
    for i, (j, k) in enumerate(zip(positions[:-1], positions[1:])):
        article_num_arr[j:k] = i + 1
    texts = [x['text'] for x in partition['train'].select(range(1000))]
    return partition


# https://huggingface.co/datasets/mc4
def load_mc4() -> DatasetDict:
    # Dataset Structure
    # Data Instances
    # An example form the en config is:
    #
    # {'timestamp': '2018-06-24T01:32:39Z',
    #  'text': 'Farm Resources in Plumas County\nShow Beginning Farmer Organizations & Professionals (304)\nThere are 304 resources serving Plumas County in the following categories:\nMap of Beginning Farmer Organizations & Professionals serving Plumas County\nVictoria Fisher - Office Manager - Loyalton, CA\nAmy Lynn Rasband - UCCE Plumas-Sierra Administrative Assistant II - Quincy , CA\nShow Farm Income Opportunities Organizations & Professionals (353)\nThere are 353 resources serving Plumas County in the following categories:\nFarm Ranch And Forest Retailers (18)\nMap of Farm Income Opportunities Organizations & Professionals serving Plumas County\nWarner Valley Wildlife Area - Plumas County\nShow Farm Resources Organizations & Professionals (297)\nThere are 297 resources serving Plumas County in the following categories:\nMap of Farm Resources Organizations & Professionals serving Plumas County\nThere are 57 resources serving Plumas County in the following categories:\nMap of Organic Certification Organizations & Professionals serving Plumas County',
    #  'url': 'http://www.californialandcan.org/Plumas/Farm-Resources/'}
    #
    # Data Fields
    # The data have several fields:
    #
    # url: url of the source as a string
    # text: text content as a string
    # timestamp: timestamp as a string
    partition = load_dataset("mc4", "iw")
    return partition


# https://www.kaggle.com/datasets/alvations/old-newspapers/
def load_old_newspapers(raw_data_dir: Path) -> DatasetDict:
    root_dir = raw_data_dir / 'old-newspapers'
    partition = load_dataset('csv', name='old-newspapers', data_files=[str(root_dir / 'old-newspaper-heb.tsv')], sep="\t")
    texts = [x['Text'] for x in partition['train'].select(range(100))]
    return partition


def load_oscar_2301() -> DatasetDict:
    partition = load_dataset('oscar-corpus/OSCAR-2301', 'he', token='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')
    partition['train'][0]['text'].split('\n'), partition['train'][0]['meta']['sentence_identifications']
    return partition


def load_oscar_2201() -> DatasetDict:
    return load_dataset('oscar-corpus/OSCAR-2201', 'he', token='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')


def load_oscar_2109() -> DatasetDict:
    return load_dataset('oscar-corpus/OSCAR-2109', 'original_he', token='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')


def load_oscar_2109_dedup() -> DatasetDict:
    return load_dataset('oscar-corpus/OSCAR-2109', 'deduplicated_he', token='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')


# https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset
def load_ted_talks(raw_data_dir: Path) -> DatasetDict:
    root_dir = raw_data_dir / 'ted_talks'
    partition = load_dataset('csv', name='ted_talks', data_files=[str(root_dir / '2020-05-01' / 'ted_talks_he.csv')])
    texts = [f"{x['title']}\n{x['description']}\n{x['transcript']}" for x in partition['train'].select(range(100))]
    return partition


# https://huggingface.co/datasets/LevMuchnik/SupremeCourtOfIsrael
def load_supreme_court() -> DatasetDict:
    partition = load_dataset('LevMuchnik/SupremeCourtOfIsrael')

    def clean(text: str) -> str:
        return '\n'.join([re.sub(r'\s+', ' ', line.strip()) for line in text.split('\n') if len(line.strip()) > 0])

    texts = [clean(x['text'].strip()) for x in partition['train'].select(range(100))]
    return partition


# https://dumps.wikimedia.org/backup-index-bydb.html
# https://github.com/apertium/WikiExtractor
# python WikiExtractor.py --infn ~/dev/aseker00/bclm/data/raw/wiki/hewiki/extracted/hewiki-20230920-pages-articles-multistream.xml > log
def load_wiki(raw_data_dir: Path) -> DatasetDict:
    root_dir = raw_data_dir / 'wiki'
    partition = load_dataset('text', name='wiki', data_dir=str(root_dir / 'hewiki' / 'extracted' / 'WikiExtractor' / 'text'))
    positions = [i for i, x in enumerate(partition['train']) if len(x['text']) == 0]
    article_num_arr = np.zeros(len(partition['train']), dtype=int)
    for i, (j, k) in enumerate(zip(positions[:-1], positions[1:])):
        article_num_arr[j:k] = i + 1
    return partition


def save_laion(dst_path: Path) -> DatasetDict:
    partition = load_dataset('laion/laion2B-multi', streaming=True)
    dataset = partition['train']
    with bz2.open(dst_path, 'wb') as f:
        for i, sample in tqdm(enumerate(dataset)):
            if sample['LANGUAGE'] == 'iw':
                f.write(json.dumps(sample).encode())
                f.write(b'\n')
    return partition


def load_laion() -> DatasetDict:
    partition = load_dataset('laion/laion2B-multi')

    def filter_lang(x: dict, lang: str) -> bool:
        return x['LANGUAGE'] == lang

    return DatasetDict({p: partition[p].filter(filter_lang) for p in partition})


def save_partition(partition: DatasetDict, dst_dir: Path):
    for split in partition:
        dataset = partition[split]
        with bz2.open(dst_dir / split / '.json.bz2', 'wb') as f:
            for sample in tqdm(dataset):
                f.write(json.dumps(sample).encode())
                f.write(b'\n')


def main():
    raw_data_dir = Path('data/raw')
    # partition = load_ted_talks(raw_data_dir)
    # partition = load_mc4()
    # partition = load_supreme_court()
    # partition = load_oscar_2301()
    laion = load_laion()
    laion_dst_dir = raw_data_dir / 'laion-iw'
    laion_dst_dir.mkdir(parents=True, exist_ok=True)
    save_partition(laion, laion_dst_dir)
    # partition = load_hebrew_text_corpus(raw_data_dir)
    # partition = load_old_newspapers(raw_data_dir)
    # partition = load_heb_songs_lyrics(raw_data_dir)
    # partition = load_heb_architecture_corpus(raw_data_dir)
    # partition = load_wiki(raw_data_dir)
    # print(len(partition['train']))


if __name__ == '__main__':
    main()
