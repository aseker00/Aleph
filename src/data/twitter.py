# https://twitterdev.github.io/tweet_parser/index.html

from tweet_parser.tweet import Tweet
from tweet_parser.tweet_parser_errors import NotATweetError
import json
from pathlib import Path
import gzip
import bz2
import re
from tqdm import tqdm


def open_file(path: Path, attr):
    if path.suffix == '.gz':
        return gzip.open(path, attr)
    elif path.suffix == '.bz2':
        return bz2.open(path, attr)
    return None


def parse_line(tweet_line: str) -> list[dict]:
    try:
        return [json.loads(tweet_line)]
    except json.JSONDecodeError as e:
        if len(tweet_line) == 0:
            return []
        if tweet_line == 'NULL':
            return []
        if tweet_line == 'json':
            return []
        if tweet_line == 'Exceeded connection limit for user':
            return []
        if tweet_line.endswith('request Timedout.'):
            return []
        if e.pos == 0 or e.pos == 1:
            raise e
        # if e.msg == "Extra data":
        #     t1 = parse_line(tweet_line[:e.pos])
        #     t2 = parse_line(tweet_line[e.pos:])
        #     return t1 + t2
        if (e.msg == "Expecting ':' delimiter" or
                e.msg == "Expecting ',' delimiter" or
                e.msg == 'Expecting value' or
                e.msg == 'Extra data' or
                e.msg == 'Expecting property name enclosed in double quotes'):
            tweet_line1 = re.sub(r'([^\\])\\","ֿֿ([a-z_]*)":', r'\1\\\\","\2":', tweet_line)
            tweet_line2 = re.sub(r'\\\\"(?!,"[a-z_]*":)', r'\\"', tweet_line1)
            if tweet_line != tweet_line2:
                return parse_line(tweet_line2)
            if e.msg == "Extra data":
                t1 = parse_line(tweet_line[:e.pos])
                t2 = parse_line(tweet_line[e.pos:])
                return t1 + t2
            for offset in [2, 0, -2]:
                tweet_line = tweet_line2[e.pos + offset:]
                try:
                    return parse_line(tweet_line)
                except json.JSONDecodeError:
                    pass
        elif e.msg == 'Invalid \\escape':
            tweet_line = tweet_line[e.pos + 1:]
            try:
                return parse_line(tweet_line)
            except json.JSONDecodeError:
                pass
        elif e.msg == 'Invalid \\uXXXX escape':
            tweet_line = tweet_line[e.pos + 4:]
            try:
                return parse_line(tweet_line)
            except json.JSONDecodeError:
                pass
        raise e


def parse_file(input_path: Path) -> list[Tweet]:
    tweets = []
    in_f = open_file(input_path, 'rb')
    if not in_f:
        raise ValueError(f'Unhandled input file extension: {input_path.suffix}')
    for i, line in enumerate(in_f.readlines()):
        tweet_line = line.decode().strip()
        tweet_line = re.sub(r',"source":"<.*?>"', '', tweet_line)
        try:
            parsed_line = parse_line(tweet_line)
        except json.JSONDecodeError as e:
            print(f'{input_path}[{i}]: {e}')
            continue
        for obj in parsed_line:
            try:
                tweet = Tweet(obj)
            except (TypeError, NotATweetError):
                continue
            try:
                if tweet.lang == 'iw':
                    tweets.append(tweet)
            except KeyError:
                continue
    return tweets


def save_hebrew_tweets(in_path: Path, out_path: Path):
    tweets = parse_file(in_path)
    out_f = open_file(out_path, 'wb')
    if not out_f:
        raise ValueError(f'Unhandled input file extension: {out_path.suffix}')
    for tweet in tweets:
        out_f.write(json.dumps(tweet).encode())
        out_f.write(b'\n')
    out_f.close()


def process(in_dir: Path, out_dir: Path):
    in_paths = sorted(in_dir.glob('*.gz'))
    for i in tqdm(range(len(in_paths))):
        in_path = in_paths[i]
        out_path = out_dir / in_path.name
        if out_path.exists():
            continue
        save_hebrew_tweets(in_path, out_path)


parse_file(Path('tweets007255.json.gz'))

data_root_path = Path('/media/amit/907A-DFE1/data/twitter')
text_root_path = Path('/home/amit/data/twitter')

file_paths = [p for p in (data_root_path / 'sample-archive').rglob("*.gz")]
print(len(file_paths))
data_dirs = sorted({p.parents[0].relative_to(data_root_path) for p in file_paths})
print(data_dirs)

for data_dir in data_dirs:
    input_dir = data_root_path / data_dir
    output_dir = text_root_path / data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Processing {output_dir} ...')
    process(input_dir, output_dir)
