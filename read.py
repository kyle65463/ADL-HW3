import json

def read_jsonl(filename):
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(lines[1])
            print(data['title'])
            break

read_jsonl('data/train.jsonl')