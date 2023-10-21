import openai
import os
import json
from tqdm import tqdm
import ipdb
from utils import *

openai.api_key = ""
file_path = './bionlp11/ALL_GPTNER_all.json'

if 'craft' in file_path: ava_etype = ['Species', 'Gene/Protein', 'Chemicals']
if 'bc5cdr' in file_path: ava_etype = ['Disease', 'Chemical']
if 'bionlp11' in file_path: ava_etype = ['Gene/Protein', 'Chemical']
if 'genia' in file_path: ava_etype = ['protein', 'RNA', 'DNA', 'cell_line', 'cell_type']
if 'bionlp13' in file_path: ava_etype = ['Gene/Protein', 'Chemical', 'Disease']

data = json.load(open(file_path))

def dec(span): return 1000*span[0] + span[1]

for key in tqdm(data.keys()):
    ele = data[key]
    span_set = set()
    preds = []
    preds_type = []
    for etype in ava_etype:

        if 'GPTNER' in file_path:
            spans = parse_span(ele['gpt_result'][etype])
        if 'UNINER' in file_path:
            spans = ele['gpt_result'][etype]

        for span in spans:
            if dec(span) not in span_set:
                span_set.add(dec(span))
                preds.append(span)
                preds_type.append(etype)

    data[key]['preds'] = preds
    data[key]['golds'] = data[key]['spans'] 
    data[key]['gold_types'] = data[key]['spans_type'] 
    data[key]['prepred_types'] = preds_type
    data[key]['preds_bem'] = [{'name': " ".join(data[key]['words'][sp[0]: sp[1]+1])} for sp in preds]
    # ipdb.set_trace()
    
names = file_path.split("/")
if 'GPT' in file_path: names2 = 'S1GPT_' + names[2].split("_")[2]
if 'UNI' in file_path: names2 = 'S1UNI_' + names[2].split("_")[2]
names[2] = names2
out_file_name = '/'.join(names)
print(out_file_name)

with open(out_file_name, 'w') as of:
    json.dump(data, of)
