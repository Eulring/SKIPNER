import openai
import os
import json
from tqdm import tqdm
import ipdb
from utils import *


file_path = './genia/ALL_GPTNER_all.json'



if 'craft' in file_path: ava_etype = ['Species', 'Gene/Protein', 'Chemicals']
if 'bc5cdr' in file_path: ava_etype = ['Disease', 'Chemical']
if 'bionlp11' in file_path: ava_etype = ['Gene/Protein', 'Chemical']
if 'genia' in file_path: ava_etype = ['protein', 'RNA', 'DNA', 'cell_line', 'cell_type']
if 'bionlp13' in file_path: ava_etype = ['Gene/Protein', 'Chemical', 'Disease']

data = json.load(open(file_path))

dmd = {}
for ele in ava_etype: dmd[ele] = DL_metric()
dmd['All'] = DL_metric()

def dec(span):
    return 1000*span[0] + span[1]

for key in data.keys():
    ele = data[key]

    preds, pred_types = [], []

    span_set = set()
    for etype in ava_etype:
        if 'GPT' in file_path:
            spans = parse_span(ele['gpt_result'][etype])
        if 'UNI' in file_path:
            spans = ele['gpt_result'][etype]

        for span in spans:
            if span[0] > len(ele['words']): 
                continue
            if dec(span) not in span_set:
                span_set.add(dec(span))
                preds.append(span)
                pred_types.append(etype)

    golds = ele['spans']
    gold_types = ele['spans_type']

    for p_ in pred_types:
        for etype in ava_etype:
            if p_ == etype: dmd[etype].update(1, 'pred')
    for g_ in gold_types:
        for etype in ava_etype:
            if g_ == etype: dmd[etype].update(1, 'gold')
    

    dmd['All'].update(1, 'pred', len(preds))
    dmd['All'].update(1, 'gold', len(golds))

    nump = len(preds)
    cnt = 0
    for i in range(nump):
        pred = preds[i]
        pred_type = pred_types[i]
        pred_true_type = 'other'

        for j in range(len(golds)):
            if pred == golds[j]:
                dmd['All'].update(1, 'match')
                dmd[gold_types[j]].update(1, 'match')
                pred_true_type = gold_types[j]
                break

        if pred_type.lower() != 'other':
            dmd['All'].update(2, 'pred')
            dmd[pred_type].update(2, 'pred')

        if pred_true_type.lower() != 'other':
            dmd['All'].update(2, 'gold')
            dmd[pred_true_type].update(2, 'gold')
            if pred_type == pred_true_type:
                dmd[pred_type].update(2, 'match')
                dmd['All'].update(2, 'match')


# for etype in ava_etype:
#     print(etype + ": ") 
#     dmd[etype].compute()

print("All: ") 
P, R, F = dmd['All'].compute()
print("{:.2f} & {:.2f} & {:.2f}".format(P*100, R*100, F*100))
print(file_path)