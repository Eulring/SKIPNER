import openai
import os
import json
from tqdm import tqdm
import ipdb
from utils import *


file_path = './genia/S1GPT_S2dict-uni12_S3cot-3-k10_all.json'


if 'craft' in file_path: ava_etype = ['Species', 'Gene/Protein', 'Chemicals']
if 'bc5cdr' in file_path: ava_etype = ['Disease', 'Chemical']
if 'bionlp11' in file_path: ava_etype = ['Gene/Protein', 'Chemical']
if 'genia' in file_path: ava_etype = ['protein', 'RNA', 'DNA', 'cell_line', 'cell_type']
if 'bionlp13' in file_path: ava_etype = ['Gene/Protein', 'Chemical', 'Disease']


print(file_path)
data = json.load(open(file_path))

dmd = {}
for ele in ava_etype: dmd[ele] = DL_metric()
dmd['All'] = DL_metric()


def dec(span): return 1000*span[0] + span[1]

def parse_response(item):
    ans = []

    if 'cot' in file_path:
        for res in item['gpt_type']:
            for etype in ava_etype:
                if etype.lower() in res.lower():
                    res = etype
            if res not in ava_etype:
                res = 'other'
            ans.append(res)

    if 'greedy' in file_path or 'vote' in file_path:
        for res in item['gpt_type']:
            ans.append(res)

    if 'ALL_GPTNER' in file_path:
        ipdb.set_trace()

    return ans


for key in data.keys():
    ele = data[key]
    preds = ele['preds']
    pred_types = parse_response(ele)

    for i in range(len(pred_types)):
        if 'craft' in file_path:
            if pred_types[i].lower() in ['chemical']: 
                pred_types[i] = 'Chemicals'
        if 'genia' in file_path:
            if pred_types[i].lower() in ['gene', 'protein']: 
                pred_types[i] = 'protein'
        elif pred_types[i].lower() in ['gene', 'protein']: pred_types[i] = 'Gene/Protein'

        if pred_types[i] not in ava_etype: pred_types[i] = 'other'
        
    golds = ele['golds']
    gold_types = ele['gold_types']

    for p_ in pred_types:
        for etype in ava_etype:
            if p_ == etype: dmd[etype].update(1, 'pred')
    for g_ in gold_types:
        for etype in ava_etype:
            if g_ == etype: dmd[etype].update(1, 'gold')

    dmd['All'].update(1, 'pred', len(preds))
    dmd['All'].update(1, 'gold', len(golds))

    nump = len(ele['preds'])
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
