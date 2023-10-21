import openai
import os
import json
from tqdm import tqdm
import ipdb
from utils import *

file_path = './bionlp11/S1GPT_S2dict-uni12_all.json'
openai.api_key = ""
method_name = 'cot-2'
resume = False

if 'craft' in file_path: ava_etype = ['Species', 'Gene/Protein', 'Chemicals']
if 'bc5cdr' in file_path: ava_etype = ['Disease', 'Chemical']
if 'bionlp11' in file_path: ava_etype = ['Gene/Protein', 'Chemical']
if 'genia' in file_path: ava_etype = ['protein', 'RNA', 'DNA', 'cell_line', 'cell_type']
if 'bionlp13' in file_path: ava_etype = ['Gene/Protein', 'Chemical', 'Disease']




# output file names
names = file_path.split("_")
names[-1] = 'S3'+method_name+'_'+names[-1]
out_file_name = "_".join(names)
print(out_file_name)


save_interval = 50
if resume: data = json.load(open(out_file_name))
else: data = json.load(open(file_path))

def gptapi_cot2(sent, ename, etype):
    entities = ['\'{}\''.format(ele) for ele in ava_etype]
    prompt1 = """You are a biomedical expert. The task is to verify whether the word {} is {} entity extracted from the given sentence.""".format(ename, etype)
    prompt2 = "Input sentence: '{}'".format(sent)
    prompt3 = "Is the word '{}' in the input sentence a '{}' entity? Please answer with yes or no ".format(ename, etype)
    # prompt5 = "Now direct output the category of entity ' {} ', Your output must be one of 'Chemicals', 'Species', 'Gene/Protein', 'other'. ".format(ename)

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',        
        messages=[
            {"role": "system", "content": prompt1},
            {"role": "user", "content": prompt2 + "\n" + prompt3},
        ]
    )
    # ipdb.set_trace()
    resText = response.choices[0].message.content
    return resText

def gptapi_cot4(sent, ename, etype, ptypes, pnames):
    entities = ['\'{}\''.format(ele) for ele in ava_etype]
    prompt1 = """You are a biomedical expert. The task is to verify whether the word {} is {} entity extracted from the given sentence.""".format(ename, etype)
    prompt2 = "Input sentence: '{}'".format(sent)
    prompt4 = "Entity names similar to '{}' and their corresponding categories are arranged in order based on their similarity as follows: \n".format(ename)
    for i in range(3):
        if i == 0: 
            prompt4 += "Most similar entity: {}, and its category: {}\n".format(pnames[i], ptypes[i])
        elif i == 1:
            prompt4 += "2nd similar entity: {}, and its category: {}\n".format(pnames[i], ptypes[i])
        elif i == 2:
            prompt4 += "3rd similar entity: {}, and its category: {}\n".format(pnames[i], ptypes[i])
        else:
            prompt4 += "{}th similar entity: {}, and its category: {}\n".format(str(i+1), pnames[i], ptypes[i])
    prompt5 = "Is the word '{}' in the input sentence a '{}' entity? Please answer with yes or no.".format(ename, etype)
    # prompt5 = "Now direct output the category of entity ' {} ', Your output must be one of 'Chemicals', 'Species', 'Gene/Protein', 'other'. ".format(ename)

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',        
        messages=[
            {"role": "system", "content": prompt1},
            {"role": "user", "content": prompt2 + "\n" + prompt4 + "\n" + prompt5},
        ]
    )
    # print(prompt2 + "\n" + prompt4 + "\n" + prompt5)
    # ipdb.set_trace()
    resText = response.choices[0].message.content
    return resText



cnt = 0
for key in tqdm(data.keys()):
    if 'gpt_type' in data[key]: continue
    ele = data[key]
    sent = " ".join(ele['words'])
    preds = []
    preds_type = []

    for etype in ava_etype:
        spans = parse_span(ele['gpt_result'][etype])

        for span in spans:

            ename = " ".join(ele['words'][span[0]:span[1] + 1])
            pnames = []
            ptypes = []
            for e_ in ele['preds_bem']:
                if e_['name'] == ename:
                    pnames = e_['pred_names']
                    ptypes = e_['preds']
            if len(pnames) == 0: continue

            if 'cot-2' in method_name:
                res = gptapi_cot2(sent, ename, etype)
            if 'cot-4' in method_name:
                res = gptapi_cot4(sent, ename, etype, ptypes, pnames)
            if 'no' not in res.lower():
                preds.append(span)
                preds_type.append(etype)

    data[key]['preds'] = preds
    data[key]['golds'] = data[key]['spans'] 
    data[key]['gold_types'] = data[key]['spans_type'] 
    data[key]['gpt_type'] = preds_type

    cnt += 1
    if cnt % save_interval == 0:
        with open(out_file_name, 'w') as of:
            json.dump(data, of)
    # ipdb.set_trace()

with open(out_file_name, 'w') as of:
    json.dump(data, of)
