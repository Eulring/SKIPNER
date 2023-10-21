import openai
import os
import json
from tqdm import tqdm
import ipdb
from utils import *


file_path = './bionlp11/S1GPT_S2dict-uni12_all.json'
openai.api_key = ""


method_name = 'cot-3'
# method_name = 'cot-2'
# method_name = 'cot-1'
# method_name = 'vote'
k = 10
resume = False

if 'craft' in file_path: ava_etype = ['Species', 'Gene/Protein', 'Chemicals']
if 'bc5cdr' in file_path: ava_etype = ['Disease', 'Chemical']
if 'bionlp11' in file_path: ava_etype = ['Gene/Protein', 'Chemical']
if 'genia' in file_path: ava_etype = ['protein', 'RNA', 'DNA', 'cell_line', 'cell_type']
if 'bionlp13' in file_path: ava_etype = ['Gene/Protein', 'Chemical', 'Disease']


# output file names
names = file_path.split("_")
names[-1] = 'S3'+method_name+'-k'+str(k)+'_'+names[-1]
out_file_name = "_".join(names)
print(out_file_name)


save_interval = 50
if resume: data = json.load(open(out_file_name))
else: data = json.load(open(file_path))

## modify this function:
def gptapi_cot1(sent, ename):
    entities = ['\'{}\''.format(ele) for ele in ava_etype]
    prompt1 = """You are a biomedical expert. Given a sentence, you need to output the category of '{}' in the sentence. 
Your output must be one of [{}, 'Others'].""".format(ename, ', '.join(entities))
    prompt2 = "Input sentence: '{}'".format(sent)
    prompt3 = "Now direct output the category of entity '{}': ".format(ename)
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

def gptapi_cot2(sent, ename, etype):
    entities = ['\'{}\''.format(ele) for ele in ava_etype]
    prompt1 = """You are a biomedical expert. The task is to verify whether the word is {} entity extracted from the given sentence.""".format(etype)
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


def gptapi_cot3(sent, ename, ptypes, pnames):
    entities = ['\'{}\''.format(ele) for ele in ava_etype]
    assert ename in sent
    sent = sent.replace(ename, "##"+ename+"@@")
    prompt1 = """You are a biomedical expert. Given a sentence, you need to output the category of {} in the sentence. 
Your output must be one of [{}, 'Others'].""".format(ename, ', '.join(entities))
    prompt2 = "Input sentence: '{}'".format(sent)
    prompt4 = "Entity names similar to '{}' and their corresponding categories are arranged in order based on their similarity as follows: \n".format(ename)
    # 和A相思的实体根据相似度排序展示
    for i in range(k):
        if i == 0: 
            prompt4 += "Most similar entity: {}, and its category: {}\n".format(pnames[i], ptypes[i])
        elif i == 1:
            prompt4 += "2nd similar entity: {}, and its category: {}\n".format(pnames[i], ptypes[i])
        elif i == 2:
            prompt4 += "3rd similar entity: {}, and its category: {}\n".format(pnames[i], ptypes[i])
        else:
            prompt4 += "{}th similar entity: {}, and its category: {}\n".format(str(i+1), pnames[i], ptypes[i])
            # (  "Rank " + str(i+1)+'. < #Name: ' +  pnames[i] + '  #Category: ' + ptypes[i] + ' > \n')
    prompt5 = "If category is not one of [{}], you need to output 'other'.".format(', '.join(entities))
    prompt6 = "Select a category from [{}, 'Others'] to determine the category of '{}': ".format(', '.join(entities), ename)

    # print(prompt2 + '\n' + prompt4 + '\n' + prompt5 + '\n' + prompt6)
    
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',        
        messages=[
            {"role": "system", "content": prompt1},
            {"role": "user", "content": prompt2 + '\n' + prompt4 + '\n' + prompt5 + '\n' + prompt6},
        ]
    )
    resText = response.choices[0].message.content
    return resText


## modify this function:
def askChatGpt(ele):
    sent = ' '.join(ele['words'])
    preds = ele['preds']
    preds_info = ele['preds_bem']
    response = []
    if 'prepred_types' in ele:
        pred_types = ele['prepred_types']

    for i, (pred, info) in enumerate(zip(preds, preds_info)):
        if method_name == 'greedy':
            res = info['preds'][0]
        if method_name == 'vote':
            kpred = info['preds'][0:k]
            res = max(set(kpred), key=kpred.count)
        if method_name == 'cot-1':
            res = gptapi_cot1(sent, info['name'])
        if method_name == 'cot-2':
            res = gptapi_cot1(sent, info['name'])
        if method_name == 'cot-3':
            res = gptapi_cot3(sent, info['name'], info['preds'], info['pred_names'])

        response.append(res)

    return response

cnt = 0
for key in tqdm(data.keys()):
    if 'gpt_type' in data[key]: continue
    cnt += 1
    response = askChatGpt(data[key])
    data[key]['gpt_type'] = response
    if cnt % save_interval == 0:
        with open(out_file_name, 'w') as of:
            json.dump(data, of)

with open(out_file_name, 'w') as of:
    json.dump(data, of)
