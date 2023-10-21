import openai
import os
import json
from tqdm import tqdm
import ipdb
from utils import *

openai.api_key = ""


dataset_name = 'genia'

if dataset_name == 'craft': ava_etype = ['Species', 'Gene/Protein', 'Chemicals']
if dataset_name == 'bc5cdr': ava_etype = ['Disease', 'Chemical']
if dataset_name == 'bionlp11': ava_etype = ['Gene/Protein', 'Chemical']
if dataset_name == 'genia': ava_etype = ['protein', 'RNA', 'DNA', 'cell_line', 'cell_type']
if dataset_name == 'bionlp13': ava_etype = ['Gene/Protein', 'Chemical', 'Disease']

save_interval = 20
resume = True

file_path = f'./{dataset_name}/test.json'


nums = file_path.split("/")[-1].replace(".json", "").replace('test', '')
if nums == '': nums = 'all'
else: nums = 'N' + nums
out_file_name = "./##/ALL_GPTNER_".replace("##", dataset_name) + nums + ".json"
print(out_file_name)

# ipdb.set_trace()

if resume == False: data = json.load(open(file_path))
else: data = json.load(open(out_file_name))


## modify this function:
def gptapi(sent, entity):
    # return sent
    prompt1 = """You are a Biomedical expert. Given a sentence, you task is to label all '{}' entities in the given sentences. 
At the beginning, you will see an input-ouput example. Please note your output format must be consistent with the example! """.format(entity)

    if dataset_name == 'bionlp13':
        if entity == 'Gene/Protein':
            prompt21 = "Input: However , the 5 - year survival rate of patients with high MMP - 9 protein expression were lower than those with low expression"
            prompt22 = "Output: However , the 5 - year survival rate of patients with high @@MMP - 9## protein expression were lower than those with low expression"
        if entity == 'Chemical':
            prompt21 = "Input: The malate - citrate cycle was studied during aerobic glycolysis and glutaminolysis in a strain of Ehrlich ascites tumor cells which showed a very low malate - aspartate shuttle system activity ."
            prompt22 = "Output: The @@malate## - @@citrate## cycle was studied during aerobic glycolysis and glutaminolysis in a strain of Ehrlich ascites tumor cells which showed a very low @@malate## - @@aspartate## shuttle system activity ."
        if entity == 'Disease':
            prompt21 = "Input: Expression of CD154 on renal cell carcinomas and effect on cell proliferation , motility and platelet - activating factor synthesis ."
            prompt22 = "Output: Expression of CD154 on @@renal cell carcinomas## and effect on cell proliferation , motility and platelet - activating factor synthesis ."

    if dataset_name == 'craft':
        if entity == 'Species':
            prompt21 = "Input: Thereafter , the testes of the Dmrt7 mutant mice ceased to grow and the weight difference was significant ."
            prompt22 = "Output: Thereafter , the testes of the Dmrt7 mutant @@mice## ceased to grow and the weight difference was significant ."
            prompt31 = "Input: Primary mouse keratinocytes were treated with Wnt- and/or noggin-conditioned medium ( + ) or the corresponding control medium ( – ) ."
            prompt32 = "Output: Primary @@mouse## keratinocytes were treated with Wnt- and/or noggin-conditioned medium ( + ) or the corresponding control medium ( – ) ."

        if entity == 'Gene/Protein':
            prompt21 = "Input: Thereafter , the testes of the Dmrt7 mutant mice ceased to grow and the weight difference was significant ."
            prompt22 = "Output: Thereafter , the testes of the @@Dmrt7## mutant mice ceased to grow and the weight difference was significant ."
            prompt31 = "Input: Moreover , since inhibition of GSK-3β results in Snail upregulation and E-cadherin downregulation , Snail and GSK-3β may function at a crossroads in controlling hair bud development ."
            prompt32 = "Output: Moreover , since inhibition of @@GSK-3β## results in @@Snail## upregulation and @@E-cadherin## downregulation , @@Snail## and @@GSK-3β## may function at a crossroads in controlling hair bud development ."

        if entity == 'Chemicals':
            prompt21 = "Input: A common feature of these proteins is involvement with heterochromatin and/or transcriptional repression ."
            prompt22 = "Output: A common feature of these @@proteins## is involvement with heterochromatin and/or transcriptional repression ."
            prompt31 = "Input: Mice were mated during the dark period of the controlled light cycle ; presence of vaginal plugs was designated as day 0 hour 0. Females were euthanized by CO2 , and embryos were collected in Hanks ' balanced salt solution on ice ."
            prompt32 = "Output: Mice were mated during the dark period of the controlled light cycle ; presence of vaginal plugs was designated as day 0 hour 0. Females were euthanized by @@CO2## , and embryos were collected in Hanks ' balanced @@salt## solution on ice ."

    if dataset_name == 'bc5cdr':
        if entity == 'Disease':
            prompt21 = "Input: Our data suggested that the ginsenoside Re , but not Rg1 or Rb1 , may contribute toward reversal of OIH ."
            prompt22 = "Output: Our data suggested that the ginsenoside Re , but not Rg1 or Rb1 , may contribute toward reversal of @@OIH## ."
            prompt31 = "Input: However , the role of vasopressin remains to be determined in human essential hypertension ."
            prompt32 = "Output: However , the role of vasopressin remains to be determined in human essential @@hypertension## ."

        if entity == 'Chemical':
            prompt21 = "Input: Our data suggested that the ginsenoside Re , but not Rg1 or Rb1 , may contribute toward reversal of OIH ."
            prompt22 = "Output: Our data suggested that the @@ginsenoside Re## , but not @@Rg1## or @@Rb1## , may contribute toward reversal of OIH ."
            prompt31 = "Input: However , the role of vasopressin remains to be determined in human essential hypertension ."
            prompt32 = "Output: However , the role of @@vasopressin## remains to be determined in human essential hypertension ."

    if dataset_name == 'bionlp11':
        if entity == 'Gene/Protein':
            prompt21 = "Input: It is also possible that bac gene encoding for beta - antigen is not only target for Sak188 / Sak189 TCS ."
            prompt22 = "Output: It is also possible that bac gene encoding for @@beta - antigen## is not only target for @@Sak188## / @@Sak189## TCS ."
            prompt31 = "Input: We have shown that AprA protects P . entomophila against the Imd - regulated immune response ."
            prompt32 = "Output: We have shown that @@AprA## protects P . entomophila against the @@Imd## - regulated immune response ."

        if entity == 'Chemical':
            prompt21 = "Input: In E . coli , the mal genes are essential for the transport and utilization of maltose and maltodextrin [ 56 ] ."
            prompt22 = "Output: In E . coli , the mal genes are essential for the transport and utilization of @@maltose## and @@maltodextrin## [ 56 ] ."
            prompt31 = "Input: Similarly , ATP - binding and Mg2 + - binding sites located at C - terminal end of Sak188 , which are essential for phosphorylation , were lost in Sak188 protein due to the mutagenesis ."
            prompt32 = "Output: Similarly , @@ATP## - binding and @@Mg2 +## - binding sites located at C - terminal end of Sak188 , which are essential for phosphorylation , were lost in Sak188 protein due to the mutagenesis ."

    if dataset_name == 'genia':
        if entity == 'protein': 
            prompt21 = "Input: Endogenous cortisol has no effect on GCR level estimated by 3H - triamcinolone acetonide ."
            prompt22 = "Output: Endogenous cortisol has no effect on @@GCR## level estimated by 3H - triamcinolone acetonide ."
            prompt31 = "Input: Two glucocorticoid binding sites on the human glucocorticoid receptor ."
            prompt32 = "Output: Two @@glucocorticoid binding sites## on the @@human glucocorticoid receptor## ."
        if entity == 'DNA': 
            prompt21 = "Input: A new member of the leucine zipper class of proteins that binds to the HLA DR alpha promoter ."
            prompt22 = "Output: A new member of the leucine zipper class of proteins that binds to the @@HLA DR alpha promoter## ."
            prompt31 = "Input: Several mutants derived from transformed human B cell lines are defective in expressing major histocompatibility complex ( MHC ) class II genes ."
            prompt32 = "Output: Several mutants derived from transformed human B cell lines are defective in expressing @@major histocompatibility complex ( MHC ) class II genes## ."
        if entity == 'RNA': 
            prompt21 = "Input: The half - life of c - jun RNA as determined by treating HL - 60 cells with TPA and actinomycin D was 30 min ."
            prompt22 = "Output: The half - life of @@c - jun RNA## as determined by treating HL - 60 cells with TPA and actinomycin D was 30 min ."
            prompt31 = "Input: Rhom - 2 mRNA is expressed in early mouse development in central nervous system , lung , kidney , liver , and spleen but only very low levels occur in thymus ."
            prompt32 = "Output: @@Rhom - 2 mRNA## is expressed in early mouse development in central nervous system , lung , kidney , liver , and spleen but only very low levels occur in thymus ."
        if entity == 'cell_line': 
            prompt21 = "Input: In summary , platelet - activating factor is shown here to have a direct and profound effect on a pure B cell line ."
            prompt22 = "Output: In summary , platelet - activating factor is shown here to have a direct and profound effect on a @@pure B cell line## ."
            prompt31 = "Input: Tandem copies of this 67 - bp MnlI - AluI fragment , when fused to the chloramphenicol acetyltransferase gene driven by the conalbumin promoter , stimulated transcription in B cells but not in Jurkat T cells or HeLa cells ."
            prompt32 = "Output: Tandem copies of this 67 - bp MnlI - AluI fragment , when fused to the chloramphenicol acetyltransferase gene driven by the conalbumin promoter , stimulated transcription in B cells but not in @@Jurkat T cells## or @@HeLa cells## ."
        if entity == 'cell_type': 
            prompt21 = "Input: HB24 is likely to have an important role in lymphocytes as well as in certain developing tissues ."
            prompt22 = "Output: HB24 is likely to have an important role in @@lymphocytes## as well as in certain developing tissues ."
            prompt31 = "Input: We further demonstrate the impact of platelet - activating factor binding to B cells by measuring platelet - activating factor induced arachidonic acid release and 5 - hydroxyeicosatetraenoic acid production ."
            prompt32 = "Output: We further demonstrate the impact of platelet - activating factor binding to @@B cells## by measuring platelet - activating factor induced arachidonic acid release and 5 - hydroxyeicosatetraenoic acid production ."

    input_text1 = prompt21 + '\n' + prompt22
    input_text2 = "Input: {}\nOutput: ".format(sent)

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": prompt1},
            {"role": "user", "content": input_text1 + '\n\n' + input_text2}
        ]
    )
    # ipdb.set_trace()
    resText = response.choices[0].message.content
    return resText



cnt = 0
for key in tqdm(data.keys()):

    ele = data[key]
    sent = ' '.join(ele['words'])

    if 'gpt_result' in data[key]:
        if len(data[key]['gpt_result']) == len(ava_etype):
            continue

    data[key]['gpt_result'] = {}

    for etype in ava_etype:
        response = gptapi(sent, etype)
        data[key]['gpt_result'][etype] = response
    
    cnt += 1

    if cnt % save_interval == 0:
        with open(out_file_name, 'w') as of:
            json.dump(data, of)

with open(out_file_name, 'w') as of:
    json.dump(data, of)

