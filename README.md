# SKIPNER
In this project we provied both source data and infered results from ChatGPT-3.5.

The file names in BioNER with their corresponding method in paper : 



- ALL_FPTNER_all.json -> GPTNER
- cot-1 -> SKIPNER-w/o KG
- cot-2 -> GPTNER-SV
- cot-3 -> SKIPNER
- vote -> SKIPNER- TypeByVote
## Step1: Span Extraction

```
python s1_allByGPT.py (GPTNER)
python s1_spanByAll.py (data format convert)
```

## Step2: Type Prediction
```
python s3_typeByGPT.py (SKIPNER variants)
python s2_selfVerify.py (GPTNER + Self Verity)
```


## Results Evaluation


```
python s4_evaluate_dir.py (for GPTNER)
python s4_evaluate_cot.py (for other method)
```




