from datasets import load_dataset, load_metric
from bert_score import score

dataset = load_dataset('json', data_files='output-alpaca.json')
predictions = [example["output"] for example in dataset["train"]]
references = [example["existing"] for example in dataset["train"]]

rouge = load_metric("rouge")
scores = rouge.compute(predictions=predictions, references=references)
print(scores)

# Bert Score
P, R, F1 = score(predictions, references, lang='en', model_type='bert-base-multilingual-cased', rescale_with_baseline=True)

P_average = P.mean()
R_average = R.mean()
F1_average = F1.mean()

print(f'Precision: {P_average}')
print(f'Recall: {R_average}')
print(f'F1: {F1_average}')

def extract_scores(rouge_scores, P_average, R_average, F1_average):
    R1 = rouge_scores['rouge1'].mid.fmeasure * 100
    R2 = rouge_scores['rouge2'].mid.fmeasure * 100
    RL = rouge_scores['rougeL'].mid.fmeasure * 100
    BS = F1_average * 100
    return R1, R2, RL, BS

R1, R2, RL, BS = extract_scores(scores, P_average, R_average, F1_average)

print(f"R1: {R1:.2f}, R2: {R2:.2f}, RL: {RL:.2f}, BS: {BS:.2f}")
