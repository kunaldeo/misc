import json
from tqdm import tqdm

with open('indosum/dev.01.jsonl', 'r') as file:
    print("Reading lines...")
    lines = [line for line in tqdm(file, desc="Reading lines")]

data = []
id_counter = 1

print("Processing lines...")
for line in tqdm(lines, desc="Processing"):
    obj = json.loads(line)

    paragraphs = '\n'.join([' '.join([' '.join(sentence) for sentence in paragraph]) for paragraph in obj['paragraphs']])

    summary = ' '.join([' '.join(sentence) for sentence in obj['summary']])

    transformed_data = {
        "id": f"{id_counter} [Type: Instruction] [Lang: in] [Dataset: indo-sum]",
        "conversations": [
            {
                "from": "human",
                "value": f"Ringkaslah teks berikut\n{paragraphs}"
            },
            {
                "from": "gpt",
                "value": summary
            }
        ]
    }

    data.append(transformed_data)
    id_counter += 1

with open('indosum/dev.01.json', 'w') as outfile:
    json.dump(data, outfile, indent=2)
