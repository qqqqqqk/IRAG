import json

datas = []

with open("./queries.txt", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    datas.append(
        {
            "question_id": i,
            "question_text": line.strip()
        }
    )

with open("./test_subsampled.jsonl", "w") as f:
    for data in datas:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
