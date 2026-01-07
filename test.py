import json
emails_lengthen = []
with open("datasets/lengthen.jsonl", 'r', encoding='utf-8') as lengthen:
    for line in lengthen:
        data = json.loads(line)
        emails_lengthen.append(data)

print(emails_lengthen[7])
