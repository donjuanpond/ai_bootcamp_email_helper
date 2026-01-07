from openai import OpenAI
from dotenv import load_dotenv
import os
import yaml
import json
from generate import GenerateEmail

def load_jsonl(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            out[data['id']] = data
    return out

emails_shorten = load_jsonl("datasets/shorten.jsonl")

generator = GenerateEmail("gpt-4o-mini")

avg_c = 0
avg_f = 0
n = 0
for email_id in emails_shorten:
    email = emails_shorten[email_id]
    new_text = generator.generate("shorten", email)
    completeness_judge_out = json.loads(generator.generate_judge("completeness", "shorten", email, new_text))
    faithfulness_judge_out = json.loads(generator.generate_judge("faithfulness", "shorten", email, new_text))
    print(completeness_judge_out)
    print(faithfulness_judge_out)


    avg_c += completeness_judge_out['rating']
    avg_f += faithfulness_judge_out['rating']
    n += 1

avg_c /= n
avg_f /= n
print()
print(avg_c)
print(avg_f)




