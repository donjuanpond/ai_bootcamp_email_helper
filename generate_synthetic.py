from openai import OpenAI
from dotenv import load_dotenv
import os
import yaml

load_dotenv()

with open("synthetic_prompts.yaml", "r") as f:
    prompts = yaml.safe_load(f)

class GenerateEmail():    
    def __init__(self, model: str):
        # initialize client once
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.deployment_name = model

    def _call_api(self, messages, is_judge=False):
        if is_judge:
            response = self.client.chat.completions.create(
                model = "gpt-4.1",
                messages=messages,
                seed=42
            )
        else:
            response = self.client.chat.completions.create(
                model = self.deployment_name,
                messages=messages,
                seed=42
            )
        return response.choices[0].message.content
    
    def get_prompt(self, prompt_name, prompt_type='user', **kwargs):
        template = prompts[prompt_name][prompt_type]
        return template.format(**kwargs)
    
    
    def send_prompt(self, user_prompt: str, system_msg="You are a helpful assistant.", is_judge=False):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
        return self._call_api(messages, is_judge)
    
    def generate(self, action: str, prompt_params: dict) -> list:
        system_prompt = self.get_prompt(action, prompt_type='system', **prompt_params)
        user_prompt = self.get_prompt(action, prompt_type='user', **prompt_params)
        print("system prompt:", system_prompt)
        print("user prompt:", user_prompt)
        model_response = self.send_prompt(user_prompt, system_prompt)
        return model_response
    
generator = GenerateEmail("gpt-4o-mini")
#out = generator.generate("shorten", {"topic":"potential business expenses", "persona":"Cosmo Kramer", "tone":"stressed", "length":"200 words"})
#out2 = generator.generate("lengthen", {"topic":"flash flooding", "persona":"Zane the Gamer", "tone":"wise"})

prompt_list = [{"topic":"flash flooding", "persona":"Master Wu", "tone":"wise"},
               {"topic":"potential business expenses", "persona":"Cosmo Kramer", "tone":"stressed"},
               {"topic":"a bad haircut", "persona":"Jimmy McNulty", "tone":"agitated"},
               {"topic":"angry pigs", "persona":"Boba Fett", "tone":"terrified"},
               {"topic":"kids these days", "persona":"Hank Hill", "tone":"proud"},
               ]

with open("datasets/shorten_synthetic.jsonl", "w", encoding="utf-8") as f:
    id_counter = 1
    for prompt in prompt_list:
        id_prompt = prompt | {"id":str(id_counter)}
        out = generator.generate("shorten", id_prompt)
        f.write(out)
        f.write("\n")
        id_counter += 1

with open("datasets/lengthen_synthetic.jsonl", "w", encoding="utf-8") as f:
    id_counter = 1
    for prompt in prompt_list:
        id_prompt = prompt | {"id":str(id_counter)}
        out = generator.generate("lengthen", id_prompt)
        f.write(out)
        f.write("\n")
        id_counter += 1

with open("datasets/tone_synthetic.jsonl", "w", encoding="utf-8") as f:
    id_counter = 1
    for prompt in prompt_list:
        id_prompt = prompt | {"id":str(id_counter)}
        out = generator.generate("friendly", id_prompt)
        f.write(out)
        f.write("\n")
        id_counter += 1

        id_prompt = prompt | {"id":str(id_counter)}
        out2 = generator.generate("sympathetic", id_prompt)
        f.write(out2)
        f.write("\n")
        id_counter += 1

        id_prompt = prompt | {"id":str(id_counter)}
        out2 = generator.generate("professional", id_prompt)
        f.write(out2)
        f.write("\n")
        id_counter += 1
       
