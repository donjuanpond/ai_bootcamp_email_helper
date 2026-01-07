from openai import OpenAI
from dotenv import load_dotenv
import os
import yaml

load_dotenv()

with open("prompts.yaml", "r") as f:
    prompts = yaml.safe_load(f)
with open("judge_prompts.yaml","r") as g:
    judge_prompts = yaml.safe_load(g)

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
    
    def generate(self, action: str, selected_email: dict) -> list:
        system_prompt = self.get_prompt(action, prompt_type='system', **selected_email)
        user_prompt = self.get_prompt(action, prompt_type='user', **selected_email)
        print("system prompt:", system_prompt)
        print("user prompt:", user_prompt)
        model_response = self.send_prompt(user_prompt, system_prompt)
        return model_response

    def get_judge_prompt(self, prompt_name, prompt_type='user', **kwargs):
        template = judge_prompts[prompt_name][prompt_type]
        return template.format(**kwargs)

    def generate_judge(self, judge_action: str, task: str, selected_email: dict, orig_model_response: str):
        prompt_kwargs = {"task":task,
                         "selected_text":selected_email,
                         "model_response":orig_model_response}
        system_prompt = self.get_judge_prompt(judge_action, prompt_type='system',**prompt_kwargs)
        user_prompt = self.get_judge_prompt(judge_action, prompt_type='user',**prompt_kwargs)

        model_response = self.send_prompt(user_prompt, system_prompt, is_judge=True)
        return model_response
""" 
# Testing out a sample call to the API
generator = GenerateEmail("gpt-4o-mini")
print(generator.generate(
    action="friendly",
    selected_email={"id": 52, "sender": "zoe.lin@foresttrailpublishing.com", "subject": "Illustration Review for Upcoming Release", "content": "Hello team, we\u2019ve received the first batch of illustrations for the upcoming release, and I\u2019ve compiled them into a review packet. Please take time to examine each piece carefully\u2014checking for consistency in style, character accuracy, and alignment with the story\u2019s tone. If you notice any details that should be corrected or adjusted, make notes directly in the packet. I\u2019d like to send consolidated feedback to the illustrator by Friday afternoon. Thanks for lending your keen eye to this review."}
))
"""