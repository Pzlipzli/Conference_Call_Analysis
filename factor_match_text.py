import pandas as pd
import numpy as np
import json
import json5
from datetime import datetime
from openai import OpenAI
import requests
import os

url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
headers = { 
    "Content-Type": "application/json", 
    "Authorization": "Bearer 2813690ea2744a8384ff460d56722723b6506efc88b4422ca4925cf87cea7199" #Please change your KEY. If your key is XXX, the Authorization is "Authorization": "Bearer XXX"
}
client = OpenAI(api_key="sk-6cc269bac20345578bf1ed8811acad8e", base_url="https://api.deepseek.com")
# system_msg = """
# You are an expert financial analyst. Your task is to extract short, generalizable statements that reflect characteristics or behaviors of the company that may impact its future development. The information source is a conference call transcript in pdf format.
# Avoid repeating specific product names, company names, or one-time events. Instead, summarize impact of macroeconomic factors and industry dynamics, corporate financials, and company characteristics (Management's proactive, Employee's freedom, innoactiveness, competitive aggressiveness and attitude to risk-taking) in a way that is applicable to many companies.
# Each statement should be a short sentence describing one characteristic (e.g., “The company has high sensitivity to GDP growth.” or “Management emphasizes the target of cost efficiency.”).
# You should also split the response into several parts -- the first is from the opening statement of management, the second is from the first Q&A part,  the third is from the second Q&A part, and so on. Each Q&A part is following a signal like "The first question is from ..." or "The next question is from ...".
# Output a list of features in json format. Here is an example: \{"opening": ["First feature", "Second feature"], "q_and_a_1": ["Third feature", "Fourth feature"], "q_and_a_2": ["Fifth feature", "Sixth feature"]...\}.\n
# """

# with open(f"./factors.txt", 'r', encoding='utf-8') as file:
#     feature_set = file.read()

json_file = "./multi_factor_pool_0_2.json"
with open(json_file, 'r', encoding='utf-8') as f:
    data_dict = json.load(f)

    textual_set = data_dict["textual"]


system_msg1 = """
You are an assistant designed to analyze earnings call transcripts.

You will be given:
A txt file of a company's earnings conference call transcript. It is well divided into two parts: `Presentation` and `Questions And Answers`. In the `Presentation` part, the management of the company will give a speech about the performance in the last quarter or year and their future vision. In the `Questions And Answers` part, the company's management will answer questions from analysts and investors.
There are usually several questions divided by the Operator's order. For example, "The first question is from ...", "The next question is from ...", etc.
Some transcript may have a closing words part. If there is a closing part, you should seperate it from the Q&A part for analysis.

---

### Your task:
- Go through the transcript, determine the precise text content of `Presentation` part, the first question fron `Questions And Answers` part, the second question from `Questions And Answers` part, ...(Closing words part if applicable).
- Return in JSON format of the precise text content of each part. Don't give any explanations.

Example:
{
  "Pres": "The company's performance is good...",
  "Q&A1": "What is the company's strategy for the next year?...",
  "Q&A2": "How is the company's competition?...",
  ...
  "Cls": "I wanna wrap up today's conference call..."
}
"""


system_msg2 = f"""
You are an assistant designed to analyze earnings call transcripts.

You will be given:
1. A txt file of a company's earnings conference call transcript. It is well divided into two parts: `Presentation` and `Questions And Answers`. In the `Presentation` part, the management of the company will give a speech about the performance in the last quarter or year and their future vision. In the `Questions And Answers` part, the company's management will answer questions from analysts and investors.
There are usually several questions divided by the Operator's order. For example, "The first question is from ...", "The next question is from ...", etc.
Some transcript may have a closing words part. If there is a closing part, you should put it into the final Q&A part for analysis.
2. A list of textual features (called the "feature pool"). Each feature is a generalized, factual statement about company performance or strategy, in the form: "Something is ..."

---

### Your task:
- Go through the transcript, and compare the presentation content or answers from the management with the feature pool.
- Select all matching features from the pool that the response implies, supports, or reflects.
- Only use **existing** features from the pool. **Do not invent new ones**. Make sure all words in the features are exactly the same as in the pool.
- Give your matching results in lists: `Pres` (Presentation), `Q&A1` (The first question and its corresponding answer from Questions and Answers part), `Q&A2`, etc.
- If no features match, assign an empty list.

---

### Output format:
Return the result as a JSON object. Each key should be `Pres`, `Q&A1`, `Q&A2`, etc., and the value should be a list of matching features from the pool. Don't put features from other parts in the wrong place. Don't give any explanations.

Example:
{{
  "Pres": ["Profitability is improving.", "Demand is stable.", ...],
  "Q&A1": ["Liquidity is tight.", ...],
  "Q&A2": ["Market share is growing.", ...],
  ...
}}

The feature set:
{textual_set}
"""


system_message1 = {"role": "system", "content": system_msg1}
system_message2 = {"role": "system", "content": system_msg2}


def bloomberg_llm_summary(folder_path):
    items = os.listdir(folder_path)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]

    for subdir in subdirectories:
        subdir_path = os.path.join(folder_path, subdir)

        # if not subdir in ['HD', 'IBM', 'INTU', 'JNJ', 'LIN', 'MCD', 'MRK', 'MS', 'NOW', 'PG', 'PM']:
        #     continue
        # if not subdir in ['PEP']:
            # continue

        for file in os.listdir(subdir_path):
            # if file == 'AAPL_21_01_28_0600.txt':
            if file.endswith('.txt') and not file.endswith('_timestamp.txt') and not file.endswith('_text.txt'):
                file_path = os.path.join(subdir_path, file)
                ec_name = file.split('.')[0]
                # ec_file = f"/Users/lipeizhe/fintech/bloomberg_factor_match/{subdir}/{ec_name}_text.json"
                ec_file2 = f"/Users/lipeizhe/fintech/bloomberg_factor_match/{subdir}/{ec_name}.json"

                if os.path.exists(ec_file2):
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    prompt = f"Conference Call Transcript: \n{content}\n\nYou MUST reply LEGAL JSON format. Do not use ' instead of \". Do not miss any \" or cause any uncomplete brackets or braces."
                    messages = [system_message2, {"role": "user", "content": prompt}]
                    response = client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=messages,
                        temperature=0.2
                    )
                    
                # 将 prompt 添加到结果字典中
                result = response.choices[0].message.content
                result = result.replace("```json", '').replace("```", '')
                os.makedirs(f"/Users/lipeizhe/fintech/bloomberg_factor_match/{subdir}", exist_ok=True)

                with open(ec_file2, 'w', encoding='utf-8') as f:
                    try:
                        result_data = json.loads(result)
                        json.dump(result_data, f, ensure_ascii=False, indent=4)
                    except:
                        json.dump(result, f, ensure_ascii=False, indent=4)
                        print(f"{ec_name} json result is saved with error.")

                # prompt2 = f"Well Divided Conference Call Transcript: \n{result_data}\n\nYou must reply legal JSON format."
                # messages2 = [system_message2, {"role": "user", "content": prompt2}]
                # response2 = client.chat.completions.create(
                #     model="deepseek-chat",
                #     messages=messages2,
                #     temperature=0.1
                # )
                # result2 = response2.choices[0].message.content
                # result2 = result2.replace("```json\n", '').replace("```\n", '')

                # with open(ec_file2, 'w', encoding='utf-8') as f:
                #     result_data2 = json5.loads(result2)
                #     json.dump(result_data2, f, ensure_ascii=False, indent=4)

        print(f"{subdir} json result is saved.")


if __name__ == '__main__':
    bloomberg_llm_summary('./bloomberg_dataset/')