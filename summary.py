from openai import OpenAI
from oai_keys import API_KEY, BASE_URL, MODEL_NAME

import os
import yaml
from tqdm import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt

from MemeAgent.prompt_dict import *


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_oai(mess):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=mess,
        temperature=0.0,
    )

    completion_tokens = response.usage.completion_tokens
    prompt_tokens = response.usage.prompt_tokens

    return response.choices[0].message.content, completion_tokens, prompt_tokens

def prepare_message(prompt,yaml_dict):
    message = []

    QA_history = ""

    message.append({"role": "system", "content": prompt["SummarySystem"]})
    message.append({"role": "user", "content": yaml_dict["init_caption"] + QA_history + prompt["SummaryUser"]})

    return message

def get_summary(prompt, yaml_dict):
    message = prepare_message(prompt,yaml_dict)
    response, completion_tokens, prompt_tokens =call_oai(message)
    return [response, completion_tokens, prompt_tokens]

def read_yaml(path):
    result_list = []
    for file in os.listdir(path):
        if file.endswith('.yaml') and file != '0000prompt.yaml':
            with open(os.path.join(path, file), 'r') as f:
                result_list.append(yaml.load(f, Loader=yaml.FullLoader))
    return result_list

def read_yaml_file_name(path):
    result_list = []
    for file in sorted(os.listdir(path)):
        if file.endswith('.yaml') and file != '0000prompt.yaml':
            result_list.append(file)
    return result_list

def save_prompt(prompt, path):
    # save prompt to yaml
    yaml_system = {}
    yaml_system["agent system"] = prompt['system']
    yaml_system["Internet User"] = prompt['Internet User']
    yaml_system["Internet Supervisor"] = prompt['Internet Supervisor']
    yaml_system["SummarySystem"] = prompt['SummarySystem']
    yaml_system["SummaryUser"] = prompt['SummaryUser']
    yaml_system_path = os.path.join(path, "0000prompt.yaml")
    with open(yaml_system_path,'w') as file:
        yaml.dump(yaml_system,file,sort_keys=False)


import concurrent.futures
from tqdm import tqdm

if __name__ == "__main__":
    # define path to run

    #  "harmfulness" "hatefulness" "misogyny" "offensiveness" "sarcasm"
    task = 'sarcasm'
    prompt = prompt[task]

    yaml_root_path = os.path.join('./history', task)
    update_save_path = os.path.join('./history_new', task)

    
    # check if the directory exists, if no make it
    if not os.path.exists(update_save_path):
        os.makedirs(update_save_path)

    # save prompt to yaml
    save_prompt(prompt, update_save_path)

    # yaml_list = read_yaml(yaml_root_path)

    yaml_list_todo = read_yaml_file_name(yaml_root_path)
    yaml_list_done = read_yaml_file_name(update_save_path)

    def process_yaml_file(yaml_file):
        if yaml_file in yaml_list_done:
            return

        with open(os.path.join(yaml_root_path, yaml_file), 'r') as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

        if yaml_dict in yaml_list_done:
            return

        result = get_summary(prompt, yaml_dict)
        yaml_dict['summary']['output'] = result[0]
        yaml_dict['summary']['pred_label'] = 1 if 'yes' in result[0].lower() else 0
        yaml_dict['cost']['completion_tokens'] = result[1]
        yaml_dict['cost']['prompt_tokens'] = result[2]

        with open(os.path.join(update_save_path, yaml_dict['id'] + '.yaml'), 'w') as f:
            yaml.dump(yaml_dict, f, sort_keys=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(process_yaml_file, yaml_file) for yaml_file in yaml_list_todo]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")