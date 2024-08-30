import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import yaml

import logging
import tenacity
from tenacity import retry, stop_after_attempt, wait_random_exponential

from torch.utils.data import DataLoader

from openai import OpenAI

from tool import llava13b_tool
from code4supplement.MemeAgent.load_dataset import *

from sklearn.metrics import accuracy_score, f1_score

from oai_keys import API_KEY, BASE_URL, MODEL_NAME

class Agents():
    def __init__(self,agents,logger,prompt,config) -> None:
        """
        initialize agents, including functions that initiate chats and mllm calling
        agents: list of strings, each string is the name of an agent
        logger: logger object
        prompt: dict, system prompt, user prompt, and summary prompt
        config: dict, configurations
        """
        self.agents = agents
        self.logger = logger
        self.prompt = prompt
        self.config = config

        self.yaml_log = []
        self.history = []
        self.qa_history = []
        self.messages = {}
        self.img_path = ''
        self.meme_text = ''
        self.description = ''
        self.summary = ''


        self.completion_tokens = 0
        self.prompt_tokens = 0

        # openai configurations
        self.api_key = API_KEY
        self.base_url = BASE_URL
        self.model = MODEL_NAME

        # see 'tool/llava_tool.py' for more details
        # write your own code like LLavatool to change the model
        self.mllm = llava13b_tool.LLava13bTool(model_name_or_path=self.config['mllm_path'],
                              device_map="auto")

        
    # reset everything, avoid initializing too many classes
    def reset(self):
        self.messages = {}
        self.yaml_log = []
        self.history = []
        self.qa_history = []
        self.img_path = ''
        self.meme_text = ''
        self.description = ''
        self.summary = ''

        self.completion_tokens = 0
        self.prompt_tokens = 0

    # call mllm, for implementation, see 'tool/llava_tool.py', retry if error
    @tenacity.retry(wait=tenacity.wait_exponential(max=60),
                stop=tenacity.stop_after_attempt(5),
                retry=tenacity.retry_if_exception_type(TypeError),
                reraise=True)
    def call_mllm(self,img_path,prompt):
        response = self.mllm.generate(img_path,prompt, max_new_tokens=200,do_sample=False)
        return response


    # check response, see if have pattern '<question>...</question>' that calls mllm
    def if_call_mllm(self, response):
        pattern = r"<question>\n(.+)\n</question>"
        pattern1 = r"<question>(.+)</question>"
        pattern2 = r'(.*?)<question>'
        result = re.search(pattern, response)
        result1 = re.search(pattern1, response)
        result2 = re.search(pattern2, response, re.DOTALL)
        if result2:
            if result:
                # prefix = result.group(1)
                prefix = result2.group(1).replace('\\n', '').replace('\n', '')
                question = result.group(1)
                answer = self.call_mllm(self.img_path, 'Only answer what is asked. '+question)
                return prefix, question, answer 
            elif result1:
                # prefix = result1.group(1)
                prefix = result2.group(1).replace('\\n', '').replace('\n', '')
                question = result1.group(1)
                answer = self.call_mllm(self.img_path, 'Only answer what is asked. '+question)
                return prefix, question, answer
            else:
                return None
        else:
            return None

    # get meme description
    def meme_description(self) -> str:
        response = self.call_mllm(self.img_path,'Only answer what is asked with no comment. Can you explain this image?')
        return response

    # generate text, not used in current version
    def meme_text_ocr(self):
        response = self.call_mllm(self.img_path, 'Only answer what is asked. What is the text in this meme? Text:')
        return response
    
    # celebrity recognition, not used in current version
    def celebrity_recognition(self):
        response = self.call_mllm(self.img_path, 'Is there any celebrity in this image? If yes, only tell me the names.')
        return response
                    
    # call openai, input messages, get response
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def call_oai(self,mess):
        client = OpenAI(api_key=self.api_key,base_url=self.base_url)
        response = client.chat.completions.create(
        model=self.model,
        messages=mess
        )
        completion_tokens = response.usage.completion_tokens
        prompt_tokens = response.usage.prompt_tokens

        self.completion_tokens += completion_tokens
        self.prompt_tokens += prompt_tokens

        return response.choices[0].message.content
    
    # prepare system prompt and description of the meme
    def initiate_agents(self) -> None:
        # self.logger.info('initiating agents...')

        meme_description = f'The meme\'s description:\n{self.meme_description()}'

        # check if run OCR, not usen in current version 
        if self.config['is_run_ocr']:
            meme_text = f'The meme\'s text:\n{self.meme_text_ocr()}'
        else:
            meme_text = f'The meme\'s text:\n{self.meme_text}'
            
        # check if run celebrity recognition, not used in current version
        if self.config['is_celebrity_check']:
            celebrity = f'Celebrity check:\n{self.celebrity_recognition()}'
            self.description = f'This is the meme\'s information:\n\n<information>\n{meme_description}\n{meme_text}\n{celebrity}+\n</information>'
        else:
            self.description = f'This is the meme\'s information:\n\n<information>\n{meme_description}\n{meme_text}\n</information>'

        self.logger.info(self.description)

        for agent in self.agents:
            self.messages[agent] = [{'role': 'system',
                                     'content': self.prompt["system"]}]

    # prepare message for each round
    def prepare_message(self,agent):

        # if history is empty, it's the first round
        if not self.history:
            message_ = 'It\'s the first round.' +self.prompt[agent]
        # if not, get the history, and concat everything with proper format
        else:
            history_ = 'This is the discussion history:\n\n<history>\n'.join(self.history)
            message_ = history_ + '\n</history>\n\n' + self.prompt[agent]

        return self.messages[agent] + [{'role': 'user', 'content': self.description + message_}]

    # chat for one round, for each agent, get response, call mllm to get answer if needed, and store the history
    # return True means ealy ending, will return chat_for_n_rounds also
    def chat_for_one_round(self) -> bool:
        for agent in self.agents:

            self.logger.info(f'-----Agent:{agent}-----')

            message  = self.prepare_message(agent)

            # call LLM, get response
            response = self.call_oai(message) # discuss -> question
            
            # if 'TERMINATE' in response, terminate the conversation, not used in current version
            if 'TERMINATE' in response:
                self.logger.info('Conversation has been terminated....')
                return True
            
            # if <QUESTION> in response, call mllm to get answer
            result = self.if_call_mllm(response)
            if result is not None:
               prefix, question, answer = result
            else:
                self.logger.info("No pattern found, chat shutting down...")
                return True
            
        
            output = f'Round of {agent}:{prefix}\nQuestion of {agent}:{question}\nAnswer for {agent}:{answer}\n'
            
            # for yaml save log file
            self.yaml_log.append(agent)
            self.yaml_log.append(prefix)
            self.yaml_log.append(question)
            self.yaml_log.append(answer)

            self.history.append(output)

            # save QA history, we only need question and answer for summary
            qa_output = f'### Question:\n{question}\n### Answer:\n{answer}\n\n'
            self.logger.info(f'### Prefix:\n{prefix}\n')
            self.logger.info(qa_output)
            self.qa_history.append(qa_output)

        return False # means no early ending, do not return chat_for_n_rounds
    
    # Loop chat_for_one_round for n times
    def chat_for_n_rounds(self,n):
        for i in range(n):
            if self.chat_for_one_round(): # if True, means early ending (patter not found or TERMINATE), return
                return
    
    # summarize the conversation and get the results
    def summarize_and_get_results(self):
        messages = [{
            "role": "system",
            "content": self.prompt['SummarySystem']
        },
        {
            "role": "user",
            "content": self.description + '\nThis is the QA history:\n\n<history>\n'.join(self.qa_history) + '\n</history>\n\n' + self.prompt['SummaryUser']
        }]
        response = self.call_oai(messages)
        self.summary = response
        return response


def save_yaml(config, batch, agent, pred_label):
    log_dict = {}
    log_dict["dataset"] = config["dataset"]
    log_dict["id"] = batch['id'][0]
    log_dict["label"] = int(batch['label'][0])
    log_dict["text"] = batch['text'][0]
    log_dict["init_caption"] = agent.description

    round_num = 1
    for i in range(0, len(agent.yaml_log), 4):
        round_key = f"Round {round_num}"
        log_dict[round_key] = {
        "agent": agent.yaml_log[i],
        "analysis": agent.yaml_log[i + 1],
        "question": agent.yaml_log[i + 2],
        "answer": agent.yaml_log[i + 3]
        }
        round_num+=1
    
    log_dict["summary"] = {"output":agent.summary,
                           "pred_label":pred_label}
    log_dict["cost"] = {"completion_tokens":agent.completion_tokens,
                        "prompt_tokens":agent.prompt_tokens}
    
    file_path = os.path.join(config['yaml_root_path'], config["dataset"], f"{batch['id'][0]}.yaml")
    print('yaml dump')
    with open(file_path,'w') as file:
        yaml.dump(log_dict,file, sort_keys=False)

def chat(config,prompt):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(config['log_path'])
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # load data
    harmfulness_data = goat_bench_datasets(root_path=config['dataset_root_path'])
    batch_size = 1
    harmfulness_data_loader = DataLoader(harmfulness_data, batch_size=batch_size, shuffle=False, drop_last=False)

    # initiate agents, add more if needed
    agents = Agents(['Internet User','Internet Supervisor'],logger,prompt,config)

    # save prompt to yaml
    yaml_system = {}
    yaml_system["agent system"] = prompt['system']
    yaml_system["Internet User"] = prompt['Internet User']
    yaml_system["Internet Supervisor"] = prompt['Internet Supervisor']
    yaml_system["SummarySystem"] = prompt['SummarySystem']
    yaml_system["SummaryUser"] = prompt['SummaryUser']
    yaml_system_path = os.path.join(config["yaml_root_path"],config["dataset"], "0000prompt.yaml")
    with open(yaml_system_path,'w') as file:
        yaml.dump(yaml_system,file,sort_keys=False)
    
    result_dict = {'id':[], 'gt_label':[], 'pred_label':[], 'text':[], 'error':[]}
    for batch in tqdm(harmfulness_data_loader):

        agents.img_path = batch['img_path'][0]
        agents.meme_text = batch['text'][0]
        id = batch['id'][0]
        gt_label = int(batch['label'][0])
        logger.info(f'>>>>>memeID:{id}')
        try:
            
            
            agents.initiate_agents()

            agents.chat_for_n_rounds(2)
            response = agents.summarize_and_get_results()
            logger.info('>>>>>endofmemeID')
            logger.info('Summarize and judge:\n'+response)

            pred_label = 1 if 'yes' in response.lower() else 0
            result_dict['id'].append(id)    
            result_dict['gt_label'].append(gt_label)
            result_dict['pred_label'].append(pred_label)

            if gt_label==pred_label:
                logger.info(f'ID:{id} Result: TRUE')
            else:
                logger.info(f'ID:{id} Result: FALSE')
            
        except Exception as e:
            logger.info('Error occurred during conversation...')
            logger.info('Errormessage:',str(e))
            logger.info(f'ID:{id} Result: ERROR')
            result_dict['error'].append(id)
        
        save_yaml(config, batch, agents, pred_label)

        agents.reset()
        

    accuracy = accuracy_score(result_dict['gt_label'], result_dict['pred_label'])
    f1 = f1_score(result_dict['gt_label'], result_dict['pred_label'], average='macro')
    logger.info(f'accuracy:{accuracy:.4f}    f1-score:{f1:.4f}')

    with open(config['result_path'],'w') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
    
    return

if __name__ == '__main__':
    # load config from json
    with open('./config.json','r') as f:
        config = json.load(f)
    
    # load prompt from prompt_dict.py
    from prompt_dict import prompt
    
    chat(config,prompt['harmfulness'])