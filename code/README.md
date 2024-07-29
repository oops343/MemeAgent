### This is the code base for our paper

Ask, Acquire, Understand: A Multimodal Agent-based Framework for Social Abuse Detection in Memes

### The code base is organized as follows:

```bash
MemeAgent
├── config.json
├── example.ipynb
├── goat_dataset
│   ├── harmfulness
│   │   ├── images
│   │   └── test.jsonl
│   ├── hatefulness
│   │   ├── images
│   │   └── test.jsonl
│   ├── misogyny
│   │   ├── images
│   │   └── test.jsonl
│   ├── offensiveness
│   │   ├── images
│   │   └── test.jsonl
│   └── sarcasm
│       ├── images
│       └── test.jsonl
├── history
├── load_dataset.py
├── logs
├── multiAgentChat.py
├── oai_keys.py
├── prompt
│   ├── harmful.py
│   ├── hateful.py
│   ├── misogynistic.py
│   ├── offensive.py
│   └── sarcastic.py
├── prompt_dict.py
├── results
├── summary.py
└── tool
    ├── cogvlm_tool.py
    ├── llava13b_tool.py
    └── qwen_tool.py
```

#### goat_dataset
This directory contains the GOAT-bench dataset. The dataset is organized by meme type (harmfulness, hatefulness, misogyny, offensiveness, sarcasm), and original images can be downloaded from [GOAT-bench](https://huggingface.co/datasets/HKBU-NLP/GOAT-Bench) huggingface repo.

#### logs
This directory contains the logs of each run.

#### results
This directory contains the results of each run, including json files with the ground truth label, the predicted label and more.

#### history
This directory contains the history of the dialogues in yaml files. The history is saved in the format of a list of dictionaries, which can be used to further analyze the dialogues (like testing different definitions in summary).

#### prompt
This directory contains the prompts for each task.

#### tool
This directory contains the LMM tools, which are used in multiAgentChat.py to get image information.

#### config.json
This file contains the configuration for the multiAgentChat.py script. In order to run this code base, you need to check if the paths are correct.

#### oai_keys.py
This file contains the OpenAI API keys and base URL.

#### load_dataset.py
This file loads the dataset and returns the image paths and the labels etc.

#### prompt_dict.py
This file contains the prompt dictionary for each task, which is used in summary.py for selecting correct prompts.

#### multiAgentChat.py
Almost all fuctions are implemented in this file. Dilaogue generation, image information extraction, and the final prediction are all implemented in this file.

#### summary.py
This file is used to summarize the dialogues in history and get the prediction. So after running multiAgentChat.py, you can change the summary prompts to get different predictions.

#### example.ipynb
This notebook shows how to run the code base.