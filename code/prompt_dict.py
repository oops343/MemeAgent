from prompt.harmful import *
from prompt.hateful import *
from prompt.misogynistic import *
from prompt.offensive import *
from prompt.sarcastic import *

prompt = {
    "harmfulness": harmful_prompt,

    "hatefulness": hateful_prompt,

    "misogyny": misogynistic_prompt,

    "offensiveness": offensive_prompt,
    
    "sarcasm": sarcastic_prompt
}
