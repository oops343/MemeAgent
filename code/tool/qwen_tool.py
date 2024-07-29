import dashscope

from dashscope import MultiModalConversation

def qwen_generate(file_path1,prompt):

    dashscope.api_key = 'YOUR_API_KEY'
    local_file_path1 = f'file://{file_path1}'

    messages = [{
        'role': 'system',
        'content': [{
            'text': 'You are a helpful assistant.'
        }]
    }, {
        'role':
        'user',
        'content': [

            {
                'image': local_file_path1
            },
            {
                'text': 'Based on this image of a meme, answer the following question:'+prompt
            },
        ]
    }]
    response = MultiModalConversation.call(model='qwen-vl-max', messages=messages)

    return response['output']['choices'][0]['message']['content'][0]['text']