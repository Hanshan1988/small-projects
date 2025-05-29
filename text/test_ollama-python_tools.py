from ollama import ChatResponse, chat, generate
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# from transformers import AutoTokenizer
import os
import requests
from bs4 import BeautifulSoup

load_dotenv() 

model_id = 'qwen3:4b-q8_0' # 'qwen2.5:7b-instruct-q6_K'
thinking_mode = False # For thinking models with thinking disabled

def ask_stronger_model(question) -> str:
    """
    Prompts a stronger model to answer a question when the question is too complex for the current model.

    Args:
      question (str): the question to ask to the model

    Returns:
      str: The response from the model
    """
    provider = "nebius"
    hf_model_id = "google/gemma-3-27b-it"
    # hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    client = InferenceClient(
        provider=provider,
        api_key=os.environ['HF_TOKEN'],
    )

    messages = [
      {
        "role": "user",
        "content": question
      }
    ]

    # prompt = hf_tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False) 
    # generation = client.text_generation(prompt, model=hf_model_id, temperature=0.1, max_new_tokens=500)

    completion = client.chat.completions.create(
        model=hf_model_id,
        messages=messages,
        max_tokens=500,
    )

    return completion.choices[0].message.content

def python_tool(
    python_code: str,
) -> str:  
    """
    Evaluate a Python code snippet and return the result.
    """
    return eval(python_code)


def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
    a (int): The first number
    b (int): The second number

    Returns:
    int: The sum of the two numbers
    """

    # The cast is necessary as returned tool call arguments don't always conform exactly to schema
    # E.g. this would prevent "what is 30 + 12" to produce '3012' instead of 42
    return int(a) + int(b)


def subtract_two_numbers(a: int, b: int) -> int:
    """
    Subtract two numbers
    """

    # The cast is necessary as returned tool call arguments don't always conform exactly to schema
    return int(a) - int(b)


def fetch_webpage_content(url: str) -> str:
    """
    Fetch the text content of a webpage given its URL.
    
    Args:
        url: The URL of the webpage to fetch
        
    Returns:
        The text content of the webpage
    """
    try:
        # Set headers to mimic a real browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit text length to avoid overwhelming the model
        max_length = 5000
        if len(text) > max_length:
            text = text[:max_length] + "... [Content truncated]"
        
        return text
        
    except requests.exceptions.RequestException as e:
        return f"Error fetching webpage: {str(e)}"
    except Exception as e:
        return f"Error processing webpage content: {str(e)}"


# Tools can still be manually defined and passed into chat
subtract_two_numbers_tool = {
    'type': 'function',
    'function': {
        'name': 'subtract_two_numbers',
        'description': 'Subtract two numbers',
        'parameters': {
            'type': 'object',
            'required': ['a', 'b'],
            'properties': {
                'a': {'type': 'integer', 'description': 'The first number'},
                'b': {'type': 'integer', 'description': 'The second number'},
            },
        },
    },
}

user_message = 'How can I learn about multi-turn Reinformcement Learning in an agent environment?'
user_message = 'Summarise the top 10 papers from https://huggingface.co/papers/date/2025-05-23'

messages = [{
   'role': 'user', 
   'content': user_message
}]
print('Prompt:', messages[0]['content'])

available_functions = {
    'add_two_numbers': add_two_numbers,
    'subtract_two_numbers': subtract_two_numbers,
    'ask_stronger_model': ask_stronger_model,
    'fetch_webpage_content' : fetch_webpage_content
}

response: ChatResponse = chat(
    model_id, # 'llama3.1',
    messages=messages,
    tools=[
        add_two_numbers, 
        subtract_two_numbers_tool, 
        ask_stronger_model, 
        fetch_webpage_content
    ],
    # stream=True
)

# for chunk in response:
#     # Print model content
#     print(chunk.message.content, end='', flush=True)
#     # Print the tool call
#     if chunk.message.tool_calls:
#         print(chunk.message.tool_calls)

# Only needed to chat with the model using the tool call results
if response.message.tool_calls:
    # There may be multiple tool calls in the response
    for tool in response.message.tool_calls:
        if function_to_call := available_functions.get(tool.function.name):
            output = function_to_call(**tool.function.arguments)
            # Add the function response to messages for the model to use
            messages.append(response.message)
            messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})
else:
    print('No tool calls returned from model')

# Get final response from model with function outputs
if 'qwen3' in model_id.lower() and not thinking_mode:
    from transformers import AutoTokenizer
    # View final prompt
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
    prompt = tokenizer.apply_chat_template(messages, tokenize=False) + '/nothink'
    final_response = generate(model_id, prompt=prompt)
    print('Final response:', final_response.response)
else: # assert not reasoning model
    final_response = chat(model_id, messages=messages)
    print('Final response:', final_response.message.content)
