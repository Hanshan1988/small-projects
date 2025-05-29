from qwen_agent.agents import Assistant

# Define LLM
llm_cfg = {
    'model': 'qwen3:8b-q4_K_M', # 'qwen3:8b-q6_K'

    # Use the endpoint provided by Alibaba Model Studio:
    # 'model_type': 'qwen_dashscope',
    # 'api_key': os.getenv('DASHSCOPE_API_KEY'),

    # Use a custom endpoint compatible with OpenAI API:
    'model_server': 'http://localhost:11434/v1', # 'http://localhost:8000/v1',  # api_base
    'api_key': 'ollama',

    # Other parameters:
    # 'generate_cfg': {
    #         # Add: When the response content is `<think>this is the thought</think>this is the answer;
    #         # Do not add: When the response has been separated by reasoning_content and content.
    #         'thought_in_content': True,
    #     },
}

# Define Tools
tools = [
    {'mcpServers': {  # You can specify the MCP configuration file
            'time': {
                'command': 'uvx',
                'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']
            },
            "fetch": {
                "command": "uvx",
                "args": ["mcp-server-fetch"]
            }
        }
    },
    'code_interpreter',  # Built-in tools
]

# Define Agent
bot = Assistant(llm=llm_cfg, function_list=tools)

# Streaming generation
message = 'https://qwenlm.github.io/blog/ Introduce the latest developments of Qwen'
# message = "https://huggingface.co/papers/date/2025-05-23/ Get the titles of the latest papers on Hugging Face"
message = 'https://www.smh.com.au/lifestyle/health-and-wellness/the-simplicity-of-happiness-according-to-bhutans-happiness-guru-20170629-gx14wf.html Summarise the article'

messages = [{
    'role': 'user', 
    'content': message
}]
for responses in bot.run(messages=messages):
    pass
print(responses)
print()
print(responses[-1]['content'])  # Print the final response content

