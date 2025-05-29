from llama_cpp import Llama

llm = Llama(
      model_path="/Users/hanshan/models/Salesforce/Llama-xLAM-2-8b-fc-r-gguf/Llama-xLAM-2-8B-fc-r-Q6_K.gguf"
)

messages = [
    {"role": "user", "content": "Hi, how are you?"},
    {"role": "assistant", "content": "Thanks. I am doing well. How can I help you?"},
    {"role": "user", "content": "What's the weather like in London?"},
]

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature to return"}
            },
            "required": ["location"]
        }
    }
]

output = llm.create_chat_completion(
      messages = messages,
      tools=tools,
      # tool_choice={
      #   "type": "function",
      #   "function": {
      #     "name": "UserDetail"
      #   }
      # }
)
print(output['choices'][0]['message'])

