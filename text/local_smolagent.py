from smolagents import LiteLLMModel, CodeAgent, WebSearchTool, DuckDuckGoSearchTool

model = LiteLLMModel(
    model_id="ollama/qwen3:8b-q4_K_M", # "qwen/qwen2.5-coder-32b-instruct",
    api_base='http://localhost:11434',
    # temperature=0.2,
    flatten_messages_as_text=False # needed for simpler text message style
)

# messages = [{"role": "user", "content": "What is the capital of France? /nothink"}]
# response = model.generate(messages)
# print(response)

agent = CodeAgent(
    tools=[WebSearchTool()], 
    model=model, 
    stream_outputs=True
)

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts? /nothink")