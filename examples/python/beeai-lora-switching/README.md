# beeai-lora-switching
A simple example of an agent that loads two very small fine-tuned LoRA adapters from different domains, and automatically switches between them based on the user's question.
The agent's configuration including the base model and adapters is stored in a config.json file.

Run server with:

```python
uv run agent.py
```

and the client with:

```python
uv run client.py
```

When the client application starts, it sends three questions to the agent. The agent applies the appropriate adapter based on the question:

- Question 1 → uses the psychology adapter
- Question 2 → uses the sociology adapter
- Question 3 → no adapter is applied