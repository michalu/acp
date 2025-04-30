# beeai-lora-switching
A simple example of an agent that loads two very small fine-tuned LoRA adapters from different domains, and automatically switches between them based on the user's question.

The agent's configuration including the base model and adapters is stored in a config.json file.

The datasets used for LoRA fine-tuning can be found at:
- ./adapters/lora-psychology/psychology.yaml
- ./adapters/lora-sociology/sociology.yaml

Run server with:

```python
uv run agent.py
```

and the client with:

```python
uv run client.py
```

When the client application starts, it sends three questions to the agent. The agent should apply the appropriate adapter based on the question:

- Question 1 → psychology adapter is applied
- Question 2 → sociology adapter is applied
- Question 3 → no adapter is applied