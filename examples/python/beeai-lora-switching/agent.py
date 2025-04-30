import json
import logging
import traceback
from collections.abc import AsyncGenerator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server
import utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize server
server = Server()

def load_configuration() -> dict:
    """Load model and adapter configuration from a JSON file."""
    with open('./config.json', 'r') as f:
        return json.load(f)

def load_adapters(config: dict):
    """Load adapters according to configuration"""
    for adapter_name in config["adapters"]:
        adapter = config["adapters"][adapter_name]
        model.load_adapter(adapter["path"], adapter_name=adapter_name)
        logger.info(f"Adapter {adapter_name} loaded.")

# Load configuration
config = load_configuration()

# Set device
device = utils.get_device().value

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    config["base_model"],
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load adapters
load_adapters(config)

tokenizer = AutoTokenizer.from_pretrained(config["base_model"])

model.to(device)

async def detect_adapter(messages: list[Message]) -> str:
    """Select the most appropriate adapter based on user input.

    Args:
        messages (list[Message]): Incoming list of message objects.

    Returns:
        str: Selected adapter name or "None".
    """
    user_input_text = "\n".join(part.content for part in messages[-1].parts)

    adapter_descriptions = ""
    for adapter_name, adapter_config in config['adapters'].items():
        capability = adapter_config['capability']
        adapter_descriptions += (
            f'- Adapter name: "{adapter_name}": Adapter capability: {capability}.\n'
        )

    prompt = f'''
    You are given a list of adapters. Each adapter has a name and a description of its capabilities.
    You are also given a user's question.
    Your task is:
    - Find the single adapter whose description best matches the user's question or context.
    - If none of the adapters seem appropriate based on their description, return "None".
    - Respond only with the adapter's name or "None". Do not explain your reasoning.
    Adapters:
    {adapter_descriptions}
    User's Question:
    {user_input_text}
    '''.strip()

    results = await utils.inference(model, tokenizer, device, prompt, max_new_tokens=10)
    selected = results[0].replace('"', '').strip()
    return selected

@server.agent()
async def llm(messages: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """LLM agent that selects an adapter and generates a model response."""
    adapter = await detect_adapter(messages)

    if adapter in config["adapters"]:
        logger.info(f"Switching LoRA adapter to: {adapter}")
        model.set_adapter(adapter)
        for _, param in model.named_parameters():
            param.requires_grad = False
    else:
        logger.warning(f"Adapter not found or not selected: {adapter}")

    try:
        user_input_text = "\n".join(part.content for part in messages[-1].parts)

        results = await utils.inference(model, tokenizer, device, user_input_text, max_new_tokens=config["max_tokens"])

        response = results[0] if isinstance(results, list) and results else str(results)

        yield MessagePart(content=response)

    except Exception as e:
        logger.error(f"Error during agent execution: {type(e).__name__}: {e}")
        traceback.print_exc()
        yield MessagePart(content=f"Error occurred: {str(e)}")

# Run the server
server.run()
