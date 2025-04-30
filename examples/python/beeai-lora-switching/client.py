import asyncio
from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart
import os

questions = [
    "What are the psychological effects of prolonged virtual embodiment in avatars?",
    "How might infrastructural opacity reproduce algorithmic hegemony?",
    "What is the capital of Poland?"
]

async def run_sync():
    async with Client(base_url="http://localhost:8000") as client:
        for q in questions:
            run = await client.run_sync(
                agent="llm",
                input=[Message(parts=[MessagePart(content=q)])]
            )
            output = "\n".join(part.content for part in run.output[-1].parts)
            print (f'Question:{os.linesep}{q}{os.linesep}Response:{os.linesep}{output}{os.linesep}')

if __name__ == "__main__":
    asyncio.run(run_sync())