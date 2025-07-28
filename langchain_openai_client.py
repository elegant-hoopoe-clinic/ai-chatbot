"""
OpenAI client using LangChain
Integrates with LangChain's OpenAI implementation
"""

import os
from typing import Optional
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# make sure to get the .env file loaded
from dotenv import load_dotenv

load_dotenv()


class LangChainOpenAIClient:
    """OpenAI client using LangChain's ChatOpenAI"""

    def __init__(
        self, api_key: str, model: str = "gpt-4.1-nano", temperature: float = 0.1
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

        # Initialize LangChain's ChatOpenAI
        self.llm = ChatOpenAI(
            api_key=api_key, model=model, temperature=temperature, max_tokens=1000
        )

    def generate(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Generate response using LangChain's ChatOpenAI"""
        try:
            messages = []

            # Add system message if provided
            if system_message:
                messages.append(SystemMessage(content=system_message))
            else:
                messages.append(
                    SystemMessage(
                        content="You are a helpful assistant that answers questions based only on the provided context."
                    )
                )

            # Add user message
            messages.append(HumanMessage(content=prompt))

            # Generate response
            response = self.llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            return f"❌ Error calling OpenAI API: {e}"

    def is_available(self) -> bool:
        """Check if OpenAI API is accessible"""
        try:
            # Simple test call
            test_messages = [HumanMessage(content="Hello")]
            self.llm.invoke(test_messages)
            return True
        except:
            return False


def get_openai_client(model: str = "gpt-4.1-nano") -> LangChainOpenAIClient:
    """Get LangChain OpenAI client with API key from environment"""
    # Get API key from .env or environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set.\n"
            "Set it with: set OPENAI_API_KEY=your_api_key_here\n"
            "Get your API key from: https://platform.openai.com/api-keys"
        )

    return LangChainOpenAIClient(api_key, model)
