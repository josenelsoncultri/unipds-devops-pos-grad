import os
from dotenv import load_dotenv
from crewai import LLM
from crewai.llms.cache import strip_cache_breakpoint

load_dotenv()

# Patch CrewAI's LLM message formatter to strip unsupported `cache_breakpoint` key for non-Anthropic providers (e.g. Groq)
_original_format_messages = LLM._format_messages_for_provider

def _patched_format_messages_for_provider(self, messages):
    formatted_messages = _original_format_messages(self, messages)
    if formatted_messages and not getattr(self, "is_anthropic", False):
        for msg in formatted_messages:
            if isinstance(msg, dict):
                strip_cache_breakpoint(msg)
    return formatted_messages

LLM._format_messages_for_provider = _patched_format_messages_for_provider

# Centraliza a inteligência do projeto
nexus_llm = LLM(
    #model="gemini/gemini-3.6-flash",
    #api_key=os.getenv("GEMINI_API_KEY"),
    model="openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),    
    temperature=0.2
)