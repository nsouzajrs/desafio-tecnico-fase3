"""Pipeline LangChain (LCEL): monta o prompt com paciente + protocolos e chama a LLM.

    entradas -> RunnableLambda(monta user_prompt) -> ChatPromptTemplate -> LLM -> StrOutputParser

O formato do prompt vem de src/prompts.py, o MESMO usado no fine-tuning.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from src.prompts import SYSTEM_PROMPT, build_user_prompt


def build_answer_chain(llm: Runnable) -> Runnable:
    """Espera um dict com: question, chunks (lista de Chunk) e patient_context (texto ou None)."""
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{user_prompt}")])

    def to_prompt_inputs(inputs: dict) -> dict:
        return {"user_prompt": build_user_prompt(inputs["question"], inputs["chunks"], inputs.get("patient_context"))}

    return RunnableLambda(to_prompt_inputs) | prompt | llm | StrOutputParser()
