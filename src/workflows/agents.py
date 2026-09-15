"""Agent factory instantiating CrewAI agents from YAML configuration (FR-601)."""

from typing import Any, List, Optional
from crewai import Agent

from ..config.config_loader import default_config_loader


def create_agent(
    agent_key: str,
    tools: Optional[List[Any]] = None,
    llm: Any = None,
    verbose: Optional[bool] = None,
) -> Agent:
    """Instantiates a CrewAI Agent using parameters defined in config/agents/agents.yaml."""
    cfg = default_config_loader.get_agent_config(agent_key)

    is_verbose = verbose if verbose is not None else cfg.get("verbose", True)

    agent_kwargs: dict = {
        "role": cfg["role"],
        "goal": cfg["goal"],
        "backstory": cfg["backstory"],
        "verbose": is_verbose,
        "tools": tools or [],
        "allow_delegation": False,
    }

    if llm is not None:
        agent_kwargs["llm"] = llm

    return Agent(**agent_kwargs)


def create_rag_agent(tools: Optional[List[Any]] = None, llm: Any = None) -> Agent:
    return create_agent("rag_agent", tools=tools, llm=llm)


def create_memory_agent(tools: Optional[List[Any]] = None, llm: Any = None) -> Agent:
    return create_agent("memory_agent", tools=tools, llm=llm)


def create_web_agent(tools: Optional[List[Any]] = None, llm: Any = None) -> Agent:
    return create_agent("web_search_agent", tools=tools, llm=llm)


def create_external_api_agent(tools: Optional[List[Any]] = None, llm: Any = None) -> Agent:
    return create_agent("external_api_agent", tools=tools, llm=llm)


def create_evaluator_agent(tools: Optional[List[Any]] = None, llm: Any = None) -> Agent:
    return create_agent("evaluator_agent", tools=tools, llm=llm)


def create_synthesizer_agent(tools: Optional[List[Any]] = None, llm: Any = None) -> Agent:
    return create_agent("synthesizer_agent", tools=tools, llm=llm)
