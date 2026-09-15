"""Task factory creating CrewAI tasks from YAML configuration and runtime parameters (FR-601)."""

from typing import Any, List, Optional
from crewai import Agent, Task

from ..config.config_loader import default_config_loader


def create_task(
    task_key: str,
    agent: Agent,
    context: Optional[List[Task]] = None,
    **kwargs: Any,
) -> Task:
    """Instantiates a CrewAI Task using templates from config/tasks/tasks.yaml."""
    cfg = default_config_loader.get_task_config(task_key)

    description_template = cfg["description"]
    # Safely interpolate provided keyword arguments
    description = description_template.format(**kwargs)
    expected_output = cfg["expected_output"]

    task_kwargs: dict = {
        "description": description,
        "expected_output": expected_output,
        "agent": agent,
    }

    if context:
        task_kwargs["context"] = context

    return Task(**task_kwargs)
