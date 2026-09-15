"""YAML configuration loader for agents, tasks, and system parameters."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigLoader:
    """Loads and caches configuration files for agents and tasks."""

    def __init__(self, base_config_dir: Optional[Path] = None):
        if base_config_dir is None:
            # Look for config/ directory relative to workspace root or current file
            project_root = Path(__file__).resolve().parent.parent.parent
            base_config_dir = project_root / "config"
        self.base_config_dir = Path(base_config_dir)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load_yaml(self, file_path: Path | str) -> Dict[str, Any]:
        """Loads a YAML file and caches the result."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.base_config_dir / path

        path_str = str(path.resolve())
        if path_str in self._cache:
            return self._cache[path_str]

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._cache[path_str] = data
        return data

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Retrieves configuration dictionary for a specific agent."""
        agents_yaml_path = self.base_config_dir / "agents" / "agents.yaml"
        all_agents = self.load_yaml(agents_yaml_path)
        if agent_name not in all_agents:
            available = ", ".join(all_agents.keys())
            raise KeyError(
                f"Agent '{agent_name}' not found in {agents_yaml_path}. Available: [{available}]"
            )
        return all_agents[agent_name]

    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        """Retrieves configuration dictionary for a specific task."""
        tasks_yaml_path = self.base_config_dir / "tasks" / "tasks.yaml"
        all_tasks = self.load_yaml(tasks_yaml_path)
        if task_name not in all_tasks:
            available = ", ".join(all_tasks.keys())
            raise KeyError(
                f"Task '{task_name}' not found in {tasks_yaml_path}. Available: [{available}]"
            )
        return all_tasks[task_name]


# Default singleton instance
default_config_loader = ConfigLoader()
