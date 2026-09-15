"""Tests for configuration loader."""

import pytest
from src.config.config_loader import ConfigLoader, default_config_loader


def test_default_config_loader_initialization():
    """Verify that default config loader finds the config directory."""
    assert default_config_loader.base_config_dir.exists()


def test_get_agent_config_success():
    """Verify loading valid agent configurations."""
    rag_config = default_config_loader.get_agent_config("rag_agent")
    assert rag_config["role"] == "Document RAG Specialist"
    assert "goal" in rag_config
    assert "backstory" in rag_config
    assert rag_config["verbose"] is True

    evaluator_config = default_config_loader.get_agent_config("evaluator_agent")
    assert "Context Evaluator" in evaluator_config["role"]


def test_get_agent_config_missing_raises_keyerror():
    """Verify missing agent name raises KeyError."""
    with pytest.raises(KeyError) as exc_info:
        default_config_loader.get_agent_config("non_existent_agent")
    assert "non_existent_agent" in str(exc_info.value)


def test_get_task_config_success():
    """Verify loading valid task configurations."""
    task_config = default_config_loader.get_task_config("rag_task")
    assert "{query}" in task_config["description"]
    assert "expected_output" in task_config


def test_get_task_config_missing_raises_keyerror():
    """Verify missing task name raises KeyError."""
    with pytest.raises(KeyError):
        default_config_loader.get_task_config("non_existent_task")
