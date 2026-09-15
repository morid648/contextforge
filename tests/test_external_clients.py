"""Tests for Web search client and ArXiv client."""

from unittest.mock import MagicMock, patch
from src.tools.web_search_client import WebSearchClient
from src.tools.external_api_client import ArXivClient


def test_web_search_client_missing_key():
    """Verify graceful handling when web search API key is absent (FR-303)."""
    client = WebSearchClient(api_key="")
    assert client.is_available() is False
    res = client.search("latest news")
    assert res["status"] == "ERROR"
    assert "WEB_SEARCH_API_KEY" in res["error"]
    assert res["results"] == []


def test_arxiv_client_xml_parsing():
    """Verify ArXiv client parses Atom XML correctly."""
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2401.12345v1</id>
        <title>Context Engineering for LLM Agents</title>
        <summary>This paper explores context filtering.</summary>
        <published>2024-01-15T12:00:00Z</published>
        <author><name>Jane Doe</name></author>
        <arxiv:primary_category term="cs.AI"/>
      </entry>
    </feed>"""

    client = ArXivClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = sample_xml

    with patch("requests.get", return_value=mock_resp):
        res = client.search("context engineering")

    assert res["status"] == "OK"
    assert len(res["papers"]) == 1
    paper = res["papers"][0]
    assert paper["title"] == "Context Engineering for LLM Agents"
    assert paper["authors"] == ["Jane Doe"]
    assert paper["category"] == "cs.AI"
    assert "http://arxiv.org/abs/2401.12345v1" in paper["url"]


def test_web_search_client_firecrawl_detection():
    """Verify Firecrawl provider detection and successful search normalization."""
    client = WebSearchClient(api_key="fc-testkey123")
    assert client.is_available() is True
    assert client._provider == "firecrawl"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "data": [
            {
                "title": "Firecrawl Doc",
                "url": "https://docs.firecrawl.dev",
                "description": "Turn websites into LLM-ready markdown",
            }
        ],
    }

    with patch("requests.post", return_value=mock_resp):
        res = client.search("web scraping")

    assert res["status"] == "OK"
    assert len(res["results"]) == 1
    assert res["results"][0]["title"] == "Firecrawl Doc"
    assert res["results"][0]["url"] == "https://docs.firecrawl.dev"
    assert "LLM-ready" in res["results"][0]["snippet"]


def test_web_search_client_websearchapi_detection():
    """Verify WebSearchAPI.ai provider detection and request handling."""
    client = WebSearchClient(api_key="wsa_testkey456")
    assert client.is_available() is True
    assert client._provider == "websearchapi"
