"""External domain API client for ArXiv scientific publications (FR-401 - FR-403)."""

import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
import requests


class ArXivClient:
    """Queries the public ArXiv API and parses Atom XML entries into normalized records."""

    def __init__(self, base_url: str = "https://export.arxiv.org/api/query", timeout: int = 10):
        self.base_url = base_url
        self.timeout = timeout
        self.ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

    def search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Queries ArXiv with a search string, returning normalized paper records."""
        clean_query = query.strip()
        if not clean_query:
            return {
                "status": "INSUFFICIENT_CONTEXT",
                "error": "Empty search query provided.",
                "papers": [],
            }

        params = {
            "search_query": f"all:{clean_query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            resp = requests.get(self.base_url, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                return {
                    "status": "ERROR",
                    "error": f"ArXiv API returned HTTP {resp.status_code}",
                    "papers": [],
                }

            root = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", self.ns)

            if not entries:
                return {
                    "status": "INSUFFICIENT_CONTEXT",
                    "error": None,
                    "papers": [],
                }

            papers: List[Dict[str, Any]] = []
            for entry in entries:
                title_elem = entry.find("atom:title", self.ns)
                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else "Untitled"

                summary_elem = entry.find("atom:summary", self.ns)
                summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else ""
                # Bounded abstract
                if len(summary) > 400:
                    summary = summary[:400].strip() + "..."

                id_elem = entry.find("atom:id", self.ns)
                url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""

                published_elem = entry.find("atom:published", self.ns)
                published = published_elem.text.strip() if published_elem is not None and published_elem.text else ""

                author_elems = entry.findall("atom:author/atom:name", self.ns)
                authors = [a.text.strip() for a in author_elems if a.text]

                cat_elem = entry.find("arxiv:primary_category", self.ns)
                category = cat_elem.attrib.get("term", "") if cat_elem is not None else ""

                papers.append({
                    "title": title,
                    "authors": authors,
                    "abstract": summary,
                    "url": url,
                    "published_date": published,
                    "category": category,
                })

            return {
                "status": "OK",
                "error": None,
                "papers": papers,
            }

        except Exception as exc:
            return {
                "status": "ERROR",
                "error": f"ArXiv request failed: {str(exc)}",
                "papers": [],
            }
