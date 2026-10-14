import json

from job_search.systems import write_run
from job_search.systems.bm25 import BM25Search, tokenize


def job(id, title, text):
    return {"id": id, "title": title, "text": text}


def test_tokenize_lowercases_stems_and_drops_stopwords():
    assert tokenize("Hey, any Senior Engineering jobs in Canada? 140k+") == ["senior", "engin", "job", "canada", "140k"]


def test_search_ranks_matching_jobs_first():
    jobs = [
        job("a", "Account Executive", "Sell our product to mid-market customers."),
        job("b", "Senior Data Engineer", "Build data pipelines in Python."),
        job("c", "Data Analyst", "Analyse data and build dashboards."),
        job("d", "Nurse", "Care for patients."),
    ]
    engine = BM25Search(jobs)
    ranking = engine.search("senior data engineering jobs", k=3)
    assert ranking[0] == "b"
    assert len(ranking) == 3


def test_write_run(tmp_path):
    path = write_run("bm25", tmp_path / "himalayas-2026-10-13.jsonl.gz", {"s1": ["a", "b"]}, runs_dir=tmp_path)
    assert path.name == "bm25-2026-10-13.jsonl"
    assert json.loads(path.read_text()) == {"spec_id": "s1", "ranking": ["a", "b"]}
