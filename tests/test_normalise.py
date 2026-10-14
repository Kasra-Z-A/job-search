import gzip
import hashlib
import json

import httpx
import pytest

from job_search.normalise import (
    fetch_ecb_rates,
    html_to_text,
    normalise_job,
    normalise_salary,
    normalise_seniority,
    normalise_snapshot,
)

RATES = {"EUR": 1.0, "USD": 1.1, "GBP": 0.85}


def raw_job(**overrides):
    job = {
        "guid": "https://himalayas.app/companies/acme/jobs/data-engineer",
        "title": " Data Engineer ",
        "companyName": "Acme",
        "pubDate": 1791876793,
        "expiryDate": 1797060792,
        "employmentType": "Full Time",
        "seniority": ["Senior"],
        "locationRestrictions": ["Germany", "Austria"],
        "timezoneRestrictions": [2, 1],
        "parentCategories": ["Developer", "Data Science"],
        "categories": ["Data-Engineer"],
        "minSalary": None,
        "maxSalary": None,
        "currency": None,
        "salaryPeriod": "annual",
        "description": "<p>Build pipelines.</p>",
    }
    return {**job, **overrides}


def test_salary_converts_currency_and_period_to_yearly_usd():
    s = normalise_salary(raw_job(minSalary=50, maxSalary=60, currency="EUR", salaryPeriod="hourly"), RATES)
    assert s == {"salary_status": "stated", "salary_min_usd": 114_400, "salary_max_usd": 137_280}

    s = normalise_salary(raw_job(minSalary=4250, maxSalary=4250, currency="GBP", salaryPeriod="monthly"), RATES)
    assert s["salary_min_usd"] == 66_000


@pytest.mark.parametrize(
    ("overrides", "status"),
    [
        ({}, "none"),
        ({"minSalary": 90_000, "maxSalary": 120_000, "currency": "TWD"}, "unconvertible"),
        ({"minSalary": 90_000, "maxSalary": 120_000, "currency": "USD", "salaryPeriod": "daily"}, "unconvertible"),
        # Monthly pay labelled as annual.
        ({"minSalary": 1500, "maxSalary": 2000, "currency": "USD"}, "implausible"),
        ({"minSalary": 900_000, "maxSalary": 1_200_000, "currency": "USD"}, "implausible"),
    ],
)
def test_salary_status_when_not_usable(overrides, status):
    s = normalise_salary(raw_job(**overrides), RATES)
    assert s["salary_status"] == status
    assert s["salary_min_usd"] is None and s["salary_max_usd"] is None


def test_salary_with_one_bound_uses_it_for_both():
    s = normalise_salary(raw_job(maxSalary=100_000, currency="USD"), RATES)
    assert (s["salary_min_usd"], s["salary_max_usd"]) == (100_000, 100_000)


def test_seniority_is_deduplicated_and_ordered():
    assert normalise_seniority(["Manager", "Senior", "Senior"]) == ["Senior", "Manager"]
    assert normalise_seniority(None) == []
    with pytest.raises(ValueError):
        normalise_seniority(["Wizard"])


def test_html_to_text_keeps_blocks_and_list_items():
    markup = "<div>About <b>Acme</b> &amp; co</div><ul><li>Python</li><li>SQL</li></ul><p></p><br>Apply&nbsp;now"
    assert html_to_text(markup) == "About Acme & co\n- Python\n- SQL\nApply now"
    assert html_to_text(None) == ""
    assert html_to_text("<p>wrapped\nin the source</p>") == "wrapped in the source"


def test_normalise_job_fields():
    job = normalise_job(raw_job(), RATES)
    assert job["title"] == "Data Engineer"
    assert job["countries"] == ["Austria", "Germany"]
    assert job["worldwide"] is False
    assert job["categories"] == ["Data Science", "Developer"]
    assert job["published_at"] == "2026-10-13T07:33:13+00:00"
    assert job["text"] == "Build pipelines."

    open_job = normalise_job(raw_job(locationRestrictions=[], parentCategories=None), RATES)
    assert open_job["worldwide"] is True
    assert open_job["categories"] == []


def test_fetch_ecb_rates_parses_csv():
    body = "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE\n" \
           "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2026-10-13,1.1188\n" \
           "EXR.D.GBP.EUR.SP00.A,D,GBP,EUR,SP00,A,2026-10-13,0.8446\n"
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=body)))
    assert fetch_ecb_rates("2026-10-13", client) == {"EUR": 1.0, "GBP": 0.8446, "USD": 1.1188}

    empty = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, text="")))
    with pytest.raises(ValueError):
        fetch_ecb_rates("2026-10-11", empty)


def write_snapshot(tmp_path, jobs, sha=None):
    snapshot = tmp_path / "himalayas-2026-10-13.jsonl.gz"
    with gzip.open(snapshot, "wt") as f:
        f.writelines(json.dumps(j) + "\n" for j in jobs)
    digest = sha or hashlib.sha256(snapshot.read_bytes()).hexdigest()
    (tmp_path / "himalayas-2026-10-13.manifest.json").write_text(json.dumps({"sha256": digest}))
    fx = tmp_path / "fx"
    fx.mkdir()
    (fx / "ecb-2026-10-13.json").write_text(json.dumps({"per_eur": RATES}))
    return snapshot, fx


def test_normalise_snapshot_writes_records_and_summary(tmp_path):
    snapshot, fx = write_snapshot(tmp_path, [raw_job(), raw_job(guid="b", minSalary=100_000, currency="USD")])

    summary = normalise_snapshot(snapshot, tmp_path / "out", fx)

    with gzip.open(summary["output"], "rt") as f:
        records = [json.loads(line) for line in f]
    assert [r["id"] for r in records] == [raw_job()["guid"], "b"]
    assert summary["salary_status"] == {"none": 1, "stated": 1}


def test_normalise_snapshot_refuses_a_modified_snapshot(tmp_path):
    snapshot, fx = write_snapshot(tmp_path, [raw_job()], sha="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        normalise_snapshot(snapshot, tmp_path / "out", fx)
