from job_search.systems.extract import EMPTY, accuracy, schema, to_fields
from job_search.systems.filtered import filter_outcome

RATES = {"EUR": 1.0, "USD": 1.1}


def out(**overrides):
    return {**EMPTY, **overrides}


def test_to_fields_marks_unstated_fields_unknown():
    x = to_fields("j", EMPTY, RATES)
    assert x["seniority"] is None and x["countries"] is None and not x["worldwide"]
    assert x["employment_type"] is None and x["categories"] == [] and x["salary_status"] == "none"


def test_to_fields_location_and_salary():
    x = to_fields("j", out(location_rule="countries", countries=["Spain", "France", "Spain"],
                           salary_min=50, salary_max=60, currency="eur", period="hourly",
                           seniority=["Manager", "Senior"], employment_type="Full Time"), RATES)
    assert x["countries"] == ["France", "Spain"] and not x["worldwide"]
    assert (x["salary_min_usd"], x["salary_max_usd"]) == (114_400, 137_280)
    assert x["seniority"] == ["Senior", "Manager"] and x["employment_type"] == "Full Time"
    assert to_fields("j", out(location_rule="worldwide"), RATES)["worldwide"]
    # "countries" with an empty list is treated as not stated.
    assert to_fields("j", out(location_rule="countries"), RATES)["countries"] is None


def test_unknown_level_or_location_never_excludes_a_job():
    x = to_fields("j", out(employment_type="Contractor"), RATES)
    assert filter_outcome(x, {"seniority": ["Senior"], "country": ["Canada"]}) == "unknown"
    assert filter_outcome(x, {"employment_type": ["Full Time"]}) == "no"


def test_accuracy_counts_correct_unknown_and_wrong():
    label = {"seniority": ["Senior"], "categories": ["Developer"], "countries": ["Canada"], "worldwide": False,
             "employment_type": "Full Time", "salary_status": "stated", "salary_min_usd": 100_000}
    good = to_fields("a", out(seniority=["Senior"], categories=["Developer", "Data Science"],
                              location_rule="countries", countries=["Canada"], employment_type="Full Time",
                              salary_min=102_000, salary_max=120_000, currency="USD", period="annual"), RATES)
    silent = to_fields("b", EMPTY, RATES)
    a = accuracy({"a": label, "b": label}, {"a": good, "b": silent})
    assert a["seniority"] == {"correct": 1, "unknown": 1, "n": 2}
    assert a["category"] == {"correct": 1, "unknown": 1, "n": 2}
    assert a["location"] == {"correct": 1, "unknown": 1, "n": 2}
    assert a["salary"] == {"correct": 1, "unknown": 1, "n": 2}


def test_schema_has_not_stated_options():
    s = schema(["Developer"], ["Canada"], ["Full Time"])
    assert s["properties"]["employment_type"]["enum"] == ["Full Time", "not stated"]
    assert "not stated" in s["properties"]["location_rule"]["enum"]
