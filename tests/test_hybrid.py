from job_search.systems.hybrid import fuse


def test_fuse_rewards_jobs_ranked_well_by_both():
    bm25 = ["a", "b", "c"]
    embeddings = ["b", "d", "a"]
    assert fuse([bm25, embeddings], k=60) == ["b", "a", "d", "c"]


def test_fuse_is_deterministic_on_ties_and_cut_to_depth():
    assert fuse([["y"], ["x"]], depth=1) == ["x"]
