from types import SimpleNamespace as NS

import numpy as np
import pytest

from job_search.systems.embeddings import cached_embed, embed, rank


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, model, input):
        self.calls.append(list(input))
        # Return items out of order to check that `index` is respected.
        data = [NS(index=i, embedding=[float(len(t)), 1.0]) for i, t in enumerate(input)][::-1]
        return NS(data=data, usage=NS(total_tokens=10 * len(input)))


def fake_client():
    return NS(embeddings=FakeEmbeddings())


def test_embed_batches_and_keeps_order():
    client = fake_client()
    vectors, tokens = embed(client, ["a", "bb", "ccc"], batch=2)
    assert [c for c in client.embeddings.calls] == [["a", "bb"], ["ccc"]]
    assert vectors[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert tokens == 30


def test_cache_is_reused_only_for_the_same_ids(tmp_path):
    path = tmp_path / "jobs.npz"
    client = fake_client()
    first, tokens = cached_embed(client, ["j1", "j2"], ["a", "bb"], path)
    again, cached_tokens = cached_embed(client, ["j1", "j2"], ["a", "bb"], path)
    assert np.array_equal(first, again) and tokens == 20 and cached_tokens == 0
    assert len(client.embeddings.calls) == 1
    with pytest.raises(ValueError):
        cached_embed(client, ["j1", "j3"], ["a", "bb"], path)


def test_rank_by_dot_product():
    jobs = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    queries = np.array([[0.0, 1.0], [1.0, 0.1]], dtype=np.float32)
    assert rank(jobs, queries, ["x", "y", "z"], k=2) == [["y", "z"], ["x", "z"]]
