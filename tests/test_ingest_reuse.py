"""Re-syncing a workspace embeds only chunks whose text is new; everything else keeps its stored vector."""
import hashlib
import json

import chromadb
import numpy as np
import pytest

import rag_lab.ingestion.ingest as ing
from rag_lab.config import settings
from rag_lab.retrieval.embedding_provider import FINGERPRINT_KEY


class CountingEmbedder:
    """Deterministic vectors from the text, and a record of exactly what was asked for."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [self.vector(t) for t in texts]

    @staticmethod
    def vector(text):
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 + 0.01 for b in digest[:12]]

    @property
    def embedded(self) -> list[str]:
        return [t for call in self.calls for t in call]


@pytest.fixture
def store(tmp_path, monkeypatch):
    for field in ("chroma_path", "bm25_dir", "uploads_dir"):
        monkeypatch.setattr(settings, field, str(tmp_path / "store" / field))
    embedder = CountingEmbedder()
    monkeypatch.setattr(ing, "get_embedding_model", lambda *a, **k: embedder)
    source = tmp_path / "kb" / "ws"
    source.mkdir(parents=True)
    for name in ("alpha", "bravo", "charlie"):
        (source / f"{name}.md").write_text(f"# {name.title()}\n\n" + " ".join(f"{name} sentence {i} about the policy." for i in range(40)))
    return source, embedder


def sync(source, **kw):
    result = ing.ingest_workspace("ws", str(source), **kw)
    assert result.ok, result.error
    return result


def collection():
    return chromadb.PersistentClient(path=settings.chroma_path).get_collection("ws_ws")


def contents():
    page = collection().get(include=["documents", "embeddings", "metadatas"])
    return {i: (d, np.array(e, dtype=np.float64), m) for i, d, e, m in
            zip(page["ids"], page["documents"], page["embeddings"], page["metadatas"])}


def same_contents(first, second) -> bool:
    """Ids, texts and metadata equal; vectors equal to float32 rounding (Chroma re-stores a reused vector with
    differences of about 3e-8, which no cosine ranking can see)."""
    return first.keys() == second.keys() and all(
        first[i][0] == second[i][0] and first[i][2] == second[i][2] and np.allclose(first[i][1], second[i][1], atol=1e-6, rtol=0)
        for i in first
    )


def test_the_first_sync_embeds_everything_and_an_unchanged_one_embeds_nothing(store):
    source, embedder = store
    first = sync(source)
    assert first.reused == 0 and first.embedded == first.chunks and len(embedder.embedded) == first.chunks
    embedder.calls.clear()
    again = sync(source)
    assert again.unchanged and embedder.calls == []


def test_a_changed_passage_prefix_defeats_the_unchanged_fast_path(store, monkeypatch):
    """The vectors embed the prefix; a sync must re-embed when EMBEDDING_PASSAGE_PREFIX changes,
    not report the workspace unchanged while queries embed with a different prefix."""
    source, embedder = store
    sync(source)
    embedder.calls.clear()

    monkeypatch.setattr(settings, "embedding_passage_prefix", "passage: ")
    again = sync(source)

    assert again.unchanged is False
    assert again.embedded == again.chunks and again.reused == 0, "every chunk is embedded with the new prefix"


def test_editing_one_file_embeds_only_its_new_chunks(store):
    source, embedder = store
    first = sync(source)
    embedder.calls.clear()
    (source / "bravo.md").write_text("# Bravo\n\n" + " ".join(f"bravo CHANGED sentence {i}." for i in range(40)))
    second = sync(source)
    assert 0 < second.embedded < first.chunks and second.reused + second.embedded >= second.chunks - 1
    assert all("CHANGED" in t for t in embedder.embedded), "no unchanged chunk was embedded again"
    assert second.reused > 0 and second.reused >= first.chunks // 2


def test_the_result_is_identical_to_a_full_re_embed(store, tmp_path, monkeypatch):
    source, embedder = store
    sync(source)
    (source / "bravo.md").write_text("# Bravo\n\n" + " ".join(f"bravo CHANGED sentence {i}." for i in range(40)))
    (source / "delta.md").write_text("# Delta\n\nA brand new document about travel expenses and approvals.")
    (source / "charlie.md").unlink()
    sync(source)
    incremental = contents()

    for field in ("chroma_path", "bm25_dir"):
        monkeypatch.setattr(settings, field, str(tmp_path / "fresh" / field))
    sync(source)
    assert same_contents(contents(), incremental), "ids, texts, vectors and metadata match a from-scratch build"


def test_adding_and_removing_files(store):
    source, embedder = store
    first = sync(source)
    embedder.calls.clear()
    (source / "alpha.md").unlink()
    removed = sync(source)
    assert embedder.calls == [] and removed.embedded == 0 and 0 < removed.chunks < first.chunks
    (source / "echo.md").write_text("# Echo\n\nOnly this file is new.")
    added = sync(source)
    assert embedder.embedded == ["# Echo\nOnly this file is new."] or all("Only this file is new" in t for t in embedder.embedded)
    assert added.embedded >= 1 and added.reused == removed.chunks


def test_force_recomputes_everything(store):
    source, embedder = store
    first = sync(source)
    embedder.calls.clear()
    forced = sync(source, force=True)
    assert forced.reused == 0 and len(embedder.embedded) == first.chunks


def test_identical_chunk_text_is_embedded_once(store):
    source, embedder = store
    (source / "alpha.md").write_text("# Same\n\nThe very same paragraph in two files.")
    (source / "bravo.md").write_text("# Same\n\nThe very same paragraph in two files.")
    (source / "charlie.md").unlink()
    result = sync(source)
    assert result.chunks == 2 and result.embedded == 1 and len(embedder.embedded) == 1


def test_a_changed_clearance_keeps_the_vector_but_gets_the_new_metadata(store):
    source, embedder = store
    sync(source)
    before = {m["source"]: m["clearance"] for _, _, m in contents().values()}
    assert set(before.values()) == {"public"}
    embedder.calls.clear()
    (source / "bravo.md").write_text("---\nclearance: secret\n---\n" + (source / "bravo.md").read_text())
    second = sync(source)
    after = {m["source"]: m["clearance"] for _, _, m in contents().values()}
    assert after["bravo.md"] == "secret" and after["alpha.md"] == "public"
    assert embedder.calls == [] and second.embedded == 0, "same text, same vector, nothing to embed"


def manifest_path():
    return ing._get_manifest_path("ws")


@pytest.mark.parametrize("why", ["old-manifest-without-prefix", "prefix-changed", "model-changed"])
def test_reuse_is_off_whenever_the_stored_vectors_cannot_be_trusted(store, monkeypatch, why):
    source, embedder = store
    first = sync(source)
    embedder.calls.clear()
    (source / "alpha.md").write_text("# Alpha\n\nChanged so that a sync is needed at all.")
    if why == "old-manifest-without-prefix":
        data = json.loads(manifest_path().read_text())
        del data["passage_prefix"]
        manifest_path().write_text(json.dumps(data))
    elif why == "prefix-changed":
        monkeypatch.setattr(settings, "embedding_passage_prefix", "passage: ")
    elif why == "model-changed":
        monkeypatch.setattr(ing, "embedding_fingerprint", lambda: "fastembed:another/model")
    result = sync(source)
    assert result.reused == 0 and len(embedder.embedded) >= first.chunks - 2, f"{why}: everything is embedded again"


def test_an_unreadable_previous_index_means_embedding_everything_not_failing(store, monkeypatch):
    source, embedder = store
    first = sync(source)
    embedder.calls.clear()
    (source / "alpha.md").write_text("# Alpha\n\nChanged so that a sync is needed at all.")
    real, reads = chromadb.PersistentClient, {"n": 0}

    class Client:
        def __init__(self, path):
            self._client = real(path=path)

        def get_collection(self, name):
            reads["n"] += 1
            if reads["n"] == 1:  # the read that looks for reusable vectors
                raise RuntimeError("the index cannot be read")
            return self._client.get_collection(name)

        def __getattr__(self, item):
            return getattr(self._client, item)

    monkeypatch.setattr(chromadb, "PersistentClient", lambda path: Client(path))
    result = sync(source)
    assert result.reused == 0 and len(embedder.embedded) >= first.chunks - 2


def test_a_collection_built_by_another_model_is_not_reused_even_if_the_manifest_agrees(store, monkeypatch):
    """The collection's own stamp wins: its chunk texts are all there, but its vectors came from another model."""
    source, embedder = store
    original = ing.embedding_fingerprint
    real = original()
    monkeypatch.setattr(ing, "embedding_fingerprint", lambda: "fastembed:another/model")
    first = sync(source)  # collection and manifest both stamped with the other model
    monkeypatch.setattr(ing, "embedding_fingerprint", original)
    data = json.loads(manifest_path().read_text())
    data["embedding"] = real  # the manifest now agrees with the runtime; the collection does not
    manifest_path().write_text(json.dumps(data))
    assert collection().metadata[FINGERPRINT_KEY] == "fastembed:another/model"
    embedder.calls.clear()
    (source / "alpha.md").write_text("# Alpha\n\nChanged so that a sync is needed at all.")
    result = sync(source)
    assert result.reused == 0 and len(embedder.embedded) >= first.chunks - 2


def test_the_manifest_records_the_prefix_so_the_next_sync_can_reuse(store):
    source, _ = store
    sync(source)
    assert json.loads(manifest_path().read_text())["passage_prefix"] == settings.embedding_passage_prefix


def test_the_real_model_agrees_with_itself_after_an_incremental_sync(tmp_path, monkeypatch):
    """With the real ONNX model: unchanged chunks keep their vectors; new ones match a fresh embedding."""
    import numpy as np

    from rag_lab.retrieval.embedding_provider import get_embedding_model

    for field in ("chroma_path", "bm25_dir", "uploads_dir"):
        monkeypatch.setattr(settings, field, str(tmp_path / "store" / field))
    source = tmp_path / "kb" / "ws"
    source.mkdir(parents=True)
    (source / "a.md").write_text("# Leave\n\nEmployees receive 25 days of paid annual leave per year.")
    (source / "b.md").write_text("# Expenses\n\nReceipts must be submitted within 30 days of purchase.")
    sync(source)
    old = {d: np.array(e) for d, e in zip(*[collection().get(include=["documents", "embeddings"])[k] for k in ("documents", "embeddings")])}
    (source / "b.md").write_text("# Expenses\n\nReceipts must be submitted within 14 days of purchase.")
    result = sync(source)
    assert result.reused == 1 and result.embedded == 1
    new = {d: np.array(e) for d, e in zip(*[collection().get(include=["documents", "embeddings"])[k] for k in ("documents", "embeddings")])}
    kept = [d for d in new if d in old]
    assert kept and all(float(np.abs(new[d] - old[d]).max()) < 1e-6 for d in kept), "reused: float32 rounding at most"
    changed = next(d for d in new if d not in old)
    direct = np.array(get_embedding_model().embed_documents([changed])[0])
    assert float(np.abs(new[changed] - direct).max()) < 1e-3
