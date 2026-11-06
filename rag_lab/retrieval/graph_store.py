"""In-process Knowledge Graph store for relationship-based retrieval (GraphRAG).

Stores entities, directed relationships, and chunk provenance per workspace.
Enables multi-hop graph traversal to discover bridging documents that vector
similarity alone misses.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag_lab.config import settings
from rag_lab.lang import fold, words


@dataclass
class Entity:
    id: str
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)
    source_chunks: list[str] = field(default_factory=list)

    def matches(self, query_folded: str) -> bool:
        """True if the query contains this entity's name or any alias."""
        folded_name = fold(self.name)
        if folded_name and folded_name in query_folded:
            return True
        for alias in self.aliases:
            folded_alias = fold(alias)
            if folded_alias and folded_alias in query_folded:
                return True
        return False


@dataclass
class Relationship:
    source: str
    target: str
    predicate: str
    source_chunk: str
    weight: float = 1.0


@dataclass
class TraversalPath:
    entities: list[str]
    predicates: list[str]
    chunk_ids: list[str]

    def describe(self) -> str:
        parts = []
        for i, ent in enumerate(self.entities):
            parts.append(f"[{ent}]")
            if i < len(self.predicates):
                parts.append(f"--({self.predicates[i]})-->")
        return " ".join(parts)


@dataclass
class GraphTraversalResult:
    visited_entity_ids: list[str]
    paths: list[TraversalPath]
    chunk_ids: list[str]
    subgraph_edges: list[dict[str, Any]]


class KnowledgeGraph:
    """An in-process, zero-dependency knowledge graph for a single workspace."""

    def __init__(self, workspace_id: str = "default") -> None:
        self.workspace_id = workspace_id
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        self._out_edges: dict[str, list[Relationship]] = {}
        self._in_edges: dict[str, list[Relationship]] = {}

    def _normalize_id(self, name: str) -> str:
        return "_".join(words(fold(name))) or "unknown"

    def add_entity(
        self,
        name: str,
        entity_type: str = "CONCEPT",
        aliases: list[str] | None = None,
        source_chunk: str | None = None,
    ) -> Entity:
        norm_id = self._normalize_id(name)
        if norm_id in self.entities:
            ent = self.entities[norm_id]
            if aliases:
                for a in aliases:
                    if a not in ent.aliases:
                        ent.aliases.append(a)
            if source_chunk and source_chunk not in ent.source_chunks:
                ent.source_chunks.append(source_chunk)
            return ent

        ent = Entity(
            id=norm_id,
            name=name.strip(),
            type=entity_type.upper().strip(),
            aliases=list(aliases or []),
            source_chunks=[source_chunk] if source_chunk else [],
        )
        self.entities[norm_id] = ent
        return ent

    def add_relationship(
        self,
        source_name: str,
        target_name: str,
        predicate: str,
        source_chunk: str,
        source_type: str = "CONCEPT",
        target_type: str = "CONCEPT",
        weight: float = 1.0,
    ) -> Relationship:
        src = self.add_entity(source_name, entity_type=source_type, source_chunk=source_chunk)
        tgt = self.add_entity(target_name, entity_type=target_type, source_chunk=source_chunk)
        pred = predicate.strip().upper()

        # Deduplicate identical edges
        for existing in self._out_edges.get(src.id, []):
            if existing.target == tgt.id and existing.predicate == pred:
                if source_chunk and source_chunk != existing.source_chunk:
                    # Update chunks if seen in multiple places
                    pass
                return existing

        rel = Relationship(
            source=src.id,
            target=tgt.id,
            predicate=pred,
            source_chunk=source_chunk,
            weight=weight,
        )
        self.relationships.append(rel)
        self._out_edges.setdefault(src.id, []).append(rel)
        self._in_edges.setdefault(tgt.id, []).append(rel)
        return rel

    def find_entities_in_text(self, text: str) -> list[Entity]:
        """Find all known entities mentioned in a query or passage."""
        folded = fold(text)
        matched: list[Entity] = []
        for ent in self.entities.values():
            if ent.matches(folded):
                matched.append(ent)
        return matched

    def traverse(
        self,
        start_entity_ids: list[str],
        max_hops: int = 2,
    ) -> GraphTraversalResult:
        """Traverse outbound and inbound edges from seed entities up to max_hops.

        Returns all visited entities, traversal paths, and associated chunk IDs.
        """
        visited: set[str] = set()
        chunk_ids: list[str] = []
        paths: list[TraversalPath] = []
        subgraph_edges: list[dict[str, Any]] = []

        def _dfs(
            curr_id: str,
            current_path: list[str],
            predicates: list[str],
            path_chunks: list[str],
            depth: int,
        ) -> None:
            visited.add(curr_id)
            curr_ent = self.entities.get(curr_id)
            if curr_ent:
                for cid in curr_ent.source_chunks:
                    if cid and cid not in chunk_ids:
                        chunk_ids.append(cid)

            if depth >= max_hops:
                return

            # Outbound edges
            for edge in self._out_edges.get(curr_id, []):
                next_id = edge.target
                edge_dict = {
                    "source": curr_ent.name if curr_ent else curr_id,
                    "target": self.entities[next_id].name if next_id in self.entities else next_id,
                    "predicate": edge.predicate,
                    "chunk": edge.source_chunk,
                }
                if edge_dict not in subgraph_edges:
                    subgraph_edges.append(edge_dict)

                if edge.source_chunk and edge.source_chunk not in chunk_ids:
                    chunk_ids.append(edge.source_chunk)

                new_path = current_path + [self.entities[next_id].name if next_id in self.entities else next_id]
                new_preds = predicates + [edge.predicate]
                new_chunks = path_chunks + ([edge.source_chunk] if edge.source_chunk else [])
                paths.append(TraversalPath(entities=new_path, predicates=new_preds, chunk_ids=new_chunks))

                if next_id not in visited:
                    _dfs(next_id, new_path, new_preds, new_chunks, depth + 1)

            # Inbound edges (for bidirectional multi-hop discovery)
            for edge in self._in_edges.get(curr_id, []):
                prev_id = edge.source
                edge_dict = {
                    "source": self.entities[prev_id].name if prev_id in self.entities else prev_id,
                    "target": curr_ent.name if curr_ent else curr_id,
                    "predicate": edge.predicate,
                    "chunk": edge.source_chunk,
                }
                if edge_dict not in subgraph_edges:
                    subgraph_edges.append(edge_dict)

                if edge.source_chunk and edge.source_chunk not in chunk_ids:
                    chunk_ids.append(edge.source_chunk)

                new_path = current_path + [self.entities[prev_id].name if prev_id in self.entities else prev_id]
                new_preds = predicates + [f"REV_{edge.predicate}"]
                new_chunks = path_chunks + ([edge.source_chunk] if edge.source_chunk else [])
                paths.append(TraversalPath(entities=new_path, predicates=new_preds, chunk_ids=new_chunks))

                if prev_id not in visited:
                    _dfs(prev_id, new_path, new_preds, new_chunks, depth + 1)

        for seed in start_entity_ids:
            norm_seed = self._normalize_id(seed) if seed not in self.entities else seed
            seed_ent = self.entities.get(norm_seed)
            seed_name = seed_ent.name if seed_ent else norm_seed
            _dfs(norm_seed, [seed_name], [], [], depth=0)

        return GraphTraversalResult(
            visited_entity_ids=sorted(visited),
            paths=paths,
            chunk_ids=chunk_ids,
            subgraph_edges=subgraph_edges,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "entities": {k: asdict(v) for k, v in self.entities.items()},
            "relationships": [asdict(r) for r in self.relationships],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        kg = cls(workspace_id=data.get("workspace_id", "default"))
        for k, v in data.get("entities", {}).items():
            kg.entities[k] = Entity(**v)
        for r in data.get("relationships", []):
            rel = Relationship(**r)
            kg.relationships.append(rel)
            kg._out_edges.setdefault(rel.source, []).append(rel)
            kg._in_edges.setdefault(rel.target, []).append(rel)
        return kg

    def save(self, filepath: Path | str) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, filepath: Path | str) -> KnowledgeGraph:
        p = Path(filepath)
        if not p.is_file():
            return cls(workspace_id=p.stem)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


def get_graph_path(workspace_id: str) -> Path:
    base = Path(settings.graphs_dir) if hasattr(settings, "graphs_dir") else Path("data/graphs")
    return base / f"{workspace_id}.json"


def load_workspace_graph(workspace_id: str) -> KnowledgeGraph:
    path = get_graph_path(workspace_id)
    return KnowledgeGraph.load(path)


def delete_workspace_graph(workspace_id: str) -> None:
    """Remove knowledge graph file for a workspace if it exists."""
    p = get_graph_path(workspace_id)
    if p.is_file():
        try:
            p.unlink()
        except OSError:
            pass

