from __future__ import annotations

import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from .extractors import DefaultMemoryExtractor, MemoryCandidate
from .pipeline import DefaultMemoryReconciler, MemoryPipelineConfig
from .repository import BaseRepository, EngineState, InMemoryRepository
from .types import (
    ContextPack,
    EventType,
    EvidenceItem,
    IngestionItem,
    MemoryAction,
    MemoryKind,
    MemoryLayer,
    MemoryRecord,
    MemoryScope,
    MemoryState,
    MemoryVersion,
    ProjectionStatus,
    RelationEdge,
    SearchFilters,
    SearchResult,
)
from .utils import (
    cosine_similarity,
    deep_copy_json,
    extract_entities,
    flatten_text_parts,
    generate_id,
    hashed_embedding,
    normalize_text,
    stable_hash,
    summarize_text,
    tokenize,
    unique_tokens,
    utc_now,
)


class MemoryEngine:
    def __init__(
        self,
        repository: BaseRepository | None = None,
        chat_model: Any | None = None,
        embedding_model: Any | None = None,
        extractor: DefaultMemoryExtractor | None = None,
        pipeline: MemoryPipelineConfig | None = None,
    ):
        self.repository = repository or InMemoryRepository()
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.pipeline = pipeline or MemoryPipelineConfig()
        if pipeline is not None:
            self.extractor = pipeline.extractor or DefaultMemoryExtractor()
        else:
            self.extractor = extractor or DefaultMemoryExtractor()
        self.brain = self.pipeline.brain
        self.reconciler = self.pipeline.reconciler or DefaultMemoryReconciler(
            pending_confidence_threshold=self.pipeline.pending_confidence_threshold
        )
        self.reranker = self.pipeline.reranker
        self.asset_processor = self.pipeline.asset_processor

    def add(
        self,
        *,
        messages: list[dict[str, Any]] | None = None,
        items: list[IngestionItem | dict[str, Any]] | None = None,
        text: str | None = None,
        modality: str = "text",
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        scope = MemoryScope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        ingestion_items = self._normalize_ingestion_items(
            messages=messages,
            items=items,
            text=text,
            modality=modality,
            metadata=metadata,
        )
        return self._ingest(
            operation_name="add",
            idempotency_key=idempotency_key,
            scope=scope,
            ingestion_items=ingestion_items,
            strategy="v2",
        )

    def capture(
        self,
        *,
        messages: list[dict[str, Any]] | None = None,
        events: list[dict[str, Any] | IngestionItem] | None = None,
        items: list[IngestionItem | dict[str, Any]] | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        scope = MemoryScope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        ingestion_items = self._normalize_capture_items(messages=messages, events=events, items=items)
        return self._ingest(
            operation_name="capture",
            idempotency_key=idempotency_key,
            scope=scope,
            ingestion_items=ingestion_items,
            strategy="v3",
        )

    def consolidate(
        self,
        *,
        scope: MemoryScope | dict[str, Any] | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 500,
        budget_ms: int = 50,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        resolved_scope = scope if isinstance(scope, MemoryScope) else MemoryScope.from_dict(scope)
        if resolved_scope.is_empty():
            resolved_scope = MemoryScope(user_id=user_id, agent_id=agent_id, run_id=run_id)

        def operation(state: EngineState) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            self._refresh_activation_projection(state)
            summary, events = self._run_consolidation_pass(
                state=state,
                scope=resolved_scope,
                limit=limit,
                budget_ms=budget_ms,
            )
            dormancy_transitions = self._apply_dormancy_transitions(state)
            if summary["changed"] or dormancy_transitions:
                self._rebuild_projections(state)
            state.projections["consolidation"] = {
                "last_run_at": utc_now(),
                "scope": resolved_scope.to_dict(),
                "processed_records": summary["processed_records"],
                "promotion_count": len(summary["promotions"]),
                "skipped_count": len(summary["skipped"]),
                "dormancy_transition_count": len(dormancy_transitions),
                "promotions": list(summary["promotions"]),
                "dormancy_transitions": list(dormancy_transitions),
            }
            return {
                "status": "ok",
                "scope": resolved_scope.to_dict(),
                "processed_records": summary["processed_records"],
                "promotions": summary["promotions"],
                "skipped": summary["skipped"],
                "dormancy_transitions": dormancy_transitions,
                "projection_status": ProjectionStatus.from_dict(state.projections.get("status")).to_dict(),
            }, events + dormancy_transitions

        return self.repository.transaction("consolidate", idempotency_key, operation)

    def search(
        self,
        *,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 5,
        filters: SearchFilters | dict[str, Any] | None = None,
        include_trace: bool = False,
    ) -> dict[str, Any]:
        search_filters = self._coerce_filters(
            filters=filters,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=limit,
        )
        query = normalize_text(query)
        if not self._brain_enabled():
            state = self.repository.load_state()
            ranked = self._rank_records(state=state, query=query, search_filters=search_filters)
            projection_status = ProjectionStatus.from_dict(state.projections.get("status"))
            payload = self._build_search_payload(
                state=state,
                query=query,
                search_filters=search_filters,
                ranked=ranked,
                projection_status=projection_status,
            )
            if self._should_include_trace(include_trace):
                payload["trace"] = self._search_trace_payload(
                    state=state,
                    query=query,
                    ranked=ranked,
                    search_filters=search_filters,
                    working_memory=[],
                    dormancy_transitions=[],
                )
            return payload

        def operation(state: EngineState) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            dormancy_transitions = self._prepare_brain_state(state)
            if dormancy_transitions:
                self._rebuild_projections(state)
            ranked = self._rank_records(state=state, query=query, search_filters=search_filters)
            retrieval_events = self._touch_retrieval_hits(state=state, ranked=ranked[: search_filters.limit])
            if retrieval_events:
                self._rebuild_projections(state)
            projection_status = ProjectionStatus.from_dict(state.projections.get("status"))
            payload = self._build_search_payload(
                state=state,
                query=query,
                search_filters=search_filters,
                ranked=ranked,
                projection_status=projection_status,
            )
            if self._should_include_trace(include_trace):
                payload["trace"] = self._search_trace_payload(
                    state=state,
                    query=query,
                    ranked=ranked,
                    search_filters=search_filters,
                    working_memory=[],
                    dormancy_transitions=dormancy_transitions,
                )
            return payload, dormancy_transitions + retrieval_events

        return self.repository.transaction("search", None, operation)

    def build_context(
        self,
        *,
        query: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 8,
        max_tokens: int = 480,
        filters: SearchFilters | dict[str, Any] | None = None,
        include_debug: bool = False,
        include_trace: bool = False,
        context_policy: str = "compact",
    ) -> ContextPack:
        normalized_messages = [
            {"role": item.get("role", "user"), "content": normalize_text(str(item.get("content", "")))}
            for item in (messages or [])
        ]
        resolved_query = normalize_text(query or self._last_user_message(normalized_messages))
        search_filters = self._coerce_filters(
            filters=filters,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=max(limit * 3, 12),
        )
        if not self._brain_enabled():
            state = self.repository.load_state()
            return self._build_context_artifacts(
                state=state,
                query=resolved_query,
                search_filters=search_filters,
                limit=limit,
                max_tokens=max_tokens,
                include_debug=include_debug,
                include_trace=self._should_include_trace(include_trace),
                context_policy=context_policy,
            )["memory_pack"]

        def operation(state: EngineState) -> tuple[ContextPack, list[dict[str, Any]]]:
            dormancy_transitions = self._prepare_brain_state(state)
            if dormancy_transitions:
                self._rebuild_projections(state)
            artifacts = self._build_context_artifacts(
                state=state,
                query=resolved_query,
                search_filters=search_filters,
                limit=limit,
                max_tokens=max_tokens,
                include_debug=include_debug,
                include_trace=self._should_include_trace(include_trace),
                context_policy=context_policy,
            )
            retrieval_events = self._touch_retrieval_hits(state=state, ranked=artifacts["selected_ranked"])
            if retrieval_events:
                self._rebuild_projections(state)
            memory_pack = artifacts["memory_pack"]
            self._refresh_context_pack_items(state=state, memory_pack=memory_pack)
            if self._should_include_trace(include_trace):
                memory_pack.trace["consolidation"]["dormancy_transitions"] = list(dormancy_transitions)
            return memory_pack, dormancy_transitions + retrieval_events

        return self.repository.transaction("build_context", None, operation)

    def get(self, memory_id: str) -> dict[str, Any]:
        state = self.repository.load_state()
        record = state.memories.get(memory_id)
        if record is None:
            raise KeyError(f"Unknown memory_id: {memory_id}")
        return self._record_payload(state, record)

    def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        filters: SearchFilters | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self.repository.load_state()
        search_filters = self._coerce_filters(
            filters=filters,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        records = [
            self._record_payload(state, record)
            for record in state.memories.values()
            if self._record_matches_filters(record, search_filters, include_scope=True)
        ]
        records.sort(key=lambda item: item["memory"]["updated_at"], reverse=True)
        return {
            "filters": search_filters.to_dict(),
            "results": records,
            "projection_status": ProjectionStatus.from_dict(state.projections.get("status")).to_dict(),
        }

    def update(
        self,
        *,
        memory_id: str,
        data: str | dict[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        payload = {"value": data} if isinstance(data, str) else dict(data or {})

        def operation(state: EngineState) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            record = state.memories.get(memory_id)
            if record is None:
                raise KeyError(f"Unknown memory_id: {memory_id}")
            candidate = MemoryCandidate(
                kind=record.kind,
                key=record.key,
                value=normalize_text(str(payload.get("value", record.value))),
                summary=normalize_text(str(payload.get("summary", payload.get("value", record.summary)))),
                confidence=float(payload.get("confidence", record.confidence or 1.0)),
                state=str(payload.get("state", MemoryState.ACTIVE.value)),
                metadata={**record.metadata, **dict(payload.get("metadata") or {})},
                entity_names=list(payload.get("entity_names") or record.entity_names),
                tags=list(payload.get("tags") or record.tags),
                layer=str(payload.get("layer", record.layer)),
                salience=float(payload.get("salience", record.salience)),
                source_type=str(payload.get("source_type", record.source_type)),
            )
            outcome = self._apply_candidate(
                state=state,
                candidate=candidate,
                scope=record.scope,
                evidence_ids=list(payload.get("evidence_ids") or []),
                explicit_record_id=memory_id,
                explicit_action=MemoryAction.UPDATE.value,
            )
            self._rebuild_projections(state)
            result = self._record_payload(state, state.memories[memory_id])
            result["action"] = outcome["action"]
            result["projection_status"] = ProjectionStatus.from_dict(state.projections.get("status")).to_dict()
            return result, outcome["events"]

        return self.repository.transaction("update", idempotency_key, operation)

    def delete(self, *, memory_id: str, idempotency_key: str | None = None) -> dict[str, Any]:
        def operation(state: EngineState) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            record = state.memories.get(memory_id)
            if record is None:
                raise KeyError(f"Unknown memory_id: {memory_id}")
            candidate = MemoryCandidate(
                kind=record.kind,
                key=record.key,
                value="",
                summary=f"Deleted {record.kind}: {record.key}",
                confidence=1.0,
                metadata=dict(record.metadata),
                entity_names=list(record.entity_names),
                tags=list(record.tags),
                state=MemoryState.DELETED.value,
                layer=record.layer,
                salience=record.salience,
                source_type=record.source_type,
            )
            outcome = self._apply_candidate(
                state=state,
                candidate=candidate,
                scope=record.scope,
                evidence_ids=[],
                explicit_record_id=memory_id,
                explicit_action=MemoryAction.DELETE.value,
            )
            self._rebuild_projections(state)
            result = {
                "status": "deleted",
                "memory_id": memory_id,
                "projection_status": ProjectionStatus.from_dict(state.projections.get("status")).to_dict(),
            }
            return result, outcome["events"]

        return self.repository.transaction("delete", idempotency_key, operation)

    def delete_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        filters: SearchFilters | dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        search_filters = self._coerce_filters(
            filters=filters,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        if search_filters.scope is None or search_filters.scope.is_empty():
            if not any((search_filters.kinds, search_filters.tags, search_filters.metadata, search_filters.layers, search_filters.source_types)):
                raise ValueError("delete_all requires at least one scope or filter")

        def operation(state: EngineState) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            deleted: list[str] = []
            events: list[dict[str, Any]] = []
            for record in list(state.memories.values()):
                if not self._record_matches_filters(record, search_filters, include_scope=True):
                    continue
                outcome = self._apply_candidate(
                    state=state,
                    candidate=MemoryCandidate(
                        kind=record.kind,
                        key=record.key,
                        value="",
                        summary=f"Deleted {record.kind}: {record.key}",
                        confidence=1.0,
                        metadata=dict(record.metadata),
                        entity_names=list(record.entity_names),
                        tags=list(record.tags),
                        state=MemoryState.DELETED.value,
                        layer=record.layer,
                        salience=record.salience,
                        source_type=record.source_type,
                    ),
                    scope=record.scope,
                    evidence_ids=[],
                    explicit_record_id=record.record_id,
                    explicit_action=MemoryAction.DELETE.value,
                )
                if outcome["action"] != MemoryAction.NONE.value:
                    deleted.append(record.record_id)
                    events.extend(outcome["events"])

            self._rebuild_projections(state)
            return {
                "status": "deleted",
                "deleted_ids": deleted,
                "projection_status": ProjectionStatus.from_dict(state.projections.get("status")).to_dict(),
            }, events

        return self.repository.transaction("delete_all", idempotency_key, operation)

    def history(self, memory_id: str) -> dict[str, Any]:
        state = self.repository.load_state()
        record = state.memories.get(memory_id)
        if record is None:
            raise KeyError(f"Unknown memory_id: {memory_id}")
        versions = [
            state.versions[version_id].to_dict()
            for version_id in record.version_ids
            if version_id in state.versions
        ]
        versions.sort(key=lambda item: item["created_at"])
        return {"memory_id": memory_id, "history": versions}

    def export(self) -> dict[str, Any]:
        return self.repository.export_state()

    def import_snapshot(
        self,
        payload: dict[str, Any],
        *,
        mode: str = "merge",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        incoming = EngineState.from_dict(payload)
        if mode == "replace":
            self._rebuild_projections(incoming)
            return self.repository.replace_state(incoming, operation_name="import", idempotency_key=idempotency_key)

        def operation(state: EngineState) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            for evidence_id, evidence in incoming.evidence.items():
                state.evidence[evidence_id] = evidence
            for version_id, version in incoming.versions.items():
                state.versions[version_id] = version
            for record_id, record in incoming.memories.items():
                state.memories[record_id] = record
            for edge_id, relation in incoming.relations.items():
                state.relations[edge_id] = relation
            self._rebuild_projections(state)
            return {
                "status": "imported",
                "mode": mode,
                "memory_count": len(incoming.memories),
                "evidence_count": len(incoming.evidence),
                "schema_version": state.schema_version,
                "projection_status": ProjectionStatus.from_dict(state.projections.get("status")).to_dict(),
            }, [{"type": "snapshot_merged", "memory_count": len(incoming.memories)}]

        return self.repository.transaction("import", idempotency_key, operation)

    def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        model: str | None = None,
        limit: int = 5,
        store: bool = False,
        idempotency_key: str | None = None,
        tool_events: list[dict[str, Any]] | None = None,
        memory_strategy: str = "v2",
        include_memory_pack: bool = False,
        include_trace: bool = False,
        context_policy: str = "compact",
    ) -> dict[str, Any]:
        normalized_messages = [
            {"role": item.get("role", "user"), "content": normalize_text(str(item.get("content", "")))}
            for item in messages
        ]
        user_query = self._last_user_message(normalized_messages)
        memory_results: dict[str, Any]

        memory_pack: ContextPack | None = None
        grounding: str
        if memory_strategy == "v3":
            search_filters = self._coerce_filters(
                filters=None,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                limit=max(limit * 3, 12),
            )
            if self._brain_enabled():
                def operation(state: EngineState) -> tuple[dict[str, Any], list[dict[str, Any]]]:
                    dormancy_transitions = self._prepare_brain_state(state)
                    if dormancy_transitions:
                        self._rebuild_projections(state)
                    artifacts = self._build_context_artifacts(
                        state=state,
                        query=user_query,
                        search_filters=search_filters,
                        limit=limit,
                        max_tokens=480,
                        include_debug=include_memory_pack,
                        include_trace=self._should_include_trace(include_trace),
                        context_policy=context_policy,
                    )
                    retrieval_events = self._touch_retrieval_hits(state=state, ranked=artifacts["selected_ranked"])
                    if retrieval_events:
                        self._rebuild_projections(state)
                    response_filters = SearchFilters.from_dict(search_filters.to_dict())
                    response_filters.limit = limit
                    memory_pack_local = artifacts["memory_pack"]
                    self._refresh_context_pack_items(state=state, memory_pack=memory_pack_local)
                    if self._should_include_trace(include_trace):
                        memory_pack_local.trace["consolidation"]["dormancy_transitions"] = list(dormancy_transitions)
                    return {
                        "memory_pack": memory_pack_local,
                        "grounding": artifacts["grounding"],
                        "memory_results": self._build_search_payload(
                            state=state,
                            query=user_query,
                            search_filters=response_filters,
                            ranked=artifacts["ranked"],
                            projection_status=ProjectionStatus.from_dict(state.projections.get("status")),
                        ),
                    }, dormancy_transitions + retrieval_events

                recall = self.repository.transaction("chat_context", None, operation)
                memory_pack = recall["memory_pack"]
                grounding = recall["grounding"]
                memory_results = recall["memory_results"]
            else:
                state = self.repository.load_state()
                artifacts = self._build_context_artifacts(
                    state=state,
                    query=user_query,
                    search_filters=search_filters,
                    limit=limit,
                    max_tokens=480,
                    include_debug=include_memory_pack,
                    include_trace=self._should_include_trace(include_trace),
                    context_policy=context_policy,
                )
                memory_pack = artifacts["memory_pack"]
                grounding = artifacts["grounding"]
                response_filters = SearchFilters.from_dict(search_filters.to_dict())
                response_filters.limit = limit
                memory_results = self._build_search_payload(
                    state=state,
                    query=user_query,
                    search_filters=response_filters,
                    ranked=artifacts["ranked"],
                    projection_status=artifacts["projection_status"],
                )
        else:
            memory_results = self.search(
                query=user_query,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                limit=limit,
                include_trace=self._should_include_trace(include_trace),
            )
            memory_lines = [
                f"- {entry['memory']['summary']} (memory_id={entry['memory']['record_id']})"
                for entry in memory_results["results"]
            ]
            grounding = flatten_text_parts(["Relevant memories:"] + memory_lines) if memory_lines else "Relevant memories: none"

        tool_lines = self._format_tool_events(tool_events or [])
        if tool_lines:
            grounding = flatten_text_parts([grounding, "Current tool events:", *tool_lines])

        if memory_strategy == "v3":
            system_prompt = (
                "You are a reliable assistant. Use the memory pack when relevant, prefer newer and confirmed facts, "
                "and avoid contradicted or stale memories."
            )
        else:
            system_prompt = "You are a reliable assistant. Ground your answer in the provided memories when relevant."
        model_messages = [{"role": "system", "content": f"{system_prompt}\n{grounding}"}] + normalized_messages

        if self.chat_model is not None:
            content = str(self.chat_model.invoke(model_messages))
        else:
            if memory_strategy == "v3" and memory_pack is not None:
                if self._context_pack_has_content(memory_pack):
                    content = f"I found this memory context:\n{grounding}\n\nBased on it, here is a grounded response to: {user_query}"
                else:
                    content = f"I do not have matching memory context yet. Here is a direct response to: {user_query}"
            else:
                memory_lines = [
                    f"- {entry['memory']['summary']} (memory_id={entry['memory']['record_id']})"
                    for entry in memory_results["results"]
                ]
                if memory_lines:
                    content = f"I found these relevant memories:\n{chr(10).join(memory_lines)}\n\nBased on them, here is a grounded response to: {user_query}"
                else:
                    content = f"I do not have matching memories yet. Here is a direct response to: {user_query}"

        assistant_message = {"role": "assistant", "content": content}
        if store:
            store_key = f"{idempotency_key}:store" if idempotency_key else None
            if memory_strategy == "v3" or tool_events:
                self.capture(
                    messages=normalized_messages + [assistant_message],
                    events=tool_events,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    idempotency_key=store_key,
                )
            else:
                self.add(
                    messages=normalized_messages + [assistant_message],
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    idempotency_key=store_key,
                )

        response: dict[str, Any] = {
            "id": generate_id("chatcmpl"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model or getattr(self.chat_model, "model_name", "memoripy-v3"),
            "choices": [
                {
                    "index": 0,
                    "message": assistant_message,
                    "finish_reason": "stop",
                }
            ],
            "memory": memory_results,
        }
        if include_memory_pack and memory_pack is not None:
            response["memory_pack"] = memory_pack.to_dict()
        if self._should_include_trace(include_trace):
            if memory_strategy == "v3" and memory_pack is not None:
                response["trace"] = dict(memory_pack.trace)
                response["trace"]["grounding"] = grounding
                response["trace"]["memory_strategy"] = memory_strategy
            else:
                response["trace"] = dict(memory_results.get("trace") or {})
                response["trace"]["grounding"] = grounding
                response["trace"]["memory_strategy"] = memory_strategy
        return response

    def _ingest(
        self,
        *,
        operation_name: str,
        idempotency_key: str | None,
        scope: MemoryScope,
        ingestion_items: list[IngestionItem],
        strategy: str,
    ) -> dict[str, Any]:
        def operation(state: EngineState) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            events: list[dict[str, Any]] = []
            evidence_items = [self._build_evidence_item(item, scope) for item in ingestion_items]
            for evidence in evidence_items:
                if evidence.evidence_id not in state.evidence:
                    state.evidence[evidence.evidence_id] = evidence
                    events.append(
                        {
                            "type": "evidence_added",
                            "evidence_id": evidence.evidence_id,
                            "scope": evidence.scope.to_dict(),
                            "modality": evidence.modality,
                            "event_type": evidence.event_type,
                            "source_type": evidence.source_type,
                        }
                    )

            created: list[dict[str, Any]] = []
            updated: list[dict[str, Any]] = []
            pending: list[dict[str, Any]] = []
            noop: list[dict[str, Any]] = []
            semantic_memory_ids: list[str] = []
            episodic_memory_ids: list[str] = []

            for evidence in evidence_items:
                candidates = self._candidates_for_evidence(evidence=evidence, strategy=strategy)
                for candidate in candidates:
                    outcome = self._apply_candidate(
                        state=state,
                        candidate=candidate,
                        scope=scope,
                        evidence_ids=[evidence.evidence_id],
                    )
                    events.extend(outcome["events"])
                    payload = outcome["payload"]
                    record = state.memories.get(payload.get("record_id", ""))
                    if record is not None:
                        if record.layer == MemoryLayer.EPISODIC.value:
                            episodic_memory_ids.append(record.record_id)
                        else:
                            semantic_memory_ids.append(record.record_id)

                    if outcome["state"] == MemoryState.PENDING.value and outcome["action"] != MemoryAction.NONE.value:
                        pending.append(payload)
                    elif outcome["action"] == MemoryAction.NONE.value:
                        noop.append(payload)
                    elif outcome["action"] == MemoryAction.ADD.value:
                        created.append(payload)
                    else:
                        updated.append(payload)

            self._rebuild_projections(state)
            projection_status = ProjectionStatus.from_dict(state.projections.get("status"))
            memory_ids = [
                item["record_id"]
                for item in created + updated + pending
                if item.get("record_id")
            ]
            result = {
                "id": generate_id("op"),
                "strategy": strategy,
                "scope": scope.to_dict(),
                "evidence_ids": [item.evidence_id for item in evidence_items],
                "created": created,
                "updated": updated,
                "pending": pending,
                "noop": noop,
                "memory_ids": list(dict.fromkeys(memory_ids)),
                "semantic_memory_ids": list(dict.fromkeys(semantic_memory_ids)),
                "episodic_memory_ids": list(dict.fromkeys(episodic_memory_ids)),
                "projection_status": projection_status.to_dict(),
            }
            return result, events

        return self.repository.transaction(operation_name, idempotency_key, operation)

    def _brain_enabled(self) -> bool:
        return normalize_text(getattr(self.brain, "mode", "classic")).lower() == "attention_fast"

    def _activation_projection(self, state: EngineState) -> dict[str, dict[str, Any]]:
        activation = state.projections.setdefault("activation", {})
        if not isinstance(activation, dict):
            activation = {}
            state.projections["activation"] = activation
        return activation

    def _activation_entry(self, state: EngineState, record: MemoryRecord) -> dict[str, Any]:
        activation = self._activation_projection(state)
        entry = activation.setdefault(record.record_id, {})
        if not entry.get("last_activated_at"):
            entry["last_activated_at"] = (
                record.last_accessed_at or record.last_confirmed_at or record.updated_at or record.created_at
            )
        entry["rehearsal_count"] = int(entry.get("rehearsal_count", max(record.confirmation_count, 0)))
        entry["retrieval_count"] = int(entry.get("retrieval_count", max(record.access_count, 0)))
        entry["last_consolidated_at"] = entry.get("last_consolidated_at")
        entry["activation_score"] = float(entry.get("activation_score", 0.0))
        return entry

    def _parse_timestamp(self, timestamp: str | None) -> datetime | None:
        if not timestamp:
            return None
        try:
            return datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except ValueError:
            return None

    def _latest_timestamp(self, *timestamps: str | None) -> str | None:
        candidates = [
            (parsed, original)
            for original in timestamps
            for parsed in [self._parse_timestamp(original)]
            if parsed is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _hours_since(self, timestamp: str | None) -> float:
        parsed = self._parse_timestamp(timestamp)
        if parsed is None:
            return float("inf")
        return max((datetime.now(timezone.utc) - parsed).total_seconds() / 3600.0, 0.0)

    def _normalized_counter(self, value: int) -> float:
        if value <= 0:
            return 0.0
        return min(math.log1p(value) / math.log(6.0), 1.0)

    def _activation_decay(self, timestamp: str | None) -> float:
        if not timestamp:
            return 0.0
        half_life = max(float(self.brain.attention_decay_half_life_hours), 0.001)
        age_hours = self._hours_since(timestamp)
        if math.isinf(age_hours):
            return 0.0
        return max(min(0.5 ** (age_hours / half_life), 1.0), 0.0)

    def _activation_score(
        self,
        *,
        record: MemoryRecord,
        entry: dict[str, Any],
    ) -> float:
        activation = (
            (0.45 * self._activation_decay(entry.get("last_activated_at")))
            + (0.20 * self._normalized_counter(record.access_count))
            + (0.15 * self._normalized_counter(record.confirmation_count))
            + (0.20 * min(max(record.salience, 0.0), 1.0))
        )
        return max(0.0, min(activation, 1.0))

    def _refresh_activation_projection(self, state: EngineState) -> None:
        activation = self._activation_projection(state)
        valid_record_ids = set(state.memories.keys())
        for record_id in list(activation.keys()):
            if record_id not in valid_record_ids:
                activation.pop(record_id, None)
        for record in state.memories.values():
            if record.state in (MemoryState.SUPERSEDED.value, MemoryState.DELETED.value):
                activation.pop(record.record_id, None)
                continue
            entry = self._activation_entry(state, record)
            entry["activation_score"] = self._activation_score(record=record, entry=entry)

    def _touch_memory_rehearsal(
        self,
        *,
        state: EngineState,
        record_ids: list[str],
        activated_at: str | None = None,
        last_consolidated_at: str | None = None,
    ) -> None:
        now = activated_at or utc_now()
        for record_id in list(dict.fromkeys(record_ids)):
            record = state.memories.get(record_id)
            if record is None or record.state in (MemoryState.SUPERSEDED.value, MemoryState.DELETED.value):
                continue
            entry = self._activation_entry(state, record)
            entry["last_activated_at"] = now
            entry["rehearsal_count"] = int(entry.get("rehearsal_count", 0)) + 1
            if last_consolidated_at is not None:
                entry["last_consolidated_at"] = last_consolidated_at
            entry["activation_score"] = self._activation_score(record=record, entry=entry)

    def _apply_dormancy_transitions(self, state: EngineState) -> list[dict[str, Any]]:
        if not self._brain_enabled():
            return []
        transitions: list[dict[str, Any]] = []
        threshold = float(self.brain.dormancy_threshold)
        window_hours = float(self.brain.consolidation_window_hours)
        for record in state.memories.values():
            if record.state != MemoryState.ACTIVE.value:
                continue
            if record.layer not in (MemoryLayer.SEMANTIC.value, MemoryLayer.EPISODIC.value):
                continue
            entry = self._activation_entry(state, record)
            entry["activation_score"] = self._activation_score(record=record, entry=entry)
            last_touch = self._latest_timestamp(
                entry.get("last_activated_at"),
                record.last_accessed_at,
                record.last_confirmed_at,
                record.updated_at,
            )
            if entry["activation_score"] >= threshold:
                continue
            if self._hours_since(last_touch) < window_hours:
                continue
            record.state = MemoryState.DORMANT.value
            transitions.append(
                {
                    "type": "memory_dormant",
                    "memory_id": record.record_id,
                    "state": record.state,
                    "activation_score": entry["activation_score"],
                }
            )
        return transitions

    def _canonical_match_counts(
        self,
        ranked: list[tuple[float, MemoryRecord, dict[str, float]]],
    ) -> dict[tuple[str, str, str], int]:
        counts: dict[tuple[str, str, str], int] = defaultdict(int)
        for _, record, _ in ranked:
            counts[(record.kind, record.key.lower(), normalize_text(record.value).lower())] += 1
        return counts

    def _should_reactivate_dormant(
        self,
        *,
        record: MemoryRecord,
        breakdown: dict[str, float],
        canonical_match_counts: dict[tuple[str, str, str], int],
    ) -> bool:
        if record.state != MemoryState.DORMANT.value:
            return False
        exact_cue = float(breakdown.get("exact_cue", 0.0))
        canonical_key = (record.kind, record.key.lower(), normalize_text(record.value).lower())
        return exact_cue > 0.85 or canonical_match_counts.get(canonical_key, 0) <= 1

    def _touch_retrieval_hits(
        self,
        *,
        state: EngineState,
        ranked: list[tuple[float, MemoryRecord, dict[str, float]]],
    ) -> list[dict[str, Any]]:
        if not self._brain_enabled():
            return []
        now = utc_now()
        events: list[dict[str, Any]] = []
        canonical_match_counts = self._canonical_match_counts(ranked)
        for _, record, breakdown in ranked:
            entry = self._activation_entry(state, record)
            record.access_count += 1
            record.last_accessed_at = now
            entry["last_activated_at"] = now
            entry["retrieval_count"] = int(entry.get("retrieval_count", 0)) + 1
            if self._should_reactivate_dormant(
                record=record,
                breakdown=breakdown,
                canonical_match_counts=canonical_match_counts,
            ):
                record.state = MemoryState.ACTIVE.value
                events.append(
                    {
                        "type": "memory_reactivated",
                        "memory_id": record.record_id,
                        "state": record.state,
                        "activation_score": entry["activation_score"],
                    }
                )
            entry["activation_score"] = self._activation_score(record=record, entry=entry)
        return events

    def _prepare_brain_state(self, state: EngineState) -> list[dict[str, Any]]:
        self._refresh_activation_projection(state)
        return self._apply_dormancy_transitions(state)

    def _coerce_filters(
        self,
        *,
        filters: SearchFilters | dict[str, Any] | None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> SearchFilters:
        search_filters = SearchFilters.from_dict(filters if isinstance(filters, dict) else None)
        if isinstance(filters, SearchFilters):
            search_filters = filters
        if search_filters.scope is None:
            search_filters.scope = MemoryScope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        if limit:
            search_filters.limit = limit
        return search_filters

    def _normalize_ingestion_items(
        self,
        *,
        messages: list[dict[str, Any]] | None,
        items: list[IngestionItem | dict[str, Any]] | None,
        text: str | None,
        modality: str,
        metadata: dict[str, Any] | None,
    ) -> list[IngestionItem]:
        normalized: list[IngestionItem] = []
        if messages:
            for message in messages:
                item_metadata = dict(message.get("metadata") or {})
                normalized.append(
                    IngestionItem(
                        content=str(message.get("content", "")),
                        modality="text",
                        role=message.get("role"),
                        metadata=item_metadata,
                        event_type=EventType.MESSAGE.value,
                        name=message.get("role"),
                        attributes=item_metadata,
                        occurred_at=message.get("timestamp") or item_metadata.get("timestamp"),
                        source_type=EventType.MESSAGE.value,
                    )
                )
        if items:
            for item in items:
                normalized.append(item if isinstance(item, IngestionItem) else IngestionItem.from_dict(item))
        if text is not None:
            normalized.append(
                IngestionItem(
                    content=text,
                    modality=modality,
                    metadata=dict(metadata or {}),
                    event_type=EventType.INGESTION.value,
                    attributes=dict(metadata or {}),
                    source_type=EventType.INGESTION.value,
                )
            )
        if not normalized:
            raise ValueError("add() requires messages, items, or text")
        return self._apply_asset_processor(normalized)

    def _normalize_capture_items(
        self,
        *,
        messages: list[dict[str, Any]] | None,
        events: list[dict[str, Any] | IngestionItem] | None,
        items: list[IngestionItem | dict[str, Any]] | None,
    ) -> list[IngestionItem]:
        normalized: list[IngestionItem] = []
        if items:
            for item in items:
                normalized.append(item if isinstance(item, IngestionItem) else IngestionItem.from_dict(item))
        if messages:
            for message in messages:
                item_metadata = dict(message.get("metadata") or {})
                normalized.append(
                    IngestionItem(
                        content=str(message.get("content", "")),
                        modality="text",
                        role=message.get("role"),
                        metadata=item_metadata,
                        event_type=EventType.MESSAGE.value,
                        name=message.get("role"),
                        attributes=item_metadata,
                        occurred_at=message.get("timestamp") or item_metadata.get("timestamp"),
                        source_type=EventType.MESSAGE.value,
                    )
                )
        if events:
            for raw in events:
                if isinstance(raw, IngestionItem):
                    normalized.append(raw)
                    continue
                event = dict(raw or {})
                event_type = str(event.get("event_type", EventType.TOOL_RESULT.value))
                attributes = dict(event.get("attributes") or {})
                metadata = dict(event.get("metadata") or {})
                normalized.append(
                    IngestionItem(
                        content=self._event_text(event),
                        modality=str(event.get("modality", "text")),
                        role=event.get("role"),
                        metadata=metadata,
                        asset_ref=event.get("asset_ref"),
                        event_type=event_type,
                        name=event.get("name"),
                        attributes=attributes,
                        occurred_at=event.get("occurred_at") or event.get("timestamp") or metadata.get("timestamp"),
                        source_type=str(event.get("source_type", event_type)),
                    )
                    )
        if not normalized:
            raise ValueError("capture() requires messages, events, or items")
        return self._apply_asset_processor(normalized)

    def _apply_asset_processor(self, items: list[IngestionItem]) -> list[IngestionItem]:
        if self.asset_processor is None:
            return items
        processed: list[IngestionItem] = []
        for item in items:
            outputs = self.asset_processor.process(item)
            if not outputs:
                processed.append(item)
                continue
            for output in outputs:
                processed.append(output if isinstance(output, IngestionItem) else IngestionItem.from_dict(output))
        return processed

    def _event_text(self, payload: dict[str, Any]) -> str:
        attributes = dict(payload.get("attributes") or {})
        metadata = dict(payload.get("metadata") or {})
        preferred_keys = (
            "content",
            "text",
            "summary",
            "result",
            "output",
            "value",
            "message",
            "query",
            "input",
        )
        for key in preferred_keys:
            value = payload.get(key)
            if value is None:
                value = attributes.get(key)
            if value is None:
                value = metadata.get(key)
            if value:
                return normalize_text(str(value))

        parts: list[str] = []
        if payload.get("name"):
            parts.append(str(payload["name"]))
        for key in ("tool", "tool_name", "action", "status"):
            if payload.get(key):
                parts.append(f"{key}: {payload[key]}")
        for key, value in attributes.items():
            if value in (None, "", [], {}):
                continue
            parts.append(f"{key}: {value}")
        return normalize_text(flatten_text_parts(parts))

    def _build_evidence_item(self, item: IngestionItem, scope: MemoryScope) -> EvidenceItem:
        metadata = dict(item.metadata or {})
        attributes = dict(item.attributes or {})
        derived_text = normalize_text(
            item.content
            or str(
                metadata.get("text")
                or metadata.get("ocr_text")
                or metadata.get("transcript")
                or metadata.get("caption")
                or attributes.get("text")
                or attributes.get("result")
                or attributes.get("output")
                or attributes.get("summary")
                or ""
            )
        )
        if not derived_text and attributes:
            derived_text = normalize_text(
                flatten_text_parts(f"{key}: {value}" for key, value in attributes.items() if value not in (None, "", [], {}))
            )
        event_type = item.event_type or EventType.INGESTION.value
        source_type = item.source_type or event_type
        content_hash = stable_hash(
            item.modality,
            item.role,
            derived_text,
            item.asset_ref,
            metadata,
            event_type,
            item.name,
            attributes,
            scope.to_dict(),
        )
        return EvidenceItem(
            evidence_id=f"evidence_{content_hash[:24]}",
            content_hash=content_hash,
            modality=item.modality,
            text=derived_text,
            role=item.role,
            metadata=metadata,
            asset_ref=item.asset_ref,
            scope=scope,
            created_at=utc_now(),
            event_type=event_type,
            name=item.name,
            attributes=attributes,
            occurred_at=item.occurred_at or metadata.get("timestamp"),
            source_type=source_type,
        )

    def _canonical_lookup_key(self, scope: MemoryScope, kind: str, key: str) -> str:
        return stable_hash(scope.to_dict(), kind, key.lower())

    def _existing_record_id(
        self,
        *,
        state: EngineState,
        scope: MemoryScope,
        candidate: MemoryCandidate,
        lookup_key: str,
    ) -> str | None:
        record_id = state.lookup.get(lookup_key)
        if record_id:
            return record_id
        for record in state.memories.values():
            if record.scope.to_dict() != scope.to_dict():
                continue
            if record.kind != candidate.kind or record.key != candidate.key:
                continue
            if record.layer != candidate.layer:
                continue
            if record.state == MemoryState.DELETED.value:
                continue
            return record.record_id
        return None

    def _candidates_for_evidence(self, *, evidence: EvidenceItem, strategy: str) -> list[MemoryCandidate]:
        if strategy != "v3":
            return [self._finalize_candidate(candidate, evidence) for candidate in self.extractor.extract(evidence)]

        semantic_candidates: list[MemoryCandidate]
        if hasattr(self.extractor, "extract_semantic"):
            semantic_candidates = list(self.extractor.extract_semantic(evidence))
        else:
            semantic_candidates = [
                candidate
                for candidate in self.extractor.extract(evidence)
                if candidate.layer != MemoryLayer.EPISODIC.value
            ]

        candidates: list[MemoryCandidate] = []
        for candidate in semantic_candidates:
            finalized = self._finalize_candidate(candidate, evidence)
            if self._should_promote_semantic(finalized, evidence):
                candidates.append(finalized)

        episode = self._build_episodic_candidate(evidence, semantic_candidates)
        if episode is not None:
            candidates.append(episode)
        return candidates

    def _finalize_candidate(self, candidate: MemoryCandidate, evidence: EvidenceItem) -> MemoryCandidate:
        salience = max(candidate.salience, self._score_salience(evidence=evidence, candidate=candidate))
        metadata = {
            **dict(candidate.metadata),
            "event_type": evidence.event_type,
            "source_type": evidence.source_type,
        }
        return MemoryCandidate(
            kind=candidate.kind,
            key=candidate.key,
            value=normalize_text(candidate.value),
            summary=normalize_text(candidate.summary),
            confidence=candidate.confidence,
            state=candidate.state,
            metadata=metadata,
            entity_names=list(dict.fromkeys(candidate.entity_names)),
            tags=list(dict.fromkeys(candidate.tags)),
            layer=candidate.layer,
            salience=salience,
            source_type=evidence.source_type,
        )

    def _should_promote_semantic(self, candidate: MemoryCandidate, evidence: EvidenceItem) -> bool:
        threshold = self.pipeline.semantic_promotion_threshold
        if candidate.layer != MemoryLayer.SEMANTIC.value:
            return False
        if candidate.kind in (
            MemoryKind.PROFILE_ATTRIBUTE.value,
            MemoryKind.PREFERENCE.value,
            MemoryKind.RELATION.value,
        ) and candidate.confidence >= threshold:
            return True
        if candidate.salience >= 0.78:
            return True
        if evidence.event_type == EventType.TOOL_RESULT.value and candidate.confidence >= 0.65:
            return True
        return False

    def _build_episodic_candidate(
        self,
        evidence: EvidenceItem,
        semantic_candidates: list[MemoryCandidate],
    ) -> MemoryCandidate | None:
        if not hasattr(self.extractor, "build_episode_candidate"):
            return None
        episode = self.extractor.build_episode_candidate(evidence)
        if episode is None:
            return None
        finalized = self._finalize_candidate(episode, evidence)
        if semantic_candidates or evidence.event_type != EventType.MESSAGE.value:
            if finalized.salience >= 0.34:
                return finalized
            return None
        if finalized.salience >= 0.44:
            return finalized
        return None

    def _score_salience(self, *, evidence: EvidenceItem, candidate: MemoryCandidate | None = None) -> float:
        text = evidence.text
        token_count = len(tokenize(text))
        score = 0.15 + min(token_count / 50.0, 0.25)
        if evidence.role == "user":
            score += 0.08
        if evidence.event_type == EventType.TOOL_RESULT.value:
            score += 0.3
        elif evidence.event_type == EventType.TOOL_CALL.value:
            score += 0.18
        elif evidence.event_type == EventType.ASSISTANT_ACTION.value:
            score += 0.22
        if candidate is not None:
            if candidate.kind == MemoryKind.PROFILE_ATTRIBUTE.value:
                score += 0.28
            elif candidate.kind == MemoryKind.PREFERENCE.value:
                score += 0.2
            elif candidate.kind == MemoryKind.RELATION.value:
                score += 0.2
            if candidate.layer == MemoryLayer.EPISODIC.value:
                score += 0.05
        if any(phrase in text.lower() for phrase in ("my name is", "i live in", "my favorite", "i work at", "i work for")):
            score += 0.18
        return max(0.0, min(score, 1.0))

    def _apply_candidate(
        self,
        *,
        state: EngineState,
        candidate: MemoryCandidate,
        scope: MemoryScope,
        evidence_ids: list[str],
        explicit_record_id: str | None = None,
        explicit_action: str | None = None,
    ) -> dict[str, Any]:
        lookup_key = self._canonical_lookup_key(scope, candidate.kind, candidate.key)
        existing_record = None
        if explicit_record_id and explicit_record_id in state.memories:
            existing_record = state.memories[explicit_record_id]
        else:
            matched_record_id = self._existing_record_id(
                state=state,
                scope=scope,
                candidate=candidate,
                lookup_key=lookup_key,
            )
            if matched_record_id:
                existing_record = state.memories.get(matched_record_id)
        decision = self.reconciler.reconcile(
            state=state,
            scope=scope,
            candidate=candidate,
            existing_record=existing_record,
            explicit_action=explicit_action,
        )
        record_id = explicit_record_id or decision.matched_record_id
        now = utc_now()
        events: list[dict[str, Any]] = []
        desired_state = decision.state
        reasoning_trace = list(dict.fromkeys(decision.reasoning_trace or ["memory candidate reconciled"]))
        candidate_metadata = {**dict(candidate.metadata), **dict(decision.metadata)}

        if record_id and record_id in state.memories:
            record = state.memories[record_id]
            current_version = state.versions.get(record.current_version_id) if record.current_version_id else None
            same_value = normalize_text(record.value).lower() == normalize_text(candidate.value).lower()
            if candidate.kind == MemoryKind.PREFERENCE.value:
                record_sentiment = normalize_text(str(record.metadata.get("sentiment", ""))).lower()
                candidate_sentiment = normalize_text(str(candidate.metadata.get("sentiment", ""))).lower()
                if record_sentiment and candidate_sentiment and record_sentiment != candidate_sentiment:
                    same_value = False

            if decision.action == MemoryAction.DELETE.value:
                if record.state == MemoryState.DELETED.value:
                    return {
                        "action": MemoryAction.NONE.value,
                        "state": record.state,
                        "payload": {"record_id": record.record_id, "version_id": record.current_version_id, "summary": record.summary},
                        "events": [],
                    }
                version_id = generate_id("version")
                if current_version is not None:
                    current_version.state = MemoryState.SUPERSEDED.value
                    current_version.contradicted_by = list(dict.fromkeys(current_version.contradicted_by + [version_id]))
                version = MemoryVersion(
                    version_id=version_id,
                    record_id=record.record_id,
                    action=MemoryAction.DELETE.value,
                    state=MemoryState.DELETED.value,
                    value="",
                    summary=candidate.summary,
                    created_at=now,
                    confidence=1.0,
                    evidence_ids=evidence_ids,
                    metadata=candidate_metadata,
                    reasoning_trace=reasoning_trace,
                    supersedes_version_id=record.current_version_id,
                    salience=candidate.salience,
                    source_type=candidate.source_type,
                    layer=record.layer,
                    citation_evidence_ids=list(evidence_ids),
                    contradicted_by=[],
                )
                state.versions[version.version_id] = version
                record.current_version_id = version.version_id
                record.version_ids.append(version.version_id)
                record.state = MemoryState.DELETED.value
                record.summary = candidate.summary
                record.value = ""
                record.updated_at = now
                record.last_confirmed_at = now
                record.evidence_ids = list(dict.fromkeys(record.evidence_ids + evidence_ids))
                record.citation_evidence_ids = list(dict.fromkeys(record.citation_evidence_ids + evidence_ids))
                record.metadata = {**record.metadata, **candidate_metadata}
                record.search_text = self._search_text(record)
                record.contradicted_by = []
                events.append({"type": "memory_deleted", "record_id": record.record_id, "version_id": version.version_id})
                return {
                    "action": MemoryAction.DELETE.value,
                    "state": record.state,
                    "payload": {"record_id": record.record_id, "version_id": version.version_id, "summary": record.summary},
                    "events": events,
                }

            if same_value:
                record.evidence_ids = list(dict.fromkeys(record.evidence_ids + evidence_ids))
                record.citation_evidence_ids = list(dict.fromkeys(record.citation_evidence_ids + evidence_ids))
                record.updated_at = now
                record.last_confirmed_at = now
                record.confidence = max(record.confidence, candidate.confidence)
                record.salience = max(record.salience, candidate.salience)
                record.metadata = {**record.metadata, **candidate_metadata}
                record.tags = list(dict.fromkeys(record.tags + candidate.tags))
                record.entity_names = list(dict.fromkeys(record.entity_names + candidate.entity_names))
                record.confirmation_count += 1
                record.contradicted_by = []
                if record.state == MemoryState.PENDING.value and candidate.confidence >= self.pipeline.pending_confidence_threshold:
                    record.state = MemoryState.ACTIVE.value
                if current_version is not None:
                    current_version.state = record.state
                    current_version.evidence_ids = list(dict.fromkeys(current_version.evidence_ids + evidence_ids))
                    current_version.citation_evidence_ids = list(
                        dict.fromkeys(current_version.citation_evidence_ids + evidence_ids)
                    )
                    current_version.confidence = max(current_version.confidence, candidate.confidence)
                    current_version.salience = max(current_version.salience, candidate.salience)
                    current_version.source_type = candidate.source_type
                    current_version.metadata = {**current_version.metadata, **candidate_metadata}
                    current_version.reasoning_trace = list(
                        dict.fromkeys(current_version.reasoning_trace + reasoning_trace)
                    )
                self._touch_memory_rehearsal(state=state, record_ids=[record.record_id], activated_at=now)
                return {
                    "action": MemoryAction.NONE.value,
                    "state": record.state,
                    "payload": {"record_id": record.record_id, "version_id": record.current_version_id, "summary": record.summary},
                    "events": [],
                }

            if desired_state == MemoryState.PENDING.value and explicit_record_id is None:
                return self._create_new_record(
                    state=state,
                    candidate=candidate,
                    scope=scope,
                    evidence_ids=evidence_ids,
                    now=now,
                    state_override=MemoryState.PENDING.value,
                    action_override=MemoryAction.ADD.value,
                    reasoning_trace=reasoning_trace,
                    metadata_override=candidate_metadata,
                )

            version_id = generate_id("version")
            if current_version is not None:
                current_version.state = MemoryState.SUPERSEDED.value
                current_version.contradicted_by = list(dict.fromkeys(current_version.contradicted_by + [version_id]))

            action = decision.action
            version = MemoryVersion(
                version_id=version_id,
                record_id=record.record_id,
                action=action,
                state=desired_state,
                value=candidate.value,
                summary=candidate.summary,
                created_at=now,
                confidence=candidate.confidence,
                evidence_ids=evidence_ids,
                metadata=candidate_metadata,
                reasoning_trace=reasoning_trace,
                supersedes_version_id=record.current_version_id,
                salience=candidate.salience,
                source_type=candidate.source_type,
                layer=candidate.layer,
                citation_evidence_ids=list(evidence_ids),
                contradicted_by=[],
            )
            state.versions[version.version_id] = version
            record.current_version_id = version.version_id
            record.version_ids.append(version.version_id)
            record.kind = candidate.kind
            record.key = candidate.key
            record.state = desired_state
            record.summary = candidate.summary
            record.value = candidate.value
            record.updated_at = now
            record.last_confirmed_at = now
            record.evidence_ids = list(dict.fromkeys(record.evidence_ids + evidence_ids))
            record.citation_evidence_ids = list(dict.fromkeys(record.citation_evidence_ids + evidence_ids))
            record.metadata = {**record.metadata, **candidate_metadata}
            record.tags = list(dict.fromkeys(record.tags + candidate.tags))
            record.entity_names = list(dict.fromkeys(record.entity_names + candidate.entity_names))
            record.search_text = self._search_text(record)
            record.embedding = self._make_embedding(record.search_text)
            record.confidence = candidate.confidence
            record.salience = candidate.salience
            record.source_type = candidate.source_type
            record.layer = candidate.layer
            record.confirmation_count = 1
            record.contradicted_by = []
            self._touch_memory_rehearsal(state=state, record_ids=[record.record_id], activated_at=now)
            events.append(
                {
                    "type": "memory_version_written",
                    "record_id": record.record_id,
                    "version_id": version.version_id,
                    "action": action,
                    "state": desired_state,
                }
            )
            return {
                "action": action,
                "state": record.state,
                "payload": {"record_id": record.record_id, "version_id": version.version_id, "summary": record.summary},
                "events": events,
            }

        return self._create_new_record(
            state=state,
            candidate=candidate,
            scope=scope,
            evidence_ids=evidence_ids,
            now=now,
            state_override=desired_state,
            action_override=decision.action,
            reasoning_trace=reasoning_trace,
            metadata_override=candidate_metadata,
        )

    def _create_new_record(
        self,
        *,
        state: EngineState,
        candidate: MemoryCandidate,
        scope: MemoryScope,
        evidence_ids: list[str],
        now: str,
        state_override: str,
        action_override: str,
        reasoning_trace: list[str] | None = None,
        metadata_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record_id = generate_id("memory")
        version_id = generate_id("version")
        record_metadata = dict(metadata_override or candidate.metadata)
        record = MemoryRecord(
            record_id=record_id,
            kind=candidate.kind,
            key=candidate.key,
            summary=candidate.summary,
            value=candidate.value,
            state=state_override,
            scope=scope,
            current_version_id=version_id,
            version_ids=[version_id],
            evidence_ids=list(evidence_ids),
            created_at=now,
            updated_at=now,
            metadata=record_metadata,
            tags=list(candidate.tags),
            entity_names=list(candidate.entity_names),
            search_text="",
            embedding=[],
            confidence=candidate.confidence,
            salience=candidate.salience,
            source_type=candidate.source_type,
            layer=candidate.layer,
            confirmation_count=1,
            last_confirmed_at=now,
            contradicted_by=[],
            citation_evidence_ids=list(evidence_ids),
        )
        record.search_text = self._search_text(record)
        record.embedding = self._make_embedding(record.search_text)
        version = MemoryVersion(
            version_id=version_id,
            record_id=record_id,
            action=action_override,
            state=state_override,
            value=candidate.value,
            summary=candidate.summary,
            created_at=now,
            confidence=candidate.confidence,
            evidence_ids=list(evidence_ids),
            metadata=record_metadata,
            reasoning_trace=list(reasoning_trace or ["new memory created", *candidate.tags]),
            salience=candidate.salience,
            source_type=candidate.source_type,
            layer=candidate.layer,
            citation_evidence_ids=list(evidence_ids),
            contradicted_by=[],
        )
        state.memories[record_id] = record
        state.versions[version_id] = version
        if state_override == MemoryState.ACTIVE.value and candidate.layer == MemoryLayer.SEMANTIC.value:
            state.lookup[self._canonical_lookup_key(scope, candidate.kind, candidate.key)] = record_id
        self._touch_memory_rehearsal(state=state, record_ids=[record_id], activated_at=now)
        return {
            "action": action_override,
            "state": state_override,
            "payload": {"record_id": record_id, "version_id": version_id, "summary": candidate.summary},
            "events": [
                {
                    "type": "memory_created",
                    "record_id": record_id,
                    "version_id": version_id,
                    "kind": candidate.kind,
                    "state": state_override,
                    "layer": candidate.layer,
                }
            ],
        }

    def _within_consolidation_window(self, timestamp: str | None) -> bool:
        return self._hours_since(timestamp) <= float(self.brain.consolidation_window_hours)

    def _semantic_candidates_for_consolidation(
        self,
        *,
        state: EngineState,
        record: MemoryRecord,
    ) -> list[tuple[MemoryCandidate, str, str]]:
        candidates: list[tuple[MemoryCandidate, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for evidence_id in record.citation_evidence_ids or record.evidence_ids:
            evidence = state.evidence.get(evidence_id)
            if evidence is None:
                continue
            if hasattr(self.extractor, "extract_semantic"):
                semantic_candidates = list(self.extractor.extract_semantic(evidence))
            else:
                semantic_candidates = [
                    candidate
                    for candidate in self.extractor.extract(evidence)
                    if candidate.layer != MemoryLayer.EPISODIC.value
                ]
            for candidate in semantic_candidates:
                finalized = self._finalize_candidate(candidate, evidence)
                if finalized.layer != MemoryLayer.SEMANTIC.value:
                    continue
                dedupe_key = (
                    finalized.kind,
                    finalized.key.lower(),
                    normalize_text(finalized.value).lower(),
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                candidates.append((finalized, evidence_id, evidence.event_type))
        return candidates

    def _consolidation_cluster_key(self, *, scope: MemoryScope, candidate: MemoryCandidate) -> str:
        if candidate.kind == MemoryKind.PROFILE_ATTRIBUTE.value:
            topic = candidate.key.lower()
        elif candidate.kind == MemoryKind.PREFERENCE.value:
            topic = normalize_text(str(candidate.metadata.get("topic", candidate.key))).lower()
        elif candidate.kind == MemoryKind.RELATION.value:
            entities = sorted(name.lower() for name in candidate.entity_names if normalize_text(name))
            topic = "::".join(entities or [candidate.key.lower()])
        elif candidate.source_type in (EventType.TOOL_RESULT.value, EventType.TOOL_CALL.value):
            entities = sorted(name.lower() for name in candidate.entity_names if normalize_text(name))
            topic = "::".join([normalize_text(str(candidate.metadata.get("name", candidate.key))).lower(), *entities])
        else:
            topic = normalize_text(candidate.key or " ".join(tokenize(candidate.value)[:6])).lower()
        return stable_hash(scope.to_dict(), candidate.kind, topic)

    def _cluster_conflict_reason(self, cluster: dict[str, Any]) -> str | None:
        candidates: list[MemoryCandidate] = cluster["candidates"]
        values = {normalize_text(candidate.value).lower() for candidate in candidates if normalize_text(candidate.value)}
        if cluster["kind"] == MemoryKind.PROFILE_ATTRIBUTE.value and len(values) > 1:
            return "profile_value_conflict"
        if cluster["kind"] == MemoryKind.PREFERENCE.value:
            sentiments = {
                normalize_text(str(candidate.metadata.get("sentiment", ""))).lower()
                for candidate in candidates
                if normalize_text(str(candidate.metadata.get("sentiment", "")))
            }
            if len(sentiments) > 1 or len(values) > 1:
                return "preference_conflict"
        return None

    def _weighted_candidate_score(self, candidate: MemoryCandidate) -> float:
        return (candidate.confidence * 0.7) + (candidate.salience * 0.3)

    def _representative_consolidation_candidate(self, cluster: dict[str, Any], when: str) -> MemoryCandidate:
        grouped: dict[str, list[MemoryCandidate]] = defaultdict(list)
        for candidate in cluster["candidates"]:
            grouped[normalize_text(candidate.value).lower()].append(candidate)
        best_group = max(
            grouped.values(),
            key=lambda items: (
                len(items),
                max(self._weighted_candidate_score(item) for item in items),
            ),
        )
        representative = max(best_group, key=self._weighted_candidate_score)
        support_count = len(cluster["evidence_ids"])
        entity_names = list(
            dict.fromkeys(name for candidate in cluster["candidates"] for name in candidate.entity_names)
        )
        tags = list(dict.fromkeys(tag for candidate in cluster["candidates"] for tag in candidate.tags))
        average_confidence = sum(self._weighted_candidate_score(item) for item in cluster["candidates"]) / max(
            len(cluster["candidates"]),
            1,
        )
        metadata = {
            **dict(representative.metadata),
            "consolidated_from_record_ids": sorted(cluster["record_ids"]),
            "support_count": support_count,
            "last_consolidated_at": when,
        }
        return MemoryCandidate(
            kind=representative.kind,
            key=representative.key,
            value=representative.value,
            summary=representative.summary,
            confidence=max(representative.confidence, min(average_confidence, 1.0)),
            state=MemoryState.ACTIVE.value,
            metadata=metadata,
            entity_names=entity_names,
            tags=tags,
            layer=MemoryLayer.SEMANTIC.value,
            salience=max(representative.salience, min(average_confidence, 1.0)),
            source_type=representative.source_type,
        )

    def _run_consolidation_pass(
        self,
        *,
        state: EngineState,
        scope: MemoryScope,
        limit: int,
        budget_ms: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        started = time.monotonic()
        now = utc_now()
        min_support = max(int(self.brain.consolidation_min_support), 1)
        episodic_records = [
            record
            for record in state.memories.values()
            if record.layer == MemoryLayer.EPISODIC.value
            and record.state == MemoryState.ACTIVE.value
            and record.scope.matches(scope)
            and self._within_consolidation_window(record.updated_at)
        ]
        episodic_records.sort(key=lambda item: item.updated_at, reverse=True)

        clusters: dict[str, dict[str, Any]] = {}
        processed_records = 0
        for record in episodic_records:
            if processed_records >= max(limit, 0):
                break
            if ((time.monotonic() - started) * 1000.0) > max(budget_ms, 1):
                break
            processed_records += 1
            for candidate, evidence_id, event_type in self._semantic_candidates_for_consolidation(state=state, record=record):
                cluster_key = self._consolidation_cluster_key(scope=record.scope, candidate=candidate)
                cluster = clusters.setdefault(
                    cluster_key,
                    {
                        "scope": record.scope,
                        "kind": candidate.kind,
                        "candidates": [],
                        "record_ids": set(),
                        "evidence_ids": set(),
                        "event_types": set(),
                    },
                )
                cluster["candidates"].append(candidate)
                cluster["record_ids"].add(record.record_id)
                cluster["evidence_ids"].add(evidence_id)
                cluster["event_types"].add(event_type)

        promotions: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for cluster_key, cluster in clusters.items():
            support_count = len(cluster["evidence_ids"])
            dual_source = (
                EventType.TOOL_RESULT.value in cluster["event_types"]
                and EventType.MESSAGE.value in cluster["event_types"]
            )
            confidence = sum(self._weighted_candidate_score(item) for item in cluster["candidates"]) / max(
                len(cluster["candidates"]),
                1,
            )
            conflict_reason = self._cluster_conflict_reason(cluster)
            if conflict_reason is not None:
                skipped.append(
                    {
                        "cluster_key": cluster_key,
                        "reason": conflict_reason,
                        "support_count": support_count,
                    }
                )
                continue
            if support_count < min_support and not dual_source:
                skipped.append(
                    {
                        "cluster_key": cluster_key,
                        "reason": "insufficient_support",
                        "support_count": support_count,
                    }
                )
                continue
            if confidence < 0.78:
                skipped.append(
                    {
                        "cluster_key": cluster_key,
                        "reason": "confidence_below_threshold",
                        "support_count": support_count,
                    }
                )
                continue

            candidate = self._representative_consolidation_candidate(cluster, now)
            outcome = self._apply_candidate(
                state=state,
                candidate=candidate,
                scope=cluster["scope"],
                evidence_ids=sorted(cluster["evidence_ids"]),
            )
            record_id = outcome["payload"].get("record_id")
            if record_id:
                entry = self._activation_entry(state, state.memories[record_id])
                entry["last_consolidated_at"] = now
                entry["activation_score"] = self._activation_score(record=state.memories[record_id], entry=entry)
            for source_record_id in cluster["record_ids"]:
                source_record = state.memories.get(source_record_id)
                if source_record is None:
                    continue
                entry = self._activation_entry(state, source_record)
                entry["last_consolidated_at"] = now
                entry["activation_score"] = self._activation_score(record=source_record, entry=entry)
            promotions.append(
                {
                    "memory_id": record_id,
                    "action": outcome["action"],
                    "kind": candidate.kind,
                    "key": candidate.key,
                    "summary": candidate.summary,
                    "support_count": support_count,
                }
            )
            events.extend(outcome["events"])
            events.append(
                {
                    "type": "memory_consolidated",
                    "memory_id": record_id,
                    "kind": candidate.kind,
                    "support_count": support_count,
                }
            )

        return {
            "changed": bool(promotions),
            "processed_records": processed_records,
            "promotions": promotions,
            "skipped": skipped,
        }, events

    def _rebuild_projections(self, state: EngineState) -> None:
        lexical: dict[str, list[str]] = defaultdict(list)
        graph: dict[str, list[str]] = defaultdict(list)
        relations: dict[str, RelationEdge] = {}
        active_records = [record for record in state.memories.values() if record.state == MemoryState.ACTIVE.value]
        activation = deep_copy_json(state.projections.get("activation") or {})
        consolidation = deep_copy_json(state.projections.get("consolidation") or {})
        state.lookup = {}

        for record in state.memories.values():
            if record.state == MemoryState.ACTIVE.value and record.layer == MemoryLayer.SEMANTIC.value:
                state.lookup[self._canonical_lookup_key(record.scope, record.kind, record.key)] = record.record_id
            if record.state != MemoryState.ACTIVE.value:
                continue
            for token in set(tokenize(record.search_text)):
                lexical[token].append(record.record_id)

        for index, left in enumerate(active_records):
            left_entities = set(name.lower() for name in left.entity_names)
            if not left_entities:
                continue
            for right in active_records[index + 1 :]:
                shared = sorted(left_entities.intersection(name.lower() for name in right.entity_names))
                if not shared:
                    continue
                graph[left.record_id].append(right.record_id)
                graph[right.record_id].append(left.record_id)
                edge_id = f"edge_{stable_hash(left.record_id, right.record_id, shared)[:16]}"
                relations[edge_id] = RelationEdge(
                    edge_id=edge_id,
                    source_record_id=left.record_id,
                    target_record_id=right.record_id,
                    relation_type="shared_entity",
                    weight=float(len(shared)),
                    metadata={"shared_entities": shared},
                )

        state.relations = relations
        state.projections = {
            "lexical": {key: sorted(set(value)) for key, value in lexical.items()},
            "graph": {key: sorted(set(value)) for key, value in graph.items()},
            "activation": activation,
            "consolidation": consolidation,
            "status": ProjectionStatus(
                lexical_current=True,
                vector_current=True,
                graph_current=True,
                last_projected_at=utc_now(),
            ).to_dict(),
        }
        self._refresh_activation_projection(state)

    def _record_matches_filters(
        self,
        record: MemoryRecord,
        filters: SearchFilters,
        *,
        include_scope: bool,
    ) -> bool:
        if include_scope and record.scope.matches(filters.scope) is False:
            return False
        if filters.kinds and record.kind not in filters.kinds:
            return False
        if filters.layers and record.layer not in filters.layers:
            return False
        if filters.source_types and record.source_type not in filters.source_types:
            return False
        if filters.states:
            if record.state not in filters.states:
                return False
        elif not filters.include_pending and record.state != MemoryState.ACTIVE.value:
            allowed_states = {MemoryState.ACTIVE.value}
            if self._brain_enabled():
                allowed_states.add(MemoryState.DORMANT.value)
            if record.state not in allowed_states:
                return False
        if filters.tags and not set(filters.tags).intersection(record.tags):
            return False
        for key, value in filters.metadata.items():
            if record.metadata.get(key) != value:
                return False
        return True

    def _rank_records(
        self,
        *,
        state: EngineState,
        query: str,
        search_filters: SearchFilters,
        intent: str | None = None,
    ) -> list[tuple[float, MemoryRecord, dict[str, float]]]:
        query = normalize_text(query)
        query_tokens = unique_tokens(query)
        query_entities = extract_entities(query)
        query_embedding = self._make_embedding(query)
        query_concepts = self._query_concepts(query)
        intent = intent or self._classify_query_intent(query)
        projection_status = ProjectionStatus.from_dict(state.projections.get("status"))
        _ = projection_status

        eligible_records: list[tuple[MemoryRecord, float]] = []
        for record in state.memories.values():
            if not self._record_matches_filters(record, search_filters, include_scope=False):
                continue
            scope_score = self._scope_score(
                record_scope=record.scope,
                target_scope=search_filters.scope,
                hierarchical=search_filters.hierarchical_scope,
            )
            if scope_score <= 0:
                continue
            eligible_records.append((record, scope_score))

        graph = state.projections.get("graph") or {}
        if self._brain_enabled():
            shortlist_ids = self._brain_shortlist_ids(
                state=state,
                eligible_records=eligible_records,
                query=query,
                query_tokens=query_tokens,
                query_entities=query_entities,
                search_filters=search_filters,
                adjacency=graph,
            )
            if shortlist_ids:
                narrowed = [
                    (record, scope_score)
                    for record, scope_score in eligible_records
                    if record.record_id in shortlist_ids
                ]
                records = narrowed or eligible_records
            else:
                records = eligible_records
        else:
            candidate_ids = set()
            lexical_index = state.projections.get("lexical") or {}
            for token in query_tokens:
                candidate_ids.update(lexical_index.get(token, []))
            if candidate_ids:
                narrowed = [
                    (record, scope_score)
                    for record, scope_score in eligible_records
                    if record.record_id in candidate_ids
                ]
                records = narrowed or eligible_records
            else:
                records = eligible_records

        activation = state.projections.get("activation") or {}
        first_pass: list[tuple[float, MemoryRecord, dict[str, float]]] = []
        for record, scope_score in records:
            lexical_score = self._lexical_score(query_tokens, record.search_text)
            vector_score = max(cosine_similarity(query_embedding, record.embedding), 0.0)
            recency_score = self._recency_score(record.updated_at)
            access_score = min(math.log1p(record.access_count) / 5.0, 1.0)
            salience_score = min(max(record.salience, 0.0), 1.0)
            intent_score = self._intent_score(intent, record)
            llm_score = self._llm_rerank_score(query_concepts=query_concepts, record=record)
            total = (
                (lexical_score * 0.38)
                + (vector_score * 0.18)
                + (recency_score * 0.07)
                + (access_score * 0.03)
                + (scope_score * 0.14)
                + (salience_score * 0.11)
                + (intent_score * 0.05)
                + (llm_score * 0.04)
            )
            breakdown = {
                "base": total,
                "lexical": lexical_score,
                "vector": vector_score,
                "graph": 0.0,
                "recency": recency_score,
                "access": access_score,
                "scope": scope_score,
                "salience": salience_score,
                "intent": intent_score,
                "llm": llm_score,
                "reranker": 0.0,
                "activation": 0.0,
                "rehearsal": 0.0,
                "exact_cue": 0.0,
                "brain_graph_spread": 0.0,
                "dormancy_penalty": 0.0,
                "contradiction_penalty": 0.0,
            }
            first_pass.append((total, record, breakdown))

        first_pass.sort(key=lambda item: item[0], reverse=True)
        top_seed_ids = [record.record_id for _, record, _ in first_pass[: min(4, len(first_pass))]]

        ranked: list[tuple[float, MemoryRecord, dict[str, float]]] = []
        for base_score, record, breakdown in first_pass:
            graph_score = self._graph_score(
                record=record,
                query_entities=query_entities,
                top_seed_ids=top_seed_ids,
                adjacency=graph,
            )
            total = base_score + (graph_score * 0.1)
            breakdown["graph"] = graph_score
            if self._brain_enabled():
                entry = activation.get(record.record_id) or {}
                activation_score = float(entry.get("activation_score", 0.0))
                rehearsal_score = min(
                    (
                        int(entry.get("rehearsal_count", 0))
                        + int(entry.get("retrieval_count", 0))
                    )
                    / 6.0,
                    1.0,
                )
                exact_cue = self._direct_cue_score(
                    query=query,
                    record=record,
                    query_tokens=query_tokens,
                    query_entities=query_entities,
                )
                graph_spread = graph_score
                dormancy_penalty = 1.0 if record.state == MemoryState.DORMANT.value else 0.0
                contradiction_penalty = min(len(record.contradicted_by) * 0.5, 1.0)
                total += (
                    (activation_score * 0.18)
                    + (rehearsal_score * 0.08)
                    + (exact_cue * 0.06)
                    + (graph_spread * 0.04)
                    - (dormancy_penalty * 0.12)
                    - (contradiction_penalty * 0.10)
                )
                breakdown["activation"] = activation_score
                breakdown["rehearsal"] = rehearsal_score
                breakdown["exact_cue"] = exact_cue
                breakdown["brain_graph_spread"] = graph_spread
                breakdown["dormancy_penalty"] = dormancy_penalty
                breakdown["contradiction_penalty"] = contradiction_penalty
            ranked.append((total, record, breakdown))

        if self.reranker is not None and ranked:
            rerank_outcomes = self.reranker.rerank(
                query=query,
                candidates=ranked,
                state=state,
                search_filters=search_filters,
                intent=intent,
            )
            reranked: list[tuple[float, MemoryRecord, dict[str, float]]] = []
            for total, record, breakdown in ranked:
                outcome = rerank_outcomes.get(record.record_id)
                reranker_score = float(outcome.score) if outcome is not None else 0.0
                updated_breakdown = dict(breakdown)
                updated_breakdown["reranker"] = reranker_score
                reranked.append((total + (reranker_score * 0.12), record, updated_breakdown))
            ranked = reranked

        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked and eligible_records:
            fallback = []
            for record, scope_score in eligible_records:
                recency_score = self._recency_score(record.updated_at)
                salience_score = min(max(record.salience, 0.0), 1.0)
                total = (scope_score * 0.6) + (recency_score * 0.25) + (salience_score * 0.15)
                exact_cue = self._direct_cue_score(
                    query=query,
                    record=record,
                    query_tokens=query_tokens,
                    query_entities=query_entities,
                )
                fallback.append(
                    (
                        total,
                        record,
                        {
                            "lexical": 0.0,
                            "vector": 0.0,
                            "graph": 0.0,
                            "recency": recency_score,
                            "access": 0.0,
                            "scope": scope_score,
                            "salience": salience_score,
                            "intent": self._intent_score(intent, record),
                            "llm": 0.0,
                            "reranker": 0.0,
                            "activation": float((activation.get(record.record_id) or {}).get("activation_score", 0.0)),
                            "rehearsal": 0.0,
                            "exact_cue": exact_cue,
                            "brain_graph_spread": 0.0,
                            "dormancy_penalty": 1.0 if record.state == MemoryState.DORMANT.value else 0.0,
                            "contradiction_penalty": min(len(record.contradicted_by) * 0.5, 1.0),
                        },
                    )
                )
            fallback.sort(key=lambda item: item[0], reverse=True)
            return fallback
        return ranked

    def _brain_shortlist_ids(
        self,
        *,
        state: EngineState,
        eligible_records: list[tuple[MemoryRecord, float]],
        query: str,
        query_tokens: list[str],
        query_entities: list[str],
        search_filters: SearchFilters,
        adjacency: dict[str, list[str]],
    ) -> list[str]:
        del search_filters
        lexical_index = state.projections.get("lexical") or {}
        activation = state.projections.get("activation") or {}
        lexical_ids: list[str] = []
        for token in query_tokens:
            lexical_ids.extend(lexical_index.get(token, []))

        strong_cue_ids = [
            record.record_id
            for record, _ in eligible_records
            if self._direct_cue_score(
                query=query,
                record=record,
                query_tokens=query_tokens,
                query_entities=query_entities,
            )
            > 0.85
        ]
        activated_ids = [
            record.record_id
            for record, _ in sorted(
                eligible_records,
                key=lambda item: (
                    float((activation.get(item[0].record_id) or {}).get("activation_score", 0.0)),
                    (activation.get(item[0].record_id) or {}).get("last_activated_at") or "",
                ),
                reverse=True,
            )
            if float((activation.get(record.record_id) or {}).get("activation_score", 0.0)) > 0.0
        ]
        seed_ids = list(dict.fromkeys(lexical_ids + strong_cue_ids))[:4]
        neighbor_ids: list[str] = []
        for seed_id in seed_ids:
            neighbor_ids.extend(adjacency.get(seed_id, []))
        shortlist = list(dict.fromkeys(lexical_ids + strong_cue_ids + activated_ids + neighbor_ids))
        return shortlist[: max(int(self.brain.fast_path_candidate_limit), 1)]

    def _direct_cue_score(
        self,
        *,
        query: str,
        record: MemoryRecord,
        query_tokens: list[str],
        query_entities: list[str],
    ) -> float:
        normalized_query = normalize_text(query).lower()
        if not normalized_query:
            return 0.0
        record_value = normalize_text(record.value).lower()
        record_key = normalize_text(record.key).lower()
        record_text = normalize_text(record.search_text).lower()
        if normalized_query in {record_value, record_key}:
            return 1.0
        if normalized_query and normalized_query in record_text:
            return 0.92
        record_entities = {normalize_text(item).lower() for item in record.entity_names if normalize_text(item)}
        query_entity_set = {normalize_text(item).lower() for item in query_entities if normalize_text(item)}
        if record_entities.intersection(query_entity_set):
            return 0.9
        record_tokens = set(tokenize(record_text))
        token_overlap = len(set(query_tokens).intersection(record_tokens)) / max(len(set(query_tokens)), 1)
        if record.state == MemoryState.DORMANT.value and token_overlap > 0.0:
            return max(token_overlap, 0.86)
        return min(token_overlap, 1.0)

    def _scope_score(
        self,
        *,
        record_scope: MemoryScope,
        target_scope: MemoryScope | None,
        hierarchical: bool,
    ) -> float:
        if target_scope is None or target_scope.is_empty():
            return 1.0
        if not hierarchical:
            return 1.0 if record_scope.matches(target_scope) else 0.0

        if target_scope.user_id is not None:
            if record_scope.user_id not in (None, target_scope.user_id):
                return 0.0
        if target_scope.agent_id is not None:
            if record_scope.agent_id not in (None, target_scope.agent_id):
                return 0.0
        if target_scope.run_id is not None:
            if record_scope.run_id not in (None, target_scope.run_id):
                return 0.0

        score = 0.0
        if target_scope.user_id is None:
            score += 0.25
        elif record_scope.user_id == target_scope.user_id:
            score += 0.45
        elif record_scope.user_id is None and target_scope.agent_id and record_scope.agent_id == target_scope.agent_id:
            score += 0.18
        else:
            return 0.0

        if target_scope.agent_id is None:
            if record_scope.user_id == target_scope.user_id and record_scope.agent_id is not None:
                score += 0.18
            else:
                score += 0.15
        elif record_scope.agent_id == target_scope.agent_id:
            score += 0.25
        elif record_scope.agent_id is None and record_scope.user_id == target_scope.user_id:
            score += 0.16
        elif record_scope.agent_id == target_scope.agent_id and record_scope.user_id is None:
            score += 0.14
        else:
            return 0.0

        if target_scope.run_id is None:
            score += 0.12 if record_scope.run_id is None else 0.08
        elif record_scope.run_id == target_scope.run_id:
            score += 0.3
        elif record_scope.run_id is None:
            score += 0.12
        else:
            return 0.0

        return min(score, 1.0)

    def _make_embedding(self, text: str) -> list[float]:
        text = normalize_text(text)
        if not text:
            return []
        if self.embedding_model is None:
            return hashed_embedding(text)
        embedding = self.embedding_model.get_embedding(text)
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
            embedding = embedding[0]
        try:
            return [float(value) for value in embedding]
        except TypeError:
            return hashed_embedding(text)

    def _search_text(self, record: MemoryRecord) -> str:
        return normalize_text(
            " ".join(
                [
                    record.layer,
                    record.source_type,
                    record.kind,
                    record.key,
                    record.summary,
                    record.value,
                    " ".join(record.tags),
                    " ".join(record.entity_names),
                ]
            )
        )

    def _lexical_score(self, query_tokens: list[str], search_text: str) -> float:
        if not query_tokens:
            return 0.0
        record_tokens = set(tokenize(search_text))
        if not record_tokens:
            return 0.0
        overlap = len(set(query_tokens).intersection(record_tokens))
        substring_bonus = 0.15 if normalize_text(" ".join(query_tokens)) in search_text.lower() else 0.0
        return min((overlap / max(len(set(query_tokens)), 1)) + substring_bonus, 1.0)

    def _graph_score(
        self,
        *,
        record: MemoryRecord,
        query_entities: list[str],
        top_seed_ids: list[str],
        adjacency: dict[str, list[str]],
    ) -> float:
        score = 0.0
        record_entities = {entity.lower() for entity in record.entity_names}
        if query_entities:
            shared_entities = record_entities.intersection(entity.lower() for entity in query_entities)
            score += min(len(shared_entities) * 0.25, 0.5)
        neighbors = set(adjacency.get(record.record_id, []))
        shared_seeds = neighbors.intersection(top_seed_ids)
        score += min(len(shared_seeds) * 0.1, 0.3)
        return min(score, 1.0)

    def _recency_score(self, timestamp: str) -> float:
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        age_seconds = max((datetime.now(timezone.utc) - parsed).total_seconds(), 0.0)
        age_days = age_seconds / 86400.0
        return 1.0 / (1.0 + age_days)

    def _query_concepts(self, text: str) -> list[str]:
        if self.chat_model is not None and hasattr(self.chat_model, "extract_concepts"):
            try:
                concepts = list(self.chat_model.extract_concepts(text))
                return [normalize_text(str(item)).lower() for item in concepts if normalize_text(str(item))]
            except Exception:
                return unique_tokens(text)[:12]
        return unique_tokens(text)[:12]

    def _llm_rerank_score(self, *, query_concepts: list[str], record: MemoryRecord) -> float:
        if not query_concepts:
            return 0.0
        record_concepts = {
            normalize_text(item).lower()
            for item in (record.entity_names + record.tags + [record.key, record.kind])
            if normalize_text(item)
        }
        if not record_concepts:
            return 0.0
        overlap = len(set(query_concepts).intersection(record_concepts))
        return min(overlap / max(len(set(query_concepts)), 1), 1.0)

    def _classify_query_intent(self, query: str) -> str:
        lower_query = normalize_text(query).lower()
        if any(
            phrase in lower_query
            for phrase in (
                "what do you know about me",
                "what do you remember about me",
                "tell me about me",
                "what do you know so far",
            )
        ):
            return "general"
        if any(token in lower_query for token in ("favorite", "prefer", "like", "love", "dislike", "hate")):
            return "preference"
        if any(
            token in lower_query
            for token in ("who knows", "relationship", "works at", "visited", "friends with", "related to")
        ):
            return "relationship"
        if any(token in lower_query for token in ("recent", "earlier", "last time", "remember when", "what happened")):
            return "episodic"
        if any(token in lower_query for token in ("tool", "calendar", "weather", "lookup", "search result", "result")):
            return "tool"
        if any(token in lower_query for token in ("name", "age", "live", "from", "work", "profile", "who am i")):
            return "profile"
        return "general"

    def _intent_score(self, intent: str, record: MemoryRecord) -> float:
        if intent == "profile":
            return 1.0 if record.kind in (MemoryKind.PROFILE_ATTRIBUTE.value, MemoryKind.FACT.value) else 0.2
        if intent == "preference":
            return 1.0 if record.kind == MemoryKind.PREFERENCE.value else 0.2
        if intent == "relationship":
            return 1.0 if record.kind in (MemoryKind.RELATION.value, MemoryKind.ENTITY.value) else 0.2
        if intent == "episodic":
            return 1.0 if record.layer == MemoryLayer.EPISODIC.value else 0.25
        if intent == "tool":
            return 1.0 if record.source_type in (EventType.TOOL_RESULT.value, EventType.TOOL_CALL.value) else 0.25
        return 0.5 if record.layer == MemoryLayer.SEMANTIC.value else 0.35

    def _section_for_record(self, *, record: MemoryRecord, intent: str) -> str:
        if record.source_type in (EventType.TOOL_RESULT.value, EventType.TOOL_CALL.value):
            return "tool_observations"
        if record.layer == MemoryLayer.EPISODIC.value:
            return "recent_episodes"
        if record.kind == MemoryKind.PREFERENCE.value:
            return "preferences"
        if record.kind == MemoryKind.RELATION.value:
            return "relationships"
        if intent == "relationship" and record.kind == MemoryKind.ENTITY.value:
            return "relationships"
        return "profile"

    def _context_item_payload(
        self,
        *,
        state: EngineState,
        record: MemoryRecord,
        score: float,
        breakdown: dict[str, float],
    ) -> dict[str, Any]:
        citation_ids = record.citation_evidence_ids or record.evidence_ids
        citations = []
        for evidence_id in citation_ids:
            evidence = state.evidence.get(evidence_id)
            if evidence is None:
                continue
            citations.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "summary": summarize_text(evidence.text, max_words=18),
                    "event_type": evidence.event_type,
                    "source_type": evidence.source_type,
                    "occurred_at": evidence.occurred_at or evidence.created_at,
                }
            )
        return {
            "memory_id": record.record_id,
            "kind": record.kind,
            "key": record.key,
            "state": record.state,
            "summary": record.summary,
            "value": record.value,
            "layer": record.layer,
            "source_type": record.source_type,
            "scope": record.scope.to_dict(),
            "score": score,
            "rank_breakdown": dict(breakdown),
            "updated_at": record.updated_at,
            "salience": record.salience,
            "confirmation_count": record.confirmation_count,
            "citation_evidence_ids": list(citation_ids),
            "citations": citations,
        }

    def _estimate_tokens(self, text: str) -> int:
        return max(1, math.ceil(len(normalize_text(text).split()) * 1.35))

    def _build_search_payload(
        self,
        *,
        state: EngineState,
        query: str,
        search_filters: SearchFilters,
        ranked: list[tuple[float, MemoryRecord, dict[str, float]]],
        projection_status: ProjectionStatus,
    ) -> dict[str, Any]:
        results: list[SearchResult] = []
        for score, record, breakdown in ranked[: search_filters.limit]:
            evidence_items = [
                state.evidence[evidence_id]
                for evidence_id in record.citation_evidence_ids or record.evidence_ids
                if evidence_id in state.evidence
            ]
            results.append(
                SearchResult(
                    memory=record,
                    score=score,
                    rank_breakdown=breakdown,
                    evidence=evidence_items,
                    projection_status=projection_status,
                )
            )

        return {
            "query": query,
            "filters": search_filters.to_dict(),
            "results": [result.to_dict() for result in results],
            "projection_status": projection_status.to_dict(),
        }

    def _search_trace_payload(
        self,
        *,
        state: EngineState,
        query: str,
        ranked: list[tuple[float, MemoryRecord, dict[str, float]]],
        search_filters: SearchFilters,
        working_memory: list[dict[str, Any]],
        dormancy_transitions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "query": query,
            "filters": search_filters.to_dict(),
            "pipeline": self.pipeline.describe(),
            "activation": {
                "mode": self.brain.mode,
                "results": self._activation_trace_items(state=state, ranked=ranked),
            },
            "working_memory": {
                "selected_memory_ids": [item["memory_id"] for item in working_memory],
                "items": working_memory,
            },
            "consolidation": {
                "last_run": dict(state.projections.get("consolidation") or {}),
                "dormancy_transitions": list(dormancy_transitions),
            },
            "ranking": {
                "candidate_count": len(ranked),
                "reranker_applied": self.reranker is not None,
                "results": self._ranking_trace_items(state=state, ranked=ranked),
            },
        }

    def _ranking_trace_items(
        self,
        *,
        state: EngineState,
        ranked: list[tuple[float, MemoryRecord, dict[str, float]]],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for score, record, breakdown in ranked[: self.pipeline.max_trace_results]:
            current_version = state.versions.get(record.current_version_id) if record.current_version_id else None
            activation_entry = (state.projections.get("activation") or {}).get(record.record_id) or {}
            items.append(
                {
                    "memory_id": record.record_id,
                    "kind": record.kind,
                    "key": record.key,
                    "score": score,
                    "state": record.state,
                    "rank_breakdown": dict(breakdown),
                    "latest_action": current_version.action if current_version is not None else None,
                    "reasoning_trace": list(current_version.reasoning_trace if current_version is not None else []),
                    "activation": {
                        "activation_score": float(activation_entry.get("activation_score", 0.0)),
                        "last_activated_at": activation_entry.get("last_activated_at"),
                        "rehearsal_count": int(activation_entry.get("rehearsal_count", 0)),
                        "retrieval_count": int(activation_entry.get("retrieval_count", 0)),
                        "last_consolidated_at": activation_entry.get("last_consolidated_at"),
                    },
                }
            )
        return items

    def _activation_trace_items(
        self,
        *,
        state: EngineState,
        ranked: list[tuple[float, MemoryRecord, dict[str, float]]],
    ) -> list[dict[str, Any]]:
        activation = state.projections.get("activation") or {}
        items: list[dict[str, Any]] = []
        for _, record, breakdown in ranked[: self.pipeline.max_trace_results]:
            entry = activation.get(record.record_id) or {}
            items.append(
                {
                    "memory_id": record.record_id,
                    "activation_score": float(entry.get("activation_score", 0.0)),
                    "last_activated_at": entry.get("last_activated_at"),
                    "rehearsal_count": int(entry.get("rehearsal_count", 0)),
                    "retrieval_count": int(entry.get("retrieval_count", 0)),
                    "last_consolidated_at": entry.get("last_consolidated_at"),
                    "factors": {
                        "activation": float(breakdown.get("activation", 0.0)),
                        "rehearsal": float(breakdown.get("rehearsal", 0.0)),
                        "exact_cue": float(breakdown.get("exact_cue", 0.0)),
                        "brain_graph_spread": float(breakdown.get("brain_graph_spread", 0.0)),
                        "dormancy_penalty": float(breakdown.get("dormancy_penalty", 0.0)),
                        "contradiction_penalty": float(breakdown.get("contradiction_penalty", 0.0)),
                    },
                }
            )
        return items

    def _should_include_trace(self, include_trace: bool) -> bool:
        return bool(include_trace or self.pipeline.default_include_trace)

    def _build_context_artifacts(
        self,
        *,
        state: EngineState,
        query: str,
        search_filters: SearchFilters,
        limit: int,
        max_tokens: int,
        include_debug: bool,
        include_trace: bool,
        context_policy: str,
    ) -> dict[str, Any]:
        projection_status = ProjectionStatus.from_dict(state.projections.get("status"))
        intent = self._classify_query_intent(query)
        ranked = self._rank_records(state=state, query=query, search_filters=search_filters, intent=intent)
        sections = self._empty_context_sections()
        citations: dict[str, dict[str, Any]] = {}
        selected_ranked: list[tuple[float, MemoryRecord, dict[str, float]]] = []
        selected_memory_ids: list[str] = []
        selected_fact_keys: set[str] = set()
        selected_values: set[tuple[str, str]] = set()
        selected_non_episodic_evidence_ids: set[str] = set()
        dropped_duplicate_count = 0
        dropped_budget_count = 0
        policy = self._context_policy_settings(context_policy=context_policy, limit=limit)
        canonical_match_counts = self._canonical_match_counts(ranked)

        working_memory_ids: list[str] = []
        if self._brain_enabled():
            for score, record, breakdown in self._working_memory_candidates(ranked=ranked):
                if len(working_memory_ids) >= min(policy["max_total"], 3, max(int(self.brain.working_memory_size), 1)):
                    break
                if record.state == MemoryState.DORMANT.value and not self._should_reactivate_dormant(
                    record=record,
                    breakdown=breakdown,
                    canonical_match_counts=canonical_match_counts,
                ):
                    continue

                context_item = self._context_item_payload(state=state, record=record, score=score, breakdown=breakdown)
                fact_key = self._canonical_context_fact_key(
                    record=record,
                    section_name="working_memory",
                    context_item=context_item,
                )
                normalized_value = normalize_text(context_item["value"]).lower()
                evidence_ids = set(context_item["citation_evidence_ids"])
                value_key = ("working_memory", normalized_value)
                if fact_key in selected_fact_keys or (normalized_value and value_key in selected_values):
                    dropped_duplicate_count += 1
                    continue

                trial_sections = {name: list(items) for name, items in sections.items()}
                trial_sections["working_memory"].append(context_item)
                trial_citations = dict(citations)
                self._add_context_citations(
                    citations=trial_citations,
                    state=state,
                    record=record,
                    context_item=context_item,
                )
                trial_grounding = self._format_context_sections(
                    sections=trial_sections,
                    intent=intent,
                    context_policy=context_policy,
                    citations=list(trial_citations.values()),
                )
                trial_tokens = self._estimate_tokens(trial_grounding)
                if selected_memory_ids and trial_tokens > max_tokens:
                    dropped_budget_count += 1
                    continue

                sections = trial_sections
                citations = trial_citations
                selected_ranked.append((score, record, breakdown))
                selected_memory_ids.append(record.record_id)
                working_memory_ids.append(record.record_id)
                selected_fact_keys.add(fact_key)
                if normalized_value:
                    selected_values.add(value_key)
                if record.layer != MemoryLayer.EPISODIC.value:
                    selected_non_episodic_evidence_ids.update(evidence_ids)

        selection_candidates = sorted(
            ranked,
            key=lambda item: (self._selection_priority(record=item[1], intent=intent), item[0]),
            reverse=True,
        )

        for score, record, breakdown in selection_candidates:
            if len(selected_memory_ids) >= policy["max_total"]:
                continue
            if record.record_id in selected_memory_ids:
                continue

            section_name = self._section_for_record(record=record, intent=intent)
            if len(sections[section_name]) >= policy["max_per_section"]:
                continue

            if record.state == MemoryState.DORMANT.value and not self._should_reactivate_dormant(
                record=record,
                breakdown=breakdown,
                canonical_match_counts=canonical_match_counts,
            ):
                continue

            context_item = self._context_item_payload(state=state, record=record, score=score, breakdown=breakdown)
            fact_key = self._canonical_context_fact_key(record=record, section_name=section_name, context_item=context_item)
            normalized_value = normalize_text(context_item["value"]).lower()
            evidence_ids = set(context_item["citation_evidence_ids"])

            value_key = (section_name, normalized_value)
            if fact_key in selected_fact_keys or (normalized_value and value_key in selected_values):
                dropped_duplicate_count += 1
                continue

            if (
                record.layer == MemoryLayer.EPISODIC.value
                and intent != "episodic"
                and evidence_ids.intersection(selected_non_episodic_evidence_ids)
            ):
                dropped_duplicate_count += 1
                continue

            trial_sections = {name: list(items) for name, items in sections.items()}
            trial_sections[section_name].append(context_item)
            trial_citations = dict(citations)
            self._add_context_citations(
                citations=trial_citations,
                state=state,
                record=record,
                context_item=context_item,
            )
            trial_grounding = self._format_context_sections(
                sections=trial_sections,
                intent=intent,
                context_policy=context_policy,
                citations=list(trial_citations.values()),
            )
            trial_tokens = self._estimate_tokens(trial_grounding)
            if selected_memory_ids and trial_tokens > max_tokens:
                dropped_budget_count += 1
                continue

            sections = trial_sections
            citations = trial_citations
            selected_ranked.append((score, record, breakdown))
            selected_memory_ids.append(record.record_id)
            selected_fact_keys.add(fact_key)
            if normalized_value:
                selected_values.add(value_key)
            if record.layer != MemoryLayer.EPISODIC.value:
                selected_non_episodic_evidence_ids.update(evidence_ids)

        grounding = self._format_context_sections(
            sections=sections,
            intent=intent,
            context_policy=context_policy,
            citations=list(citations.values()),
        )
        prompt_tokens_estimate = self._estimate_tokens(grounding) if grounding else 0
        omitted_memory_ids = [
            record.record_id
            for _, record, _ in ranked
            if record.record_id not in selected_memory_ids
        ]

        debug: dict[str, Any] = {}
        if include_debug:
            debug = {
                "intent": intent,
                "candidate_count": len(ranked),
                "included_memory_ids": list(selected_memory_ids),
                "max_tokens": max_tokens,
                "used_tokens": prompt_tokens_estimate,
                "prompt_tokens_estimate": prompt_tokens_estimate,
                "selected_count": len(selected_memory_ids),
                "dropped_duplicate_count": dropped_duplicate_count,
                "dropped_budget_count": dropped_budget_count,
                "omitted_memory_ids": omitted_memory_ids,
                "grounding_preview": grounding,
                "filters": search_filters.to_dict(),
            }

        trace: dict[str, Any] = {}
        if include_trace:
            trace = {
                "query": query,
                "intent": intent,
                "pipeline": self.pipeline.describe(),
                "ranking": {
                    "candidate_count": len(ranked),
                    "reranker_applied": self.reranker is not None,
                    "results": self._ranking_trace_items(state=state, ranked=ranked),
                },
                "activation": {
                    "mode": self.brain.mode,
                    "results": self._activation_trace_items(state=state, ranked=ranked),
                },
                "working_memory": {
                    "selected_memory_ids": list(working_memory_ids),
                    "items": list(sections["working_memory"]),
                },
                "consolidation": {
                    "last_run": dict(state.projections.get("consolidation") or {}),
                    "dormancy_transitions": [],
                },
                "grounding": {
                    "context_policy": context_policy,
                    "selected_memory_ids": list(selected_memory_ids),
                    "omitted_memory_ids": omitted_memory_ids,
                    "prompt_tokens_estimate": prompt_tokens_estimate,
                    "dropped_duplicate_count": dropped_duplicate_count,
                    "dropped_budget_count": dropped_budget_count,
                    "section_counts": {name: len(items) for name, items in sections.items()},
                    "citation_count": len(citations),
                },
            }

        memory_pack = ContextPack(
            query=query,
            scope=search_filters.scope or MemoryScope(),
            intent=intent,
            working_memory=sections["working_memory"],
            profile=sections["profile"],
            preferences=sections["preferences"],
            relationships=sections["relationships"],
            recent_episodes=sections["recent_episodes"],
            tool_observations=sections["tool_observations"],
            citations=list(citations.values()),
            projection_status=projection_status,
            debug=debug,
            trace=trace,
        )
        return {
            "memory_pack": memory_pack,
            "grounding": grounding,
            "ranked": ranked,
            "selected_ranked": selected_ranked,
            "selected_memory_ids": list(selected_memory_ids),
            "working_memory_ids": list(working_memory_ids),
            "projection_status": projection_status,
        }

    def _empty_context_sections(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "working_memory": [],
            "profile": [],
            "preferences": [],
            "relationships": [],
            "recent_episodes": [],
            "tool_observations": [],
        }

    def _context_policy_settings(self, *, context_policy: str, limit: int) -> dict[str, int]:
        normalized_policy = context_policy if context_policy in {"compact", "balanced", "verbose"} else "compact"
        defaults = {
            "compact": {"max_total": 6, "max_per_section": 2},
            "balanced": {"max_total": 8, "max_per_section": 3},
            "verbose": {"max_total": max(1, limit), "max_per_section": max(1, limit)},
        }[normalized_policy]
        return {
            "max_total": max(1, min(limit, defaults["max_total"])),
            "max_per_section": max(1, min(limit, defaults["max_per_section"])),
        }

    def _working_memory_candidates(
        self,
        *,
        ranked: list[tuple[float, MemoryRecord, dict[str, float]]],
    ) -> list[tuple[float, MemoryRecord, dict[str, float]]]:
        if not self._brain_enabled():
            return []
        return sorted(
            ranked,
            key=lambda item: (item[2].get("activation", 0.0), item[0]),
            reverse=True,
        )

    def _selection_priority(self, *, record: MemoryRecord, intent: str) -> int:
        priority = 0
        if record.source_type in (EventType.TOOL_RESULT.value, EventType.TOOL_CALL.value):
            priority += 5 if intent == "tool" else 2
        if record.layer == MemoryLayer.EPISODIC.value and intent == "episodic":
            priority += 6
        elif record.layer != MemoryLayer.EPISODIC.value:
            priority += 4
        if intent == "profile" and record.kind == MemoryKind.PROFILE_ATTRIBUTE.value:
            priority += 2
        if intent == "preference" and record.kind == MemoryKind.PREFERENCE.value:
            priority += 2
        if intent == "relationship" and record.kind == MemoryKind.RELATION.value:
            priority += 2
        return priority

    def _canonical_context_fact_key(
        self,
        *,
        record: MemoryRecord,
        section_name: str,
        context_item: dict[str, Any],
    ) -> str:
        normalized_value = normalize_text(context_item["value"]).lower()
        evidence_key = ",".join(sorted(context_item["citation_evidence_ids"]))
        if record.layer == MemoryLayer.EPISODIC.value:
            return f"{section_name}:{record.layer}:{evidence_key or normalized_value}"
        return f"{section_name}:{record.kind}:{record.key.lower()}:{normalized_value}"

    def _add_context_citations(
        self,
        *,
        citations: dict[str, dict[str, Any]],
        state: EngineState,
        record: MemoryRecord,
        context_item: dict[str, Any],
    ) -> None:
        for evidence_id in context_item["citation_evidence_ids"]:
            evidence = state.evidence.get(evidence_id)
            if evidence is None or evidence_id in citations:
                continue
            citations[evidence_id] = {
                "evidence_id": evidence.evidence_id,
                "memory_id": record.record_id,
                "summary": summarize_text(evidence.text, max_words=18),
                "text": evidence.text,
                "event_type": evidence.event_type,
                "source_type": evidence.source_type,
                "scope": evidence.scope.to_dict(),
                "occurred_at": evidence.occurred_at or evidence.created_at,
            }

    def _format_context_pack(self, memory_pack: ContextPack, context_policy: str = "compact") -> str:
        return self._format_context_sections(
            sections={
                "working_memory": memory_pack.working_memory,
                "profile": memory_pack.profile,
                "preferences": memory_pack.preferences,
                "relationships": memory_pack.relationships,
                "recent_episodes": memory_pack.recent_episodes,
                "tool_observations": memory_pack.tool_observations,
            },
            intent=memory_pack.intent,
            context_policy=context_policy,
            citations=memory_pack.citations,
        )

    def _format_context_sections(
        self,
        *,
        sections: dict[str, list[dict[str, Any]]],
        intent: str,
        context_policy: str,
        citations: list[dict[str, Any]],
    ) -> str:
        ordered_sections = [
            ("Working Memory", sections["working_memory"]),
            ("Profile", sections["profile"]),
            ("Preferences", sections["preferences"]),
            ("Relationships", sections["relationships"]),
            ("Recent Episodes", sections["recent_episodes"]),
            ("Tool Observations", sections["tool_observations"]),
        ]
        lines = [f"Intent: {intent}"]

        if context_policy == "compact":
            for title, items in ordered_sections:
                if not items:
                    continue
                compact_items: list[str] = []
                for item in items:
                    compact_item = self._compact_context_item(item)
                    if compact_item:
                        compact_items.append(compact_item)
                if compact_items:
                    lines.append(f"{title}: {'; '.join(compact_items)}")
            return flatten_text_parts(lines)

        for title, items in ordered_sections:
            if not items:
                continue
            lines.append(f"{title}:")
            for item in items:
                line = f"- {item['summary']}"
                if context_policy == "verbose":
                    line = f"{line} (memory_id={item['memory_id']})"
                lines.append(line)

        if context_policy == "verbose" and citations:
            lines.append("Citations:")
            for citation in citations[:6]:
                lines.append(f"- {citation['evidence_id']}: {citation['summary']}")
        return flatten_text_parts(lines)

    def _compact_context_item(self, item: dict[str, Any]) -> str:
        summary = normalize_text(item.get("summary", ""))
        if not summary:
            return ""
        if ":" not in summary:
            return summary
        prefix, remainder = summary.split(":", 1)
        prefix = normalize_text(prefix).lower().replace(" ", "_")
        remainder = normalize_text(remainder)
        if prefix in {"episode", "tool_result", "tool_call", "assistant_action"}:
            return remainder
        return f"{prefix}={remainder}" if remainder else prefix

    def _format_tool_events(self, tool_events: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for event in tool_events:
            text = self._event_text(event)
            if not text:
                continue
            name = event.get("name") or event.get("event_type", "tool_event")
            lines.append(f"- {name}: {summarize_text(text, max_words=18)}")
        return lines

    def _refresh_context_pack_items(self, *, state: EngineState, memory_pack: ContextPack) -> None:
        sections = (
            memory_pack.working_memory,
            memory_pack.profile,
            memory_pack.preferences,
            memory_pack.relationships,
            memory_pack.recent_episodes,
            memory_pack.tool_observations,
        )
        for items in sections:
            for item in items:
                record = state.memories.get(str(item.get("memory_id", "")))
                if record is None:
                    continue
                item["state"] = record.state

    def _context_pack_has_content(self, memory_pack: ContextPack) -> bool:
        return any(
            (
                memory_pack.working_memory,
                memory_pack.profile,
                memory_pack.preferences,
                memory_pack.relationships,
                memory_pack.recent_episodes,
                memory_pack.tool_observations,
            )
        )

    def _last_user_message(self, messages: list[dict[str, Any]]) -> str:
        for item in reversed(messages):
            if item.get("role") == "user" and item.get("content"):
                return str(item["content"])
        if messages:
            return str(messages[-1].get("content", ""))
        return ""

    def _record_payload(self, state: EngineState, record: MemoryRecord) -> dict[str, Any]:
        current_version = state.versions.get(record.current_version_id) if record.current_version_id else None
        evidence = [
            state.evidence[evidence_id].to_dict()
            for evidence_id in record.citation_evidence_ids or record.evidence_ids
            if evidence_id in state.evidence
        ]
        return {
            "memory": record.to_dict(),
            "current_version": current_version.to_dict() if current_version else None,
            "evidence": evidence,
            "history_size": len(record.version_ids),
        }
