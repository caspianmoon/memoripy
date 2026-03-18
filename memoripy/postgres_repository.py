from __future__ import annotations

from typing import Any, Callable

from .repository import BaseRepository, EngineState
from .utils import deep_copy_json, utc_now


class PostgresRepository(BaseRepository):
    """
    SQL-backed repository that persists the full engine state and keeps a
    projection table with embeddings for future direct SQL retrieval.

    The implementation lazily imports SQLAlchemy/pgvector so the base package
    stays stdlib-only unless the postgres extras are installed.
    """

    def __init__(
        self,
        database_url: str,
        *,
        table_prefix: str = "memoripy",
        vector_dimensions: int = 32,
    ):
        try:
            import sqlalchemy as sa
            from sqlalchemy import JSON, Column, DateTime, Integer, MetaData, String, Table, Text, create_engine, select
        except ImportError as exc:
            raise RuntimeError("SQLAlchemy is not installed. Install memoripy with the postgres extras.") from exc

        try:
            from pgvector.sqlalchemy import Vector
        except ImportError:
            Vector = None

        self._sa = sa
        self._select = select
        self._engine = create_engine(database_url, future=True)
        self._metadata = MetaData()
        self._vector_dimensions = vector_dimensions
        self._vector_enabled = Vector is not None and database_url.startswith("postgres")

        self._state_table = Table(
            f"{table_prefix}_state",
            self._metadata,
            Column("id", Integer, primary_key=True),
            Column("payload", JSON, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )
        self._events_table = Table(
            f"{table_prefix}_events",
            self._metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("operation", String(64), nullable=False),
            Column("idempotency_key", String(255), nullable=True),
            Column("payload", JSON, nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
        )

        embedding_column = Column("embedding_json", JSON, nullable=True)
        if self._vector_enabled:
            embedding_column = Column("embedding", Vector(vector_dimensions), nullable=True)

        self._projection_table = Table(
            f"{table_prefix}_memory_projection",
            self._metadata,
            Column("record_id", String(64), primary_key=True),
            Column("user_id", String(255), nullable=True),
            Column("agent_id", String(255), nullable=True),
            Column("run_id", String(255), nullable=True),
            Column("kind", String(64), nullable=False),
            Column("key", String(255), nullable=False),
            Column("layer", String(64), nullable=False),
            Column("state", String(64), nullable=False),
            Column("summary", Text, nullable=False),
            Column("value", Text, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
            embedding_column,
            Column("payload", JSON, nullable=False),
        )
        self._metadata.create_all(self._engine)

    def load_state(self) -> EngineState:
        with self._engine.begin() as connection:
            row = connection.execute(self._select(self._state_table.c.payload).where(self._state_table.c.id == 1)).first()
            if row is None:
                return EngineState()
            return EngineState.from_dict(row[0])

    def transaction(
        self,
        operation_name: str,
        idempotency_key: str | None,
        operation: Callable[[EngineState], tuple[Any, list[dict[str, Any]]]],
    ) -> Any:
        with self._engine.begin() as connection:
            state = self._load_state_for_update(connection)
            key = f"{operation_name}:{idempotency_key}" if idempotency_key else None
            if key and key in state.idempotency:
                return deep_copy_json(state.idempotency[key]["result"])

            result, events = operation(state)
            timestamp = self._sa.func.now()
            for event in events:
                connection.execute(
                    self._events_table.insert().values(
                        operation=operation_name,
                        idempotency_key=idempotency_key,
                        payload=deep_copy_json(event),
                        created_at=timestamp,
                    )
                )
            if key:
                state.idempotency[key] = {"result": deep_copy_json(result), "events": deep_copy_json(events)}
            self._upsert_state(connection, state)
            self._sync_projection_table(connection, state)
            return deep_copy_json(result)

    def replace_state(
        self,
        state: EngineState,
        operation_name: str = "replace_state",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        with self._engine.begin() as connection:
            current_state = self._load_state_for_update(connection)
            key = f"{operation_name}:{idempotency_key}" if idempotency_key else None
            if key and key in current_state.idempotency:
                return deep_copy_json(current_state.idempotency[key]["result"])

            result = {
                "status": "ok",
                "operation": operation_name,
                "schema_version": state.schema_version,
                "replaced_at": utc_now(),
            }
            if key:
                state.idempotency[key] = {"result": deep_copy_json(result), "events": []}
            connection.execute(
                self._events_table.insert().values(
                    operation=operation_name,
                    idempotency_key=idempotency_key,
                    payload={"type": "state_replaced", "schema_version": state.schema_version},
                    created_at=self._sa.func.now(),
                )
            )
            self._upsert_state(connection, state)
            self._sync_projection_table(connection, state)
            return result

    def _load_state_for_update(self, connection: Any) -> EngineState:
        row = connection.execute(
            self._select(self._state_table.c.payload).where(self._state_table.c.id == 1).with_for_update()
        ).first()
        if row is None:
            return EngineState()
        return EngineState.from_dict(row[0])

    def _upsert_state(self, connection: Any, state: EngineState) -> None:
        payload = state.to_dict()
        existing = connection.execute(self._select(self._state_table.c.id).where(self._state_table.c.id == 1)).first()
        if existing is None:
            connection.execute(
                self._state_table.insert().values(
                    id=1,
                    payload=payload,
                    updated_at=self._sa.func.now(),
                )
            )
            return
        connection.execute(
            self._state_table.update()
            .where(self._state_table.c.id == 1)
            .values(payload=payload, updated_at=self._sa.func.now())
        )

    def _sync_projection_table(self, connection: Any, state: EngineState) -> None:
        connection.execute(self._projection_table.delete())
        for record in state.memories.values():
            payload = record.to_dict()
            values = {
                "record_id": record.record_id,
                "user_id": record.scope.user_id,
                "agent_id": record.scope.agent_id,
                "run_id": record.scope.run_id,
                "kind": record.kind,
                "key": record.key,
                "layer": record.layer,
                "state": record.state,
                "summary": record.summary,
                "value": record.value,
                "updated_at": self._sa.func.now(),
                "payload": payload,
            }
            if self._vector_enabled:
                values["embedding"] = list(record.embedding or []) or None
            else:
                values["embedding_json"] = list(record.embedding or [])
            connection.execute(self._projection_table.insert().values(**values))
