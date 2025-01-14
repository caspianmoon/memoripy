import os

from dotenv import load_dotenv
from pynamodb.attributes import (
    UnicodeAttribute,
    NumberAttribute,
    ListAttribute,
    MapAttribute,
)
from pynamodb.models import Model
from pydantic import BaseModel
from memoripy import BaseStorage
from memoripy.memory_store import MemoryStore

load_dotenv()


def _get_host() -> str | None:
    return os.environ.get("MEMORIPY_DYNAMO_HOST", None)


def _get_region() -> str:
    return os.environ.get("MEMORIPY_DYNAMO_REGION", "us-east-1")


def _get_read_capacity() -> str:
    return os.environ.get("MEMORIPY_DYNAMO_READ_CAPACITY", "1")


def _get_write_capacity() -> str:
    return os.environ.get("MEMORIPY_DYNAMO_WRITE_CAPACITY", "1")


class ShortTermMemoryAttr(MapAttribute):
    id = UnicodeAttribute()
    prompt = UnicodeAttribute()
    output = UnicodeAttribute()
    timestamp = NumberAttribute()
    access_count = NumberAttribute()
    decay_factor = NumberAttribute()
    embedding = ListAttribute(of=NumberAttribute)
    concepts = ListAttribute(of=UnicodeAttribute)


class ShortTermMemory(BaseModel):
    id: str
    prompt: str
    output: str
    timestamp: float
    access_count: int
    decay_factor: float
    embedding: list[float]
    concepts: list[str]

    def get(self, key, default):
        return getattr(self, key, default)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @classmethod
    def new_from_attr(cls, attr: ShortTermMemoryAttr):
        return cls(
            id=attr.id,
            prompt=attr.prompt,
            output=attr.output,
            timestamp=attr.timestamp,
            access_count=int(attr.access_count),
            decay_factor=attr.decay_factor,
            embedding=[float(x) for x in attr.embedding],
            concepts=[str(x) for x in attr.concepts],
        )


class LongTermTermMemoryAttr(MapAttribute):
    id = UnicodeAttribute()
    prompt = UnicodeAttribute()
    output = UnicodeAttribute()
    timestamp = NumberAttribute()
    access_count = NumberAttribute()
    decay_factor = NumberAttribute()
    total_score = NumberAttribute()


class LongTermMemory(BaseModel):
    id: str
    prompt: str
    output: str
    timestamp: float
    access_count: int
    decay_factor: float
    total_score: float

    def get(self, key, default):
        return getattr(self, key, default)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @classmethod
    def new_from_attr(cls, attr: LongTermTermMemoryAttr):
        return cls(
            id=attr.id,
            prompt=attr.prompt,
            output=attr.output,
            timestamp=attr.timestamp,
            access_count=int(attr.access_count),
            decay_factor=attr.decay_factor,
            total_score=attr.total_score,
        )


class Memory(Model):
    class Meta:
        table_name = "memoripy_memory"
        region = _get_region()
        host = _get_host()

    set_id = UnicodeAttribute(hash_key=True)
    short_term_memory = ListAttribute(of=ShortTermMemoryAttr)
    long_term_memory = ListAttribute(of=LongTermTermMemoryAttr)


class DynamoStorage(BaseStorage):
    """Leverage DynamoDB for storage of memory interactions."""

    def __init__(self, set_id: str):
        """
        Create an instance of DynamoStorage
        Args:
            set_id: A unique identifier for the memory set - This should be
            leveraged when different memory sets make sense (e.g. when you're
            working in a multi-user system). For example, if you had a chatbot
            that handled multiple users, you would need to instantiate multiple
            instances of MemoryManager (one for each user), and you could use
            `set_id` property on DynamoStorage to indicate which user the memory
            was associated with.
        """
        if not Memory.exists():
            Memory.create_table(
                read_capacity_units=int(_get_read_capacity()),
                write_capacity_units=int(_get_write_capacity()),
                wait=True,
            )
        self.set_id = set_id

    def load_history(self):
        try:
            memory = Memory.get(self.set_id)
            short_term_memory = [
                ShortTermMemory.new_from_attr(attr) for attr in memory.short_term_memory
            ]
            long_term_memory = [
                LongTermMemory.new_from_attr(attr) for attr in memory.long_term_memory
            ]
            return short_term_memory, long_term_memory
        except Memory.DoesNotExist:
            return [], []

    def save_memory_to_history(self, memory_store: MemoryStore):
        history = Memory(
            set_id=self.set_id,
            short_term_memory=[],
            long_term_memory=[],
        )

        for idx in range(len(memory_store.short_term_memory)):
            interaction = ShortTermMemoryAttr(
                id=memory_store.short_term_memory[idx]["id"],
                prompt=memory_store.short_term_memory[idx]["prompt"],
                output=memory_store.short_term_memory[idx]["output"],
                embedding=memory_store.embeddings[idx].flatten().tolist(),
                timestamp=memory_store.timestamps[idx],
                access_count=memory_store.access_counts[idx],
                concepts=list(memory_store.concepts_list[idx]),
                decay_factor=memory_store.short_term_memory[idx].get(
                    "decay_factor", 1.0
                ),
            )
            history.short_term_memory.append(interaction)

        for memory in memory_store.long_term_memory:
            interaction = LongTermTermMemoryAttr(
                id=memory["id"],
                prompt=memory["prompt"],
                output=memory["output"],
                timestamp=memory["timestamp"],
                access_count=memory["access_count"],
                decay_factor=memory["decay_factor"],
                total_score=memory["total_score"],
            )
            history.long_term_memory.append(interaction)

        history.save()
        print(
            f"Saved interaction history to JSON. Short-term: {len(history.short_term_memory)}, Long-term: {len(history.long_term_memory)}"
        )
