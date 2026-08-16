from .admission import AdmissionConfig, AdmissionPolicy, DefaultAdmissionPolicy
from .client import AsyncMemoryClient, Memory, MemoryClient
from .implemented_models import (
    AzureOpenAIChatModel,
    AzureOpenAIEmbeddingModel,
    ChatCompletionsModel,
    EchoChatModel,
    OllamaChatModel,
    OllamaEmbeddingModel,
    OpenAIChatModel,
    OpenAIEmbeddingModel,
    OpenRouterChatModel,
    SimpleKeywordEmbeddingModel,
)
from .model import ChatModel, EmbeddingModel
from .pipeline import (
    AssetProcessor,
    BrainConfig,
    DefaultMemoryReconciler,
    KeywordBoostReranker,
    LocalAssetProcessor,
    MemoryPipelineConfig,
    MemoryReconciler,
    ReconciliationDecision,
    RetrievalConfig,
    RerankOutcome,
    Reranker,
    SemanticExtractor,
)
from .repository import (
    FileMemoryRepository,
    InMemoryRepository,
    MemoryCorruptionError,
    MemoryRepositoryError,
)
from .service import MemoryService, create_fastapi_app, serve_http
from .types import (
    AdmissionDecision,
    AdmissionReason,
    AuditFinding,
    AuditReport,
    ContextPack,
    Durability,
    EvidenceItem,
    EventType,
    IngestionItem,
    MemoryAction,
    MemoryKind,
    MemoryLayer,
    MemoryRecord,
    MemoryScope,
    MemoryState,
    MemoryVersion,
    Modality,
    ProjectionStatus,
    RelationEdge,
    RetrievalReceipt,
    SearchFilters,
    SearchResult,
    SourceType,
    TrustLevel,
)

__version__ = "0.4.0"

try:
    from .postgres_repository import PostgresRepository
except (ImportError, RuntimeError):
    PostgresRepository = None  # type: ignore[assignment]

try:
    from .in_memory_storage import InMemoryStorage
    from .json_storage import JSONStorage
    from .memory_manager import MemoryManager
    from .memory_store import MemoryStore
    from .storage import BaseStorage
except ImportError:
    BaseStorage = None  # type: ignore[assignment]
    InMemoryStorage = None  # type: ignore[assignment]
    JSONStorage = None  # type: ignore[assignment]
    MemoryManager = None  # type: ignore[assignment]
    MemoryStore = None  # type: ignore[assignment]

__all__ = [
    "AdmissionConfig",
    "AdmissionDecision",
    "AdmissionPolicy",
    "AdmissionReason",
    "AssetProcessor",
    "AsyncMemoryClient",
    "AuditFinding",
    "AuditReport",
    "AzureOpenAIChatModel",
    "AzureOpenAIEmbeddingModel",
    "BaseStorage",
    "BrainConfig",
    "ChatCompletionsModel",
    "ChatModel",
    "ContextPack",
    "DefaultAdmissionPolicy",
    "DefaultMemoryReconciler",
    "Durability",
    "EchoChatModel",
    "EmbeddingModel",
    "EvidenceItem",
    "EventType",
    "FileMemoryRepository",
    "InMemoryRepository",
    "InMemoryStorage",
    "IngestionItem",
    "JSONStorage",
    "KeywordBoostReranker",
    "LocalAssetProcessor",
    "Memory",
    "MemoryAction",
    "MemoryClient",
    "MemoryCorruptionError",
    "MemoryKind",
    "MemoryLayer",
    "MemoryManager",
    "MemoryPipelineConfig",
    "MemoryReconciler",
    "MemoryRecord",
    "MemoryRepositoryError",
    "MemoryScope",
    "MemoryService",
    "MemoryState",
    "MemoryStore",
    "MemoryVersion",
    "Modality",
    "OllamaChatModel",
    "OllamaEmbeddingModel",
    "OpenAIChatModel",
    "OpenAIEmbeddingModel",
    "OpenRouterChatModel",
    "PostgresRepository",
    "ProjectionStatus",
    "ReconciliationDecision",
    "RelationEdge",
    "RetrievalConfig",
    "RetrievalReceipt",
    "RerankOutcome",
    "Reranker",
    "SearchFilters",
    "SearchResult",
    "SemanticExtractor",
    "SimpleKeywordEmbeddingModel",
    "SourceType",
    "TrustLevel",
    "create_fastapi_app",
    "serve_http",
]
