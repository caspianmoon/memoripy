from .admission import AdmissionConfig, AdmissionPolicy, DefaultAdmissionPolicy
from .assisted import AssistedExtractionConfig, AssistedMemoryExtractor, StructuredExtractionError
from .client import AsyncMemoryClient, Memory, MemoryClient
from .comparisons import (
    GraphitiComparisonAdapter,
    HindsightComparisonAdapter,
    LangMemComparisonAdapter,
    Mem0ComparisonAdapter,
    MemoripyComparisonAdapter,
    run_comparison,
)
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
from .gateway import TenantMemoryGateway, serve_gateway
from .inspector import inspector_html, serve_inspector
from .mcp_server import MCPAccessPolicy, MemoripyMCPTools, build_mcp_server, run_mcp_server
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
from .temporal import TemporalBounds, infer_temporal_bounds
from .tenant import (
    ADMIN_SCOPE,
    READ_SCOPE,
    WRITE_SCOPE,
    TenantPrincipal,
    TenantRegistry,
    TenantStoreManager,
)
from .tuning import (
    RetrievalProfile,
    TuningResult,
    load_retrieval_profile,
    save_retrieval_profile,
    tune_retrieval,
)
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
    "ADMIN_SCOPE",
    "AssistedExtractionConfig",
    "AssistedMemoryExtractor",
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
    "GraphitiComparisonAdapter",
    "HindsightComparisonAdapter",
    "LangMemComparisonAdapter",
    "MCPAccessPolicy",
    "Mem0ComparisonAdapter",
    "MemoripyComparisonAdapter",
    "MemoripyMCPTools",
    "READ_SCOPE",
    "RetrievalProfile",
    "StructuredExtractionError",
    "TemporalBounds",
    "TenantMemoryGateway",
    "TenantPrincipal",
    "TenantRegistry",
    "TenantStoreManager",
    "TuningResult",
    "WRITE_SCOPE",
    "build_mcp_server",
    "infer_temporal_bounds",
    "inspector_html",
    "load_retrieval_profile",
    "run_comparison",
    "run_mcp_server",
    "save_retrieval_profile",
    "serve_gateway",
    "serve_inspector",
    "tune_retrieval",
    "create_fastapi_app",
    "serve_http",
]
