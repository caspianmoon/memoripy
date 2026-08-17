from memoripy import AssistedMemoryExtractor, MemoryClient, MemoryPipelineConfig
from memoripy.implemented_models import OllamaChatModel

client = MemoryClient(
    pipeline=MemoryPipelineConfig(
        extractor=AssistedMemoryExtractor(OllamaChatModel(model_name="llama3.1:8b"))
    )
)

client.capture(
    messages=[{"role": "user", "content": "Please stop scheduling meetings before noon."}],
    user_id="example-user",
)

print(client.search(query="meeting constraint", user_id="example-user"))
