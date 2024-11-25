import numpy as np
import time
import uuid
from pydantic import BaseModel, Field
from .in_memory_storage import InMemoryStorage
from langchain_core.messages import HumanMessage, SystemMessage
from .memory_store import MemoryStore
from .model import ChatModel, EmbeddingModel
from sentence_transformers import SentenceTransformer


class ConceptExtractionResponse(BaseModel):
    concepts: list[str] = Field(description="List of key concepts extracted from the text.")


class SentenceTransformerEmbedding(EmbeddingModel):
    """
    Implementation of EmbeddingModel using SentenceTransformers
    """
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text)
    
    def initialize_embedding_dimension(self) -> int:
        # all-mpnet-base-v2 has 768 dimensions
        return 768


class MemoryManager:
    """
    Manages the memory store, including loading and saving history,
    adding interactions, retrieving relevant interactions, and generating responses.
    """

    def __init__(self, chat_model: ChatModel, embedding_model: EmbeddingModel = None, storage=None):
        embedding_model = SentenceTransformerEmbedding()
        self.chat_model = chat_model
        self.embedding_model = embedding_model

        # Initialize memory store with the correct dimension
        self.dimension = self.embedding_model.initialize_embedding_dimension()
        self.memory_store = MemoryStore(dimension=self.dimension)

        if storage is None:
            self.storage = InMemoryStorage()
        else:
            self.storage = storage

        self.initialize_memory()

    def standardize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Standardize embedding to the target dimension by padding with zeros or truncating.
        """
        current_dim = len(embedding)
        if current_dim == self.dimension:
            return embedding
        elif current_dim < self.dimension:
            # Pad with zeros
            return np.pad(embedding, (0, self.dimension - current_dim), 'constant')
        else:
            # Truncate to match target dimension
            return embedding[:self.dimension]

    def load_history(self):
        return self.storage.load_history()

    def save_memory_to_history(self):
        self.storage.save_memory_to_history(self.memory_store)

    def add_interaction(self, prompt: str, output: str, embedding: np.ndarray, concepts: list[str]):
        timestamp = time.time()
        interaction_id = str(uuid.uuid4())
        interaction = {
            "id": interaction_id,
            "prompt": prompt,
            "output": output,
            "embedding": embedding.tolist(),
            "timestamp": timestamp,
            "access_count": 1,
            "concepts": list(concepts),
            "decay_factor": 1.0,
        }
        self.memory_store.add_interaction(interaction)
        self.save_memory_to_history()

    def get_embedding(self, text: str) -> np.ndarray:
        print(f"Generating embedding for the provided text...")
        embedding = self.embedding_model.get_embedding(text)
        if embedding is None:
            raise ValueError("Failed to generate embedding.")
        standardized_embedding = self.standardize_embedding(embedding)
        return np.array(standardized_embedding).reshape(1, -1)

    def extract_concepts(self, text: str) -> list[str]:
        print("Extracting key concepts from the provided text...")
        return self.chat_model.extract_concepts(text)

    def initialize_memory(self):
        short_term, long_term = self.load_history()
        for interaction in short_term:
            # Standardize the dimension of each interaction's embedding
            interaction['embedding'] = self.standardize_embedding(np.array(interaction['embedding']))
            self.memory_store.add_interaction(interaction)
        self.memory_store.long_term_memory.extend(long_term)

        self.memory_store.cluster_interactions()
        print(f"Memory initialized with {len(self.memory_store.short_term_memory)} interactions in short-term and {len(self.memory_store.long_term_memory)} in long-term.")

    def retrieve_relevant_interactions(self, query: str, similarity_threshold=40, exclude_last_n=0) -> list:
        query_embedding = self.get_embedding(query)
        query_concepts = self.extract_concepts(query)
        return self.memory_store.retrieve(query_embedding, query_concepts, similarity_threshold, exclude_last_n=exclude_last_n)

    def generate_response(self, prompt: str, last_interactions: list, retrievals: list, context_window=3) -> str:
        context = ""
        if last_interactions:
            context_interactions = last_interactions[-context_window:]
            context += "\n".join([f"Previous prompt: {r['prompt']}\nPrevious output: {r['output']}" for r in context_interactions])
            print(f"Using the following last interactions as context for response generation:\n{context}")
        else:
            context = "No previous interactions available."
            print(context)

        if retrievals:
            retrieved_context_interactions = retrievals[:context_window]
            retrieved_context = "\n".join([f"Relevant prompt: {r['prompt']}\nRelevant output: {r['output']}" for r in retrieved_context_interactions])
            print(f"Using the following retrieved interactions as context for response generation:\n{retrieved_context}")
            context += "\n" + retrieved_context

        messages = [
            SystemMessage(content="You're a helpful assistant."),
            HumanMessage(content=f"{context}\nCurrent prompt: {prompt}")
        ]
        
        response = self.chat_model.invoke(messages)

        return response
