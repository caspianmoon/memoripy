import numpy as np
import time
import uuid
import os
import torch
from pydantic import BaseModel, Field
from .in_memory_storage import InMemoryStorage
from langchain_core.messages import HumanMessage, SystemMessage
from .memory_store import MemoryStore
from .model import ChatModel, EmbeddingModel
from sentence_transformers import SentenceTransformer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

class ConceptExtractionResponse(BaseModel):
    concepts: list[str] = Field(description="List of key concepts extracted from the text.")

class MemoryManager:
    """
    Manages the memory store, including loading and saving history,
    adding interactions, retrieving relevant interactions, and generating responses.
    """

    def __init__(self, chat_model: ChatModel, storage=None):
        # Initialize the model with specific device placement
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2',
            trust_remote_code=True,
            device=device
        )
        self.chat_model = chat_model
        self.device = device

        # Initialize memory store with the correct dimension
        self.dimension = 768
        self.memory_store = MemoryStore(dimension=self.dimension)

        if storage is None:
            self.storage = InMemoryStorage()
        else:
            self.storage = storage

        self.initialize_memory()

    def __del__(self):
        """
        Cleanup when the MemoryManager is destroyed
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def standardize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Standardize embedding to the target dimension by padding with zeros or truncating.
        """
        current_dim = len(embedding)
        if current_dim == self.dimension:
            return embedding
        elif current_dim < self.dimension:
            return np.pad(embedding, (0, self.dimension - current_dim), 'constant')
        else:
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

    def get_embedding(self, text: str, max_tokens: int = 768, stride: int = 192) -> np.ndarray:
        """
        Get embedding using sliding window approach with proper resource management
        """
        print(f"Generating embedding for the provided text...")
        try:
            words = text.split()
            embeddings = []
            
            # Use sliding window if text is long
            if len(words) > max_tokens:
                for i in range(0, len(words) - max_tokens + 1, stride):
                    window = ' '.join(words[i:i + max_tokens])
                    print(f"Processing window {len(embeddings)+1}: tokens {i} to {i + max_tokens}")
                    
                    with torch.no_grad():  # Disable gradient computation
                        window_embedding = self.embedding_model.encode(
                            window,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                            batch_size=1  # Process one at a time
                        )
                    embeddings.append(window_embedding)
                    
                    # Clear cache if using GPU
                    if self.device in ["cuda", "mps"]:
                        torch.cuda.empty_cache() if self.device == "cuda" else torch.mps.empty_cache()
                
                # Average all window embeddings
                embedding = np.mean(embeddings, axis=0)
            else:
                with torch.no_grad():
                    embedding = self.embedding_model.encode(
                        text,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=1
                    )
            
            if embedding is None:
                raise ValueError("Failed to generate embedding.")
                
            standardized_embedding = self.standardize_embedding(embedding)
            return np.array(standardized_embedding).reshape(1, -1)
            
        except Exception as e:
            print(f"Error in get_embedding: {e}")
            raise
        finally:
            # Clean up resources
            if self.device in ["cuda", "mps"]:
                torch.cuda.empty_cache() if self.device == "cuda" else torch.mps.empty_cache()

    def extract_concepts(self, text: str) -> list[str]:
        print("Extracting key concepts from the provided text...")
        return self.chat_model.extract_concepts(text)

    def initialize_memory(self):
        short_term, long_term = self.load_history()
        for interaction in short_term:
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