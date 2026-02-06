"""
RAG System with Attention Extraction
Supports both Chinese and English models, with attention extraction for visualization
"""
import torch
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Tuple, Optional
import json

class RAGSystemWithAttention:
    def __init__(self,
                 document_path: str,
                 embedding_model_id: str = 'BAAI/bge-large-en-v1.5',  # English embedding model
                 reranker_model_id: str = 'BAAI/bge-reranker-large',  # Large reranker works well for English
                 llm_model_id: str = 'Qwen/Qwen2.5-0.5B-Instruct'):  # Small Qwen model (supports English)
        """
        Initialize RAG system with attention extraction capabilities

        Args:
            document_path: Path to the document file
            embedding_model_id: Model for document embeddings
            reranker_model_id: Model for reranking retrieved documents
            llm_model_id: LLM model for generation (must support output_attentions)
        """
        print("=== RAG System with Attention Extraction ===")
        print(f"Initializing...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Load documents and build index
        self.documents = self._load_and_chunk_documents(document_path)
        print(f"✓ Loaded {len(self.documents)} document chunks")

        # Load models
        print(f"Loading embedding model: {embedding_model_id}")
        self.embedding_model = SentenceTransformer(embedding_model_id, device=self.device)

        print(f"Loading reranker model: {reranker_model_id}")
        self.reranker_model = CrossEncoder(reranker_model_id, device=self.device)

        print(f"Loading LLM: {llm_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager"  # Required for output_attentions=True
        )
        print("  ✓ Using 'eager' attention implementation for attention extraction")

        # Build vector store
        print("Building FAISS vector index...")
        self.vector_store = self._build_vector_store()
        print("✓ RAG System initialized successfully\n")

    def _load_and_chunk_documents(self, path: str) -> List[str]:
        """Load document and split into chunks by paragraphs"""
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Split by double newlines and filter empty chunks
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        return chunks

    def _build_vector_store(self):
        """Build FAISS vector index for documents"""
        # Generate embeddings for all documents
        doc_embeddings = self.embedding_model.encode(
            self.documents,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Create FAISS index
        embedding_dim = doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(doc_embeddings.astype('float32'))

        return index

    def _retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Retrieve relevant documents from vector store"""
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.vector_store.search(query_embedding, top_k)
        retrieved_docs = [self.documents[i] for i in indices[0]]
        return retrieved_docs

    def _rerank(self, query: str, docs: List[str]) -> List[str]:
        """Rerank retrieved documents using CrossEncoder"""
        pairs = [(query, doc) for doc in docs]
        scores = self.reranker_model.predict(pairs)
        doc_scores = list(zip(docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in doc_scores]
        return reranked_docs

    def _clean_tokens_for_visualization(self, token_ids: torch.Tensor) -> List[str]:
        """
        Convert token IDs to readable strings for visualization

        Args:
            token_ids: Tensor of token IDs

        Returns:
            List of readable token strings
        """
        readable_tokens = []

        for token_id in token_ids:
            # Try to decode single token
            try:
                # Decode the token ID to text
                token_text = self.tokenizer.decode([token_id.item()], skip_special_tokens=False)

                # Clean up the token
                token_text = token_text.strip()

                # Handle empty or whitespace tokens
                if not token_text or token_text.isspace():
                    token_text = "▁"  # Use underscore to represent space

                # Limit token length for visualization
                if len(token_text) > 15:
                    token_text = token_text[:12] + "..."

                readable_tokens.append(token_text)

            except Exception as e:
                # Fallback to token ID if decoding fails
                readable_tokens.append(f"[{token_id.item()}]")

        return readable_tokens

    def _generate_with_attention(self,
                                   query: str,
                                   context: Optional[str] = None,
                                   max_new_tokens: int = 256) -> Dict:
        """
        Generate answer and extract attention weights

        Returns:
            dict with keys:
                - answer: str
                - tokens: List[str]
                - attentions: torch.Tensor (layers, heads, seq_len, seq_len)
                - input_ids: torch.Tensor
                - token_type_ids: List[str] - indicates which part each token belongs to
                - context_range: tuple (start, end) - range of context tokens in sequence
        """
        # Build prompt and track positions
        context_marker_start = "--- Known Information ---\n"
        context_marker_end = "\n---"

        if context:
            prompt = f"""You are an intelligent Q&A bot. Please answer the user's question based *only* on the "Known Information" provided below.
Do not make up or add any external knowledge.
If the known information cannot answer the question, please say "I cannot answer this question based on the known information."

{context_marker_start}{context}{context_marker_end}

Question: {query}

Answer:"""
        else:
            prompt = f"""Please answer the following question:

Question: {query}

Answer:"""

        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        input_length = model_inputs.input_ids.shape[1]

        # Find context position in tokenized sequence
        context_range = None
        if context:
            # Tokenize the context part to find its position
            context_with_markers = f"{context_marker_start}{context}{context_marker_end}"
            context_tokens = self.tokenizer.encode(context_with_markers, add_special_tokens=False)

            # Search for context tokens in the full input
            full_input_ids = model_inputs.input_ids[0].tolist()
            context_len = len(context_tokens)

            for i in range(len(full_input_ids) - context_len + 1):
                if full_input_ids[i:i+context_len] == context_tokens:
                    context_range = (i, i + context_len)
                    print(f"  ✓ Found RAG context at token positions: {context_range}")
                    break

        # Generate with attention output
        with torch.no_grad():
            outputs = self.llm.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                output_attentions=True,
                return_dict_in_generate=True,
                temperature=0.1,
                do_sample=False
            )

        # Extract generated text
        generated_ids = outputs.sequences[0][input_length:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Get all tokens (input + generated) - convert to readable format
        all_tokens = self._clean_tokens_for_visualization(outputs.sequences[0])

        # Create token type indicators
        token_type_ids = []
        for i in range(len(all_tokens)):
            if i < input_length:
                if context_range and context_range[0] <= i < context_range[1]:
                    token_type_ids.append("context")
                else:
                    token_type_ids.append("prompt")
            else:
                token_type_ids.append("generated")

        print(f"  Token breakdown: {token_type_ids.count('prompt')} prompt, {token_type_ids.count('context')} context, {token_type_ids.count('generated')} generated")

        # Process attention weights
        # For autoregressive generation, outputs.attentions contains per-step attentions
        # which have shape [layers, batch, heads, 1, past_length]
        # We need to do a forward pass on the complete sequence to get full attention matrix
        attention_stack = None
        try:
            print(f"  Running forward pass on complete sequence to extract full attention...")
            with torch.no_grad():
                # Forward pass on the complete generated sequence
                forward_outputs = self.llm(
                    input_ids=outputs.sequences,
                    output_attentions=True,
                    return_dict=True
                )

            if hasattr(forward_outputs, 'attentions') and forward_outputs.attentions:
                # Stack all layers (each is [batch, heads, seq_len, seq_len])
                # Take first (and only) batch
                attention_stack = torch.stack([layer_attn[0] for layer_attn in forward_outputs.attentions])
                # Shape: (num_layers, num_heads, seq_len, seq_len)
                print(f"  ✓ Extracted full attention shape: {attention_stack.shape}")
            else:
                print("  Warning: No attention weights in forward output")
        except Exception as e:
            print(f"  Warning: Failed to extract attention: {e}")
            import traceback
            traceback.print_exc()
            attention_stack = None

        return {
            'answer': answer,
            'tokens': all_tokens if all_tokens else [],
            'attentions': attention_stack,
            'input_ids': outputs.sequences[0],
            'prompt': prompt,
            'input_length': input_length,
            'token_type_ids': token_type_ids,
            'context_range': context_range
        }

    def query(self,
              query: str,
              k_retrieve: int = 10,
              k_context: int = 3,
              use_rag: bool = True) -> Dict:
        """
        Main query function that returns both RAG and non-RAG results

        Args:
            query: User question
            k_retrieve: Number of documents to retrieve initially
            k_context: Number of documents to use as context
            use_rag: Whether to use RAG or just query LLM directly

        Returns:
            dict with results and attention weights
        """
        result = {
            'query': query,
            'use_rag': use_rag,
            'retrieved_docs': [],
            'context': None
        }

        if use_rag:
            # RAG pipeline
            print(f"[RAG Mode] Retrieving documents...")
            retrieved_docs = self._retrieve(query, top_k=k_retrieve)
            print(f"  Retrieved {len(retrieved_docs)} documents")

            print(f"[RAG Mode] Reranking...")
            reranked_docs = self._rerank(query, retrieved_docs)

            # Select top k for context
            context_docs = reranked_docs[:k_context]
            context = "\n\n".join(context_docs)

            result['retrieved_docs'] = reranked_docs
            result['context'] = context
            result['context_docs_used'] = context_docs

            print(f"[RAG Mode] Generating answer with context...")
            generation_result = self._generate_with_attention(query, context)
        else:
            # Direct LLM query without RAG
            print(f"[No RAG Mode] Generating answer without context...")
            generation_result = self._generate_with_attention(query, context=None)

        # Merge generation results into main result
        result.update(generation_result)

        return result

    def query_with_comparison(self, query: str, k_retrieve: int = 10, k_context: int = 3) -> Dict:
        """
        Query with both RAG and non-RAG for comparison

        Returns:
            dict with both results for comparison
        """
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}\n")

        # Get non-RAG result
        print("1. Querying WITHOUT RAG...")
        no_rag_result = self.query(query, use_rag=False)

        print()

        # Get RAG result
        print("2. Querying WITH RAG...")
        rag_result = self.query(query, k_retrieve=k_retrieve, k_context=k_context, use_rag=True)

        print(f"\n{'='*60}")
        print("Results Summary:")
        print(f"  Without RAG: {no_rag_result['answer'][:100]}...")
        print(f"  With RAG: {rag_result['answer'][:100]}...")
        print(f"{'='*60}\n")

        return {
            'query': query,
            'no_rag': no_rag_result,
            'with_rag': rag_result,
            'parameters': {
                'k_retrieve': k_retrieve,
                'k_context': k_context
            }
        }


if __name__ == '__main__':
    # Test the RAG system
    import sys
    sys.path.append('..')

    rag_system = RAGSystemWithAttention(
        document_path='../data/story.txt'
    )

    # Load questions
    with open('../data/questions.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # Test with first question
    test_question = questions[0]['question']
    result = rag_system.query_with_comparison(test_question, k_context=3)

    print("\nAttention shapes:")
    if result['no_rag']['attentions'] is not None:
        print(f"  No RAG: {result['no_rag']['attentions'].shape}")
    if result['with_rag']['attentions'] is not None:
        print(f"  With RAG: {result['with_rag']['attentions'].shape}")
