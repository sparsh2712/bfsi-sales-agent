"""
Enhanced knowledge base with sentence embeddings for the BFSI Sales Agent
"""
import os
import logging
import json
import re
import PyPDF2
import numpy as np
from pathlib import Path
import pickle
from tenacity import retry, stop_after_attempt, wait_exponential

# New imports for NLP-driven relevancy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class KnowledgeBase:
    """
    Enhanced knowledge base using sentence embeddings for semantic search.
    Provides more accurate relevancy determination compared to keyword matching.
    """
    
    def __init__(self, config):
        """
        Initialize the knowledge base
        
        Args:
            config (dict): Knowledge base configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configure paths
        self.docs_dir = config.get("docs_dir", "data/documents")
        self.index_dir = config.get("index_dir", "data/indices")
        self.cache_results = config.get("cache_results", True)
        
        # Ensure directories exist
        Path(self.docs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        
        # Document data storage
        self.documents = {}
        self.sections = []
        self.section_embeddings = []
        
        # Initialize sentence transformer model
        self.model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.logger.info(f"Initializing sentence embedding model: {self.model_name}")
        
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            self.logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading sentence transformer model: {str(e)}")
            self.logger.warning("Falling back to keyword-based matching")
            self.embedding_model = None
        
        # Load and process documents
        self._load_documents()
        
        # Generate embeddings for all sections if model is available
        if self.embedding_model and self.sections:
            self._generate_embeddings()
    
    def _load_documents(self):
        """Load and process all documents in the docs directory"""
        self.logger.info(f"Loading documents from {self.docs_dir}")
        
        # Check if we have a cached index
        cache_path = os.path.join(self.index_dir, "knowledge_index.pkl")
        if self.cache_results and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.documents = cached_data.get('documents', {})
                    self.sections = cached_data.get('sections', [])
                    self.section_embeddings = cached_data.get('embeddings', [])
                
                self.logger.info(f"Loaded cached index with {len(self.documents)} documents and {len(self.sections)} sections")
                
                # If embeddings aren't in the cache or of different length than sections, regenerate them
                if not self.section_embeddings or len(self.section_embeddings) != len(self.sections):
                    if self.embedding_model:
                        self._generate_embeddings()
                
                return
            except Exception as e:
                self.logger.error(f"Error loading cached index: {str(e)}")
                self.logger.info("Processing documents from scratch")
        
        for filename in os.listdir(self.docs_dir):
            filepath = os.path.join(self.docs_dir, filename)
            
            try:
                if filename.endswith('.pdf'):
                    self._process_pdf(filepath)
                elif filename.endswith('.txt'):
                    self._process_text(filepath)
                elif filename.endswith('.json'):
                    self._process_json(filepath)
                else:
                    self.logger.warning(f"Unsupported file format: {filename}")
            except Exception as e:
                self.logger.error(f"Error processing document {filename}: {str(e)}")
        
        self.logger.info(f"Loaded {len(self.documents)} documents with {len(self.sections)} sections")
        
        # Cache the processed documents if enabled
        if self.cache_results:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'documents': self.documents,
                        'sections': self.sections,
                        'embeddings': self.section_embeddings
                    }, f)
                self.logger.info(f"Cached knowledge index to {cache_path}")
            except Exception as e:
                self.logger.error(f"Error caching knowledge index: {str(e)}")
    
    def _process_pdf(self, filepath):
        """
        Process a PDF document
        
        Args:
            filepath (str): Path to the PDF file
        """
        self.logger.info(f"Processing PDF: {filepath}")
        filename = os.path.basename(filepath)
        
        # Extract text from PDF
        text = ""
        with open(filepath, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        # Store the full document
        self.documents[filename] = text
        
        # Split into meaningful chunks (improved from simple paragraph splitting)
        self._extract_sections(text, filename, chunk_size=500, overlap=100)
    
    def _process_text(self, filepath):
        """
        Process a text document
        
        Args:
            filepath (str): Path to the text file
        """
        self.logger.info(f"Processing text: {filepath}")
        filename = os.path.basename(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Store the full document
        self.documents[filename] = text
        
        # Split into meaningful chunks
        self._extract_sections(text, filename, chunk_size=500, overlap=100)
    
    def _process_json(self, filepath):
        """
        Process a JSON document (expected to contain QA pairs)
        
        Args:
            filepath (str): Path to the JSON file
        """
        self.logger.info(f"Processing JSON: {filepath}")
        filename = os.path.basename(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Expect a list of QA pairs or a dict with sections
        if isinstance(data, list):
            for i, item in enumerate(data):
                if "question" in item and "answer" in item:
                    # Create a more searchable representation
                    content = f"Q: {item['question']}\nA: {item['answer']}"
                    
                    # Add enhanced metadata
                    metadata = {
                        "question": item["question"],
                        "answer": item["answer"],
                        "type": "qa_pair"
                    }
                    
                    # Extract keywords still useful for filtering
                    keywords = self._extract_keywords(item["question"] + " " + item["answer"])
                    
                    self.sections.append({
                        "doc_id": filename,
                        "section_id": f"{filename}_{i}",
                        "content": content,
                        "metadata": metadata,
                        "keywords": keywords
                    })
        elif isinstance(data, dict):
            # Handle structured document with sections
            for section_key, section_content in data.items():
                if isinstance(section_content, str):
                    self.sections.append({
                        "doc_id": filename,
                        "section_id": f"{filename}_{section_key}",
                        "content": section_content,
                        "keywords": self._extract_keywords(section_content)
                    })
                elif isinstance(section_content, dict) and "content" in section_content:
                    self.sections.append({
                        "doc_id": filename,
                        "section_id": f"{filename}_{section_key}",
                        "content": section_content["content"],
                        "metadata": {k: v for k, v in section_content.items() if k != "content"},
                        "keywords": self._extract_keywords(section_content["content"])
                    })
    
    def _extract_sections(self, text, filename, chunk_size=500, overlap=100):
        """
        Extract meaningful sections from text with overlap for context preservation
        
        Args:
            text (str): Full text to process
            filename (str): Source filename
            chunk_size (int): Target size for each chunk
            overlap (int): Overlap between consecutive chunks
        """
        # Clean text: normalize whitespace, remove excessive newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # First try to split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Process paragraphs into chunks
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size and we already have content,
            # store the current chunk and start a new one with overlap
            if current_chunk and len(current_chunk) + len(para) > chunk_size:
                chunks.append(current_chunk)
                
                # Get the last part of the current chunk for overlap
                words = current_chunk.split()
                if len(words) > overlap // 10:  # Approximate words to overlap
                    current_chunk = ' '.join(words[-(overlap // 10):])
                else:
                    current_chunk = ""
            
            # Add the paragraph to the current chunk
            if current_chunk:
                current_chunk += " " + para
            else:
                current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # If we have very few chunks, split further by sentences
        if len(chunks) <= 1 and len(text) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if current_chunk and len(current_chunk) + len(sentence) > chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            
            if current_chunk:
                chunks.append(current_chunk)
        
        # Add each chunk as a section
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 20:  # Only consider substantial chunks
                self.sections.append({
                    "doc_id": filename,
                    "section_id": f"{filename}_chunk_{i}",
                    "content": chunk.strip(),
                    "keywords": self._extract_keywords(chunk)
                })
    
    def _extract_keywords(self, text):
        """
        Extract keywords from text (still useful for filtering)
        
        Args:
            text (str): Text to extract keywords from
            
        Returns:
            list: List of keywords
        """
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words
        words = text.split()
        
        # Remove common stopwords
        stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
                    "in", "on", "at", "to", "for", "with", "by", "about", "against",
                    "between", "into", "through", "during", "before", "after", "above",
                    "below", "from", "up", "down", "of", "and", "but", "or", "because",
                    "as", "until", "while", "if", "then", "else", "when", "where", "why",
                    "how", "all", "any", "both", "each", "few", "more", "most", "other",
                    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
                    "than", "too", "very", "s", "t", "can", "will", "just", "don",
                    "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
                    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma",
                    "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
                    "won", "wouldn"}
        
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords
    
    def _generate_embeddings(self):
        """Generate embeddings for all sections using the sentence transformer model"""
        if not self.embedding_model:
            self.logger.warning("Embedding model not available, skipping embedding generation")
            return
        
        self.logger.info(f"Generating embeddings for {len(self.sections)} sections")
        
        try:
            # Extract content from all sections
            texts = [section["content"] for section in self.sections]
            
            # Generate embeddings in batches to avoid memory issues
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(texts):
                    self.logger.info(f"Generated embeddings for {min(i+batch_size, len(texts))}/{len(texts)} sections")
            
            self.section_embeddings = all_embeddings
            self.logger.info("Finished generating all embeddings")
            
            # Update the cache
            if self.cache_results:
                cache_path = os.path.join(self.index_dir, "knowledge_index.pkl")
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump({
                            'documents': self.documents,
                            'sections': self.sections,
                            'embeddings': self.section_embeddings
                        }, f)
                    self.logger.info(f"Updated cached knowledge index with embeddings")
                except Exception as e:
                    self.logger.error(f"Error updating cached knowledge index: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            self.logger.warning("Falling back to keyword-based matching")
            self.section_embeddings = []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def query(self, query_text, intent=None, top_k=3):
        """
        Query the knowledge base for information using semantic search
        
        Args:
            query_text (str): Query text
            intent (str, optional): Identified intent for context
            top_k (int): Number of top matches to retrieve and rerank
            
        Returns:
            str: Relevant information or None if not found
        """
        self.logger.info(f"Querying knowledge base with: '{query_text}', intent: {intent}")
        
        # Prefilter by intent if provided
        if intent:
            filtered_sections = [
                (i, section) for i, section in enumerate(self.sections)
                if self._is_section_relevant_to_intent(section, intent)
            ]
            
            if filtered_sections:
                section_indices = [i for i, _ in filtered_sections]
                sections = [section for _, section in filtered_sections]
                
                if self.embedding_model and self.section_embeddings:
                    filtered_embeddings = [self.section_embeddings[i] for i in section_indices]
                else:
                    filtered_embeddings = []
            else:
                # If no sections match the intent, fall back to all sections
                sections = self.sections
                filtered_embeddings = self.section_embeddings
                section_indices = list(range(len(sections)))
        else:
            sections = self.sections
            filtered_embeddings = self.section_embeddings
            section_indices = list(range(len(sections)))
        
        # If we have embeddings, use semantic search
        if self.embedding_model and filtered_embeddings:
            try:
                # Generate query embedding
                query_embedding = self.embedding_model.encode(query_text)
                
                # Calculate cosine similarity
                similarities = cosine_similarity(
                    [query_embedding], 
                    filtered_embeddings
                )[0]
                
                # Get top_k matches
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                # Only consider results with similarity above threshold
                threshold = 0.3  # Adjust based on testing
                relevant_results = [
                    (section_indices[idx], sections[idx], similarities[idx])
                    for idx in top_indices
                    if similarities[idx] > threshold
                ]
                
                # If we have results, return the highest scoring one
                if relevant_results:
                    # Reranking: Boost scores based on keyword overlap as a secondary factor
                    reranked_results = []
                    query_keywords = self._extract_keywords(query_text)
                    
                    for orig_idx, section, similarity in relevant_results:
                        # Calculate keyword overlap score
                        keyword_score = self._calculate_keyword_relevance(query_keywords, section["keywords"])
                        
                        # Combine scores (weight towards semantic similarity)
                        combined_score = similarity * 0.8 + keyword_score * 0.2
                        
                        reranked_results.append((orig_idx, section, combined_score))
                    
                    # Sort by combined score
                    reranked_results.sort(key=lambda x: x[2], reverse=True)
                    
                    best_match = reranked_results[0][1]["content"]
                    
                    # Clean up the response
                    best_match = self._clean_response(best_match)
                    
                    return best_match
            
            except Exception as e:
                self.logger.error(f"Error in semantic search: {str(e)}")
                self.logger.warning("Falling back to keyword matching")
        
        # Fall back to keyword matching if embeddings are not available or failed
        return self._keyword_based_query(query_text, intent, sections)
    
    def _keyword_based_query(self, query_text, intent, sections):
        """
        Legacy keyword-based querying as fallback
        
        Args:
            query_text (str): Query text
            intent (str): Identified intent
            sections (list): Sections to search
            
        Returns:
            str: Relevant information or None if not found
        """
        # Extract keywords from query
        query_keywords = self._extract_keywords(query_text)
        self.logger.debug(f"Query keywords: {query_keywords}")
        
        if not query_keywords:
            self.logger.warning("No keywords extracted from query")
            return None
        
        # Score each section based on keyword matches
        results = []
        for section in sections:
            score = self._calculate_keyword_relevance(query_keywords, section["keywords"])
            
            # Boost score based on intent if provided
            if intent and self._is_section_relevant_to_intent(section, intent):
                score *= 1.5
            
            if score > 0:
                results.append({
                    "section": section,
                    "score": score
                })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return the most relevant section if we have results
        if results and results[0]["score"] > 0.2:  # Threshold for relevance
            best_match = results[0]["section"]["content"]
            
            # Clean up the response
            best_match = self._clean_response(best_match)
            
            return best_match
        
        return None
    
    def _calculate_keyword_relevance(self, query_keywords, section_keywords):
        """
        Calculate keyword-based relevance score between query and section
        
        Args:
            query_keywords (list): Keywords from the query
            section_keywords (list): Keywords from the section
            
        Returns:
            float: Relevance score (0-1)
        """
        if not query_keywords or not section_keywords:
            return 0
        
        # Count matching keywords
        matches = sum(1 for keyword in query_keywords if keyword in section_keywords)
        
        # Calculate score based on proportion of query keywords matched
        score = matches / len(query_keywords)
        
        return score
    
    def _is_section_relevant_to_intent(self, section, intent):
        """
        Check if a section is relevant to the given intent
        
        Args:
            section (dict): Section data
            intent (str): Intent to check against
            
        Returns:
            bool: True if relevant, False otherwise
        """
        # Intent-specific keywords
        intent_keywords = {
            "product_info": ["product", "offer", "feature", "service", "advantage", "benefit"],
            "pricing": ["price", "cost", "fee", "charge", "rate", "interest", "emi", "money"],
            "eligibility": ["eligible", "qualify", "requirement", "criteria", "condition"],
            "process": ["process", "step", "procedure", "how", "method", "approach"],
            "fund_performance": ["performance", "return", "yield", "profit", "gain"],
            "risk": ["risk", "safe", "secure", "guarantee", "assured", "protection"],
            "investment_duration": ["duration", "period", "term", "time", "maturity"]
        }
        
        # Check for intent keywords in section
        if intent in intent_keywords:
            for keyword in intent_keywords[intent]:
                content = section["content"].lower()
                if keyword in content:
                    return True
                
                # Also check in metadata if available
                if "metadata" in section and isinstance(section["metadata"], dict):
                    for value in section["metadata"].values():
                        if isinstance(value, str) and keyword in value.lower():
                            return True
        
        return False
    
    def _clean_response(self, text):
        """
        Clean up a response for presentation
        
        Args:
            text (str): Raw response text
            
        Returns:
            str: Cleaned response text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle Q&A format
        if text.startswith("Q:"):
            parts = text.split("A:", 1)
            if len(parts) > 1:
                return parts[1].strip()
        
        return text
    
    def get_relevant_context(self, query, intent=None, max_tokens=2000):
        """
        Get relevant context for the LLM based on query
        
        Args:
            query (str): User query
            intent (str, optional): User intent
            max_tokens (int): Maximum context tokens to return
            
        Returns:
            str: Combined context from relevant sections
        """
        self.logger.info(f"Getting relevant context for: '{query}', intent: {intent}")
        
        # Use semantic search if available
        if self.embedding_model and self.section_embeddings:
            try:
                # Generate query embedding
                query_embedding = self.embedding_model.encode(query)
                
                # Calculate similarities with all sections
                similarities = cosine_similarity(
                    [query_embedding], 
                    self.section_embeddings
                )[0]
                
                # Get top matches
                top_k = 5  # Start with top 5
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                # Filter by minimum similarity
                threshold = 0.3
                relevant_sections = [
                    (self.sections[idx]["content"], similarities[idx])
                    for idx in top_indices
                    if similarities[idx] > threshold
                ]
                
                # Sort by similarity
                relevant_sections.sort(key=lambda x: x[1], reverse=True)
                
                # Combine sections into context
                context = ""
                total_length = 0
                estimated_token_ratio = 1.5  # Approximate tokens per character
                
                for section_text, _ in relevant_sections:
                    # Estimate token count and check if adding this would exceed limit
                    estimated_tokens = len(section_text) / estimated_token_ratio
                    if total_length + estimated_tokens > max_tokens:
                        break
                    
                    # Add to context
                    context += section_text + "\n\n"
                    total_length += estimated_tokens
                
                return context.strip()
            
            except Exception as e:
                self.logger.error(f"Error getting context with semantic search: {str(e)}")
                self.logger.warning("Falling back to keyword matching for context")
        
        # Fallback to keyword matching
        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            return ""
        
        # Score sections by keyword relevance
        scored_sections = []
        for section in self.sections:
            score = self._calculate_keyword_relevance(query_keywords, section["keywords"])
            if intent and self._is_section_relevant_to_intent(section, intent):
                score *= 1.5
            if score > 0.2:  # Minimum relevance threshold
                scored_sections.append((section["content"], score))
        
        # Sort by score
        scored_sections.sort(key=lambda x: x[1], reverse=True)
        
        # Combine sections into context
        context = ""
        total_length = 0
        estimated_token_ratio = 1.5
        
        for section_text, _ in scored_sections[:5]:  # Limit to top 5
            estimated_tokens = len(section_text) / estimated_token_ratio
            if total_length + estimated_tokens > max_tokens:
                break
            
            context += section_text + "\n\n"
            total_length += estimated_tokens
        
        return context.strip()