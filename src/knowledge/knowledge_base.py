"""
Knowledge base for the BFSI Sales Agent
"""
import os
import logging
import json
import re
import PyPDF2
from pathlib import Path

class KnowledgeBase:
    """
    Knowledge base for retrieving information from document sources.
    Uses simple keyword-based retrieval for the baseline version.
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
        
        # Ensure directories exist
        Path(self.docs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        
        # Document data storage
        self.documents = {}
        self.sections = []
        
        # Load and process documents
        self._load_documents()
        
    def _load_documents(self):
        """Load and process all documents in the docs directory"""
        self.logger.info(f"Loading documents from {self.docs_dir}")
        
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
        
        # Split into sections (simple paragraphs for baseline)
        paragraphs = re.split(r'\n\s*\n', text)
        
        for i, para in enumerate(paragraphs):
            if len(para.strip()) > 20:  # Only consider substantial paragraphs
                self.sections.append({
                    "doc_id": filename,
                    "section_id": f"{filename}_{i}",
                    "content": para.strip(),
                    "keywords": self._extract_keywords(para)
                })
    
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
        
        # Split into sections (simple paragraphs for baseline)
        paragraphs = re.split(r'\n\s*\n', text)
        
        for i, para in enumerate(paragraphs):
            if len(para.strip()) > 20:  # Only consider substantial paragraphs
                self.sections.append({
                    "doc_id": filename,
                    "section_id": f"{filename}_{i}",
                    "content": para.strip(),
                    "keywords": self._extract_keywords(para)
                })
    
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
                    self.sections.append({
                        "doc_id": filename,
                        "section_id": f"{filename}_{i}",
                        "content": f"Q: {item['question']}\nA: {item['answer']}",
                        "question": item["question"],
                        "answer": item["answer"],
                        "keywords": self._extract_keywords(item["question"] + " " + item["answer"])
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
    
    def _extract_keywords(self, text):
        """
        Extract keywords from text
        
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
    
    def query(self, query_text, intent=None):
        """
        Query the knowledge base for information
        
        Args:
            query_text (str): Query text
            intent (str, optional): Identified intent
            
        Returns:
            str: Relevant information or None if not found
        """
        self.logger.info(f"Querying knowledge base with: '{query_text}', intent: {intent}")
        
        # Extract keywords from query
        query_keywords = self._extract_keywords(query_text)
        self.logger.debug(f"Query keywords: {query_keywords}")
        
        if not query_keywords:
            self.logger.warning("No keywords extracted from query")
            return None
        
        # Score each section based on keyword matches
        results = []
        for section in self.sections:
            score = self._calculate_relevance_score(query_keywords, section["keywords"])
            
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
    
    def _calculate_relevance_score(self, query_keywords, section_keywords):
        """
        Calculate relevance score between query and section
        
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