import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

class KnowledgeBase:
    """Knowledge base for acne treatment information with RAG capabilities"""
    
    def __init__(self, knowledge_file: str = "acne_knowledge_base.json"):
        """
        Initialize the knowledge base
        
        Args:
            knowledge_file: Path to the JSON knowledge base file
        """
        self.knowledge_file = knowledge_file
        self.knowledge_data = {}
        self.treatments_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.sentence_model = None
        self.treatment_embeddings = None
        
        # Load knowledge base
        self.load_knowledge_base()
        
        # Initialize search components
        self.initialize_search_components()
    
    def load_knowledge_base(self):
        """
        Load knowledge base from JSON file
        """
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    self.knowledge_data = json.load(f)
                
                # Convert treatments to DataFrame for easier manipulation
                if 'treatments' in self.knowledge_data:
                    self.treatments_df = pd.DataFrame(self.knowledge_data['treatments'])
                
                print(f"Knowledge base loaded successfully with {len(self.knowledge_data.get('treatments', []))} treatments")
            else:
                print(f"Knowledge base file {self.knowledge_file} not found. Creating empty knowledge base.")
                self.knowledge_data = {'treatments': [], 'skincare_routine': {}, 'lifestyle_tips': []}
                
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            self.knowledge_data = {'treatments': [], 'skincare_routine': {}, 'lifestyle_tips': []}
    
    def initialize_search_components(self):
        """
        Initialize TF-IDF vectorizer and sentence transformer for semantic search
        """
        try:
            if self.treatments_df is not None and not self.treatments_df.empty:
                # Prepare text corpus for TF-IDF
                treatment_texts = []
                for _, treatment in self.treatments_df.iterrows():
                    text = f"{treatment['name']} {treatment['description']} {' '.join(treatment.get('keywords', []))}"
                    treatment_texts.append(text)
                
                # Initialize TF-IDF vectorizer
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(treatment_texts)
                
                # Initialize sentence transformer for semantic search
                try:
                    self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.treatment_embeddings = self.sentence_model.encode(treatment_texts)
                    print("Semantic search initialized successfully")
                except Exception as e:
                    print(f"Warning: Could not initialize sentence transformer: {e}")
                    self.sentence_model = None
                
                print("Search components initialized successfully")
            
        except Exception as e:
            print(f"Error initializing search components: {e}")
    
    def search_treatments(self, query: str, method: str = "hybrid", top_k: int = 3) -> List[Dict]:
        """
        Search for relevant treatments based on query
        
        Args:
            query: Search query
            method: Search method ('tfidf', 'semantic', 'hybrid')
            top_k: Number of top results to return
            
        Returns:
            List of relevant treatment dictionaries
        """
        if self.treatments_df is None or self.treatments_df.empty:
            return []
        
        try:
            if method == "tfidf":
                return self._tfidf_search(query, top_k)
            elif method == "semantic" and self.sentence_model is not None:
                return self._semantic_search(query, top_k)
            elif method == "hybrid":
                return self._hybrid_search(query, top_k)
            else:
                # Fallback to keyword search
                return self._keyword_search(query, top_k)
                
        except Exception as e:
            print(f"Error in search: {e}")
            return self._keyword_search(query, top_k)
    
    def _tfidf_search(self, query: str, top_k: int) -> List[Dict]:
        """
        TF-IDF based search
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant treatments
        """
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return self._keyword_search(query, top_k)
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include relevant results
                treatment = self.treatments_df.iloc[idx].to_dict()
                treatment['relevance_score'] = float(similarities[idx])
                results.append(treatment)
        
        return results
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Semantic search using sentence transformers
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant treatments
        """
        if self.sentence_model is None or self.treatment_embeddings is None:
            return self._tfidf_search(query, top_k)
        
        # Encode query
        query_embedding = self.sentence_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.treatment_embeddings).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for semantic similarity
                treatment = self.treatments_df.iloc[idx].to_dict()
                treatment['relevance_score'] = float(similarities[idx])
                results.append(treatment)
        
        return results
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Hybrid search combining TF-IDF and semantic search
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant treatments
        """
        # Get results from both methods
        tfidf_results = self._tfidf_search(query, top_k * 2)
        semantic_results = self._semantic_search(query, top_k * 2) if self.sentence_model else []
        
        # Combine and deduplicate results
        combined_results = {}
        
        # Add TF-IDF results with weight
        for result in tfidf_results:
            treatment_id = result['id']
            combined_results[treatment_id] = result.copy()
            combined_results[treatment_id]['tfidf_score'] = result['relevance_score']
            combined_results[treatment_id]['semantic_score'] = 0.0
        
        # Add semantic results with weight
        for result in semantic_results:
            treatment_id = result['id']
            if treatment_id in combined_results:
                combined_results[treatment_id]['semantic_score'] = result['relevance_score']
            else:
                combined_results[treatment_id] = result.copy()
                combined_results[treatment_id]['tfidf_score'] = 0.0
                combined_results[treatment_id]['semantic_score'] = result['relevance_score']
        
        # Calculate combined score
        for treatment_id in combined_results:
            tfidf_score = combined_results[treatment_id].get('tfidf_score', 0)
            semantic_score = combined_results[treatment_id].get('semantic_score', 0)
            # Weighted combination (60% semantic, 40% TF-IDF)
            combined_score = 0.6 * semantic_score + 0.4 * tfidf_score
            combined_results[treatment_id]['relevance_score'] = combined_score
        
        # Sort by combined score and return top results
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['relevance_score'],
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Simple keyword-based search as fallback
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant treatments
        """
        query_lower = query.lower()
        results = []
        
        for _, treatment in self.treatments_df.iterrows():
            score = 0
            treatment_dict = treatment.to_dict()
            
            # Check name
            if query_lower in treatment['name'].lower():
                score += 3
            
            # Check description
            if query_lower in treatment['description'].lower():
                score += 2
            
            # Check keywords
            keywords = treatment.get('keywords', [])
            for keyword in keywords:
                if query_lower in keyword.lower():
                    score += 1
            
            if score > 0:
                treatment_dict['relevance_score'] = score
                results.append(treatment_dict)
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]
    
    def get_treatment_by_severity(self, severity: str) -> List[Dict]:
        """
        Get treatments suitable for specific acne severity
        
        Args:
            severity: Acne severity level ('mild', 'moderate', 'severe')
            
        Returns:
            List of suitable treatments
        """
        if self.treatments_df is None:
            return []
        
        suitable_treatments = []
        for _, treatment in self.treatments_df.iterrows():
            if severity in treatment.get('severity', []):
                suitable_treatments.append(treatment.to_dict())
        
        return suitable_treatments
    
    def get_skincare_routine(self, time_of_day: str = "both") -> Dict:
        """
        Get skincare routine recommendations
        
        Args:
            time_of_day: 'morning', 'evening', or 'both'
            
        Returns:
            Skincare routine dictionary
        """
        routine = self.knowledge_data.get('skincare_routine', {})
        
        if time_of_day == "both":
            return routine
        elif time_of_day in routine:
            return {time_of_day: routine[time_of_day]}
        else:
            return {}
    
    def get_lifestyle_tips(self, category: Optional[str] = None) -> List[Dict]:
        """
        Get lifestyle tips for acne management
        
        Args:
            category: Specific category to filter by (optional)
            
        Returns:
            List of lifestyle tips
        """
        tips = self.knowledge_data.get('lifestyle_tips', [])
        
        if category:
            return [tip for tip in tips if tip.get('category', '').lower() == category.lower()]
        
        return tips
    
    def get_dermatologist_referral_criteria(self) -> List[str]:
        """
        Get criteria for when to see a dermatologist
        
        Returns:
            List of referral criteria
        """
        return self.knowledge_data.get('when_to_see_dermatologist', [])
    
    def get_myths_and_facts(self) -> List[Dict]:
        """
        Get acne myths and facts
        
        Returns:
            List of myth/fact pairs
        """
        return self.knowledge_data.get('myths_and_facts', [])
    
    def add_treatment(self, treatment: Dict) -> bool:
        """
        Add a new treatment to the knowledge base
        
        Args:
            treatment: Treatment dictionary
            
        Returns:
            Success status
        """
        try:
            self.knowledge_data['treatments'].append(treatment)
            self.treatments_df = pd.DataFrame(self.knowledge_data['treatments'])
            self.initialize_search_components()  # Reinitialize search components
            return True
        except Exception as e:
            print(f"Error adding treatment: {e}")
            return False
    
    def save_knowledge_base(self) -> bool:
        """
        Save knowledge base to file
        
        Returns:
            Success status
        """
        try:
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
            return False