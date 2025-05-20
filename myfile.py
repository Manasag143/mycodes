import fitz  # PyMuPDF
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class PDFQuestionAnsweringPipeline:
    def __init__(self):
        self.documents = {}
        self.page_texts = {}
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        
    def load_pdfs(self, pdf_dir):
        """Load multiple PDFs from a directory."""
        for filename in os.listdir(pdf_dir):
            if filename.endswith('.pdf'):
                file_path = os.path.join(pdf_dir, filename)
                self.load_pdf(file_path)
                
    def load_pdf(self, pdf_path):
        """Extract text from a PDF and store it."""
        doc_name = os.path.basename(pdf_path)
        doc = fitz.open(pdf_path)
        
        # Extract text from each page
        all_text = ""
        page_texts = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            page_texts.append(text)
            all_text += text + "\n\n"  # Add separation between pages
            
        # Store the document text
        self.documents[doc_name] = all_text
        self.page_texts[doc_name] = page_texts
        
        print(f"Loaded {doc_name} with {len(page_texts)} pages")
        
    def preprocess_text(self, text):
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_index(self):
        """Create TF-IDF vectors for document retrieval."""
        if not self.documents:
            print("No documents loaded. Please load PDFs first.")
            return
        
        # Prepare documents for vectorization
        docs = list(self.documents.values())
        processed_docs = [self.preprocess_text(doc) for doc in docs]
        
        # Create TF-IDF vectors
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
        
        print(f"Built index with {len(docs)} documents")
    
    def find_relevant_document(self, question):
        """Find the most relevant document for a question."""
        if self.tfidf_matrix is None:
            print("Index not built. Please run build_index() first.")
            return None
        
        # Process question
        processed_question = self.preprocess_text(question)
        question_vector = self.vectorizer.transform([processed_question])
        
        # Calculate similarity with documents
        similarities = cosine_similarity(question_vector, self.tfidf_matrix)[0]
        
        # Get the most similar document
        most_similar_idx = np.argmax(similarities)
        doc_name = list(self.documents.keys())[most_similar_idx]
        
        return doc_name, similarities[most_similar_idx]
    
    def find_relevant_sections(self, question, doc_name, top_n=3):
        """Find the most relevant page sections for a question."""
        if doc_name not in self.page_texts:
            return []
        
        # Get page texts for the document
        pages = self.page_texts[doc_name]
        
        # Process pages and question
        processed_pages = [self.preprocess_text(page) for page in pages]
        processed_question = self.preprocess_text(question)
        
        # Create TF-IDF vectors for pages
        page_vectorizer = TfidfVectorizer()
        page_vectors = page_vectorizer.fit_transform(processed_pages)
        question_vector = page_vectorizer.transform([processed_question])
        
        # Calculate similarity with pages
        similarities = cosine_similarity(question_vector, page_vectors)[0]
        
        # Get the top N most similar pages
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        return [(pages[i], similarities[i]) for i in top_indices]
    
    def extract_tables(self, pdf_path):
        """Extract tables from PDF using PyMuPDF."""
        tables = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Find table-like structures (this is a simplified approach)
            # For better table extraction, consider using dedicated libraries like tabula-py
            blocks = page.get_text("blocks")
            
            # Simple heuristic: look for blocks with multiple lines and columns
            for block in blocks:
                text = block[4]
                lines = text.split('\n')
                
                # Simple check for table-like structure
                if len(lines) > 3:
                    # Check if lines have similar structure (e.g., same number of spaces or tabs)
                    is_table = True
                    for i in range(1, len(lines)):
                        if lines[i].count('\t') != lines[0].count('\t'):
                            is_table = False
                            break
                    
                    if is_table:
                        tables.append(text)
        
        return tables
    
    def answer_question(self, question):
        """Answer a question based on loaded PDFs."""
        if not self.documents:
            return "No documents loaded. Please load PDFs first."
        
        # Find relevant document
        doc_name, doc_similarity = self.find_relevant_document(question)
        
        if doc_similarity < 0.1:
            return "I couldn't find relevant information to answer this question."
        
        # Find relevant sections
        relevant_sections = self.find_relevant_sections(question, doc_name)
        
        if not relevant_sections:
            return "I found a potentially relevant document but couldn't locate specific information for your question."
        
        # Compile context
        context = "\n\n".join([section[0] for section in relevant_sections])
        
        # Simple answer generation (in production, you might use an LLM API here)
        answer = self._generate_answer(question, context)
        
        return {
            "answer": answer,
            "source_document": doc_name,
            "confidence": float(doc_similarity),
            "relevant_context": context[:500] + "..." if len(context) > 500 else context
        }
    
    def _generate_answer(self, question, context):
        """
        Generate an answer based on context.
        
        For a production system, replace this with a call to an LLM API
        like OpenAI, Claude, or a local model like LlamaCpp.
        """
        # This is a placeholder method
        # In a real implementation, you would send the question and context to an LLM
        
        words = question.lower().split()
        
        # Very simple keyword matching (just for demonstration)
        for word in words:
            if word in context.lower():
                # Find the sentence containing the keyword
                sentences = re.split(r'(?<=[.!?])\s+', context)
                for sentence in sentences:
                    if word in sentence.lower():
                        return sentence
        
        return "Based on the context, I can't generate a specific answer. In a production system, this would use a language model to generate an accurate response."
    
    def answer_multiple_questions(self, questions):
        """Answer a list of questions."""
        answers = {}
        for question in questions:
            answers[question] = self.answer_question(question)
        return answers

# Example usage
if __name__ == "__main__":
    pipeline = PDFQuestionAnsweringPipeline()
    
    # Load PDFs
    pipeline.load_pdf("example.pdf")
    
    # Build search index
    pipeline.build_index()
    
    # Define questions
    questions = [
        "What is the main topic of the document?",
        "What are the key findings?",
        "When was the research conducted?",
        "Who are the authors of the paper?",
        "What methodology was used?",
        "What are the limitations mentioned?",
        "What recommendations are made?",
        "How does this compare to previous research?"
    ]
    
    # Answer questions
    results = pipeline.answer_multiple_questions(questions)
    
    # Print results
    for question, result in results.items():
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Source: {result['source_document']} (Confidence: {result['confidence']:.2f})")
