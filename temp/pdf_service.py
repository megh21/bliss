import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from langchain_google_vertexai import VertexAIEmbeddings  # Correct import path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging  # Optional: for better debugging
import pickle  # Add this import at the top of file
from typing import List, Tuple


class PDFVectorStore:
    def __init__(self, pdf_dir="files", gcp_project_id=None, gcp_region=None):
        """Initialize PDF vectorstore with config and embedding model."""
        self.pdf_dir = pdf_dir
        self.gcp_project_id = gcp_project_id or "bliss-hack25fra-9578"
        self.gcp_region = gcp_region or "europe-west1"

        # Initialize embedding model
        self.embedding_model = VertexAIEmbeddings(
            project=self.gcp_project_id,
            location=self.gcp_region,
            model_name="multimodalembedding",
        )

        # Initialize storage variables
        self.documents = []
        self.pdf_mapping = []
        self.page_numbers = []
        self.index = None
        self.dimension = None

    def process_pdfs(self):
        """Process PDFs and create embeddings + index."""
        # Load and process PDFs
        pdf_texts = {}
        if not os.path.isdir(self.pdf_dir):
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_dir}")

        # Extract text from PDFs
        for pdf_file in os.listdir(self.pdf_dir):
            if pdf_file.lower().endswith(".pdf"):
                file_path = os.path.join(self.pdf_dir, pdf_file)
                pdf_texts[pdf_file] = self._extract_text_from_pdf(file_path)

        if not pdf_texts:
            raise ValueError(f"No PDF files found in {self.pdf_dir}")

        # Process documents and create chunks
        self._process_documents(pdf_texts)

        # Generate embeddings and build index
        self._build_index()

        return True

    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file."""
        try:
            doc = fitz.open(pdf_path)
            pages = []
            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                if text.strip():
                    pages.append((page_num + 1, text))
            doc.close()
            return pages
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {e}")
            return []

    def _process_documents(self, pdf_texts):
        """Process PDF texts into chunks."""
        # Initialize text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100, length_function=len
        )

        # Process documents
        documents = []
        pdf_mapping = []
        page_numbers = []

        for pdf_name, pages in pdf_texts.items():
            for page_num, text in pages:
                if text.strip():
                    chunks = splitter.split_text(text)
                    documents.extend(chunks)
                    pdf_mapping.extend([pdf_name] * len(chunks))
                    page_numbers.extend([page_num] * len(chunks))

        self.documents = documents
        self.pdf_mapping = pdf_mapping
        self.page_numbers = page_numbers

    def _build_index(self):
        """Generate embeddings and build FAISS index."""
        # Generate embeddings
        embeddings_list = self.embedding_model.embed_documents(self.documents)
        embeddings = np.array(embeddings_list).astype("float32")

        # Create and populate index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

    def save(self, save_dir="vectorstore"):
        """Save vectorstore to disk."""
        try:
            os.makedirs(save_dir, exist_ok=True)

            # Save FAISS index
            index_path = os.path.join(save_dir, "index.faiss")
            faiss.write_index(self.index, index_path)

            # Save metadata using pickle
            metadata = {
                "documents": self.documents,
                "pdf_mapping": self.pdf_mapping,
                "page_numbers": self.page_numbers,
                "dimension": self.dimension,
            }

            metadata_path = os.path.join(save_dir, "index.pkl")
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)

            logging.info(f"Vectorstore saved successfully to {save_dir}")
            return True
        except Exception as e:
            logging.error(f"Error saving vectorstore: {e}")
            return False

    def load(self, save_dir="vectorstore") -> faiss.Index:
        """Load FAISS index from disk."""
        try:
            # Load FAISS index
            index_path = os.path.join(save_dir, "index.faiss")
            faiss_index = faiss.read_index(index_path)

            # Load metadata
            metadata_path = os.path.join(save_dir, "index.pkl")
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            # Restore metadata
            self.documents = metadata["documents"]
            self.pdf_mapping = metadata["pdf_mapping"]
            self.page_numbers = metadata["page_numbers"]
            self.dimension = metadata["dimension"]
            self.index = faiss_index

            logging.info(f"FAISS index loaded successfully from {save_dir}")
            return faiss_index
        except Exception as e:
            logging.error(f"Error loading FAISS index: {e}")
            return None

    def query(
        self, text_query: str = "", image_path: str = None, top_k: int = 1
    ) -> List[Tuple[str, int]]:
        """Search through vectors with text/image query."""
        if not self.index:
            raise ValueError("No index loaded. Call process_pdfs() or load() first.")

        try:
            # Get query embedding
            if image_path:
                image_data = self._process_image(image_path)
                if not image_data:
                    return []
                query_data = {"text": text_query, "image": image_data}
                query_embedding = self.embedding_model.embed_query(query_data)
            else:
                query_embedding = self.embedding_model.embed_query(text_query)

            # Convert to numpy array
            query_vector = np.array([query_embedding]).astype("float32")

            # Perform similarity search
            distances, indices = self.index.search(query_vector, top_k)

            # Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.pdf_mapping):
                    results.append(
                        (
                            self.pdf_mapping[idx],
                            self.page_numbers[idx],
                            float(distances[0][i]),  # Convert distance to float
                        )
                    )

            return results

        except Exception as e:
            logging.error(f"Error during query: {e}")
            return []

    def _process_image(self, image_path):
        """Process image for querying."""
        try:
            if not any(
                image_path.lower().endswith(fmt) for fmt in [".jpg", ".jpeg", ".png"]
            ):
                raise ValueError("Unsupported image format. Use: jpg, jpeg, or png")

            with open(image_path, "rb") as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return None

    def append_pdfs(self, additional_pdf_dir="new_files"):
        """Append new PDFs to existing vectorstore."""
        if not self.index:
            raise ValueError(
                "No existing index found. Call process_pdfs() or load() first."
            )

        try:
            # Process new PDFs
            pdf_texts = {}
            if not os.path.isdir(additional_pdf_dir):
                raise FileNotFoundError(
                    f"PDF directory not found: {additional_pdf_dir}"
                )

            # Extract text from new PDFs
            for pdf_file in os.listdir(additional_pdf_dir):
                if pdf_file.lower().endswith(".pdf"):
                    file_path = os.path.join(additional_pdf_dir, pdf_file)
                    pdf_texts[pdf_file] = self._extract_text_from_pdf(file_path)

            if not pdf_texts:
                raise ValueError(f"No PDF files found in {additional_pdf_dir}")

            # Store original documents count
            original_doc_count = len(self.documents)

            # Process new documents
            for pdf_name, pages in pdf_texts.items():
                for page_num, text in pages:
                    if text.strip():
                        chunks = self._process_text_chunks(text)
                        self.documents.extend(chunks)
                        self.pdf_mapping.extend([pdf_name] * len(chunks))
                        self.page_numbers.extend([page_num] * len(chunks))

            # Generate embeddings for new documents only
            new_documents = self.documents[original_doc_count:]
            new_embeddings_list = self.embedding_model.embed_documents(new_documents)
            new_embeddings = np.array(new_embeddings_list).astype("float32")

            # Add new embeddings to existing index
            self.index.add(new_embeddings)

            logging.info(
                f"Successfully appended {len(new_documents)} new document chunks"
            )
            return True

        except Exception as e:
            logging.error(f"Error appending PDFs: {e}")
            return False

    def _process_text_chunks(self, text):
        """Helper method to process text into chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100, length_function=len
        )
        return splitter.split_text(text)


if __name__ == "__main__":
    # Example usage
    vectorstore = PDFVectorStore()

    # # Process PDFs and save index
    vectorstore.process_pdfs()
    vectorstore.save()

    # Or load existing index and query
    store = vectorstore.load()

    # Append new documents
    # vectorstore.append_pdfs(additional_pdf_dir='new_files')

    # Save updated vectorstore
    # vectorstore.save()
    querry = "my mp4 player is not working whats wrong with it? I dont know how to reduce the volume"
    results = vectorstore.query("your search query", top_k=3)

    for pdf_name, page_num in results:
        print(f"PDF: {pdf_name}, Page: {page_num}")
