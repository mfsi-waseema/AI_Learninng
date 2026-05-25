"""Load a PDF, chunk text, embed with Gemini, upsert into Qdrant."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
collection_name = os.getenv("QDRANT_COLLECTION", "docs")
embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
google_api_key = os.getenv("GOOGLE_API_KEY", "")

default_pdf = Path(__file__).resolve().parent / "sample_resume.pdf"
pdf_path = Path(os.getenv("INGEST_PDF_PATH", str(default_pdf))).expanduser().resolve()

CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("INGEST_CHUNK_OVERLAP", "150"))


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t.strip())
    return "\n\n".join(parts)


def main() -> None:
    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    text = extract_pdf_text(pdf_path)
    if not text.strip():
        print(f"No extractable text from {pdf_path}", file=sys.stderr)
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    source_name = pdf_path.name

    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=google_api_key,
    )
    doc_vectors = embeddings.embed_documents(chunks)
    vector_size = len(doc_vectors[0])

    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, doc_vectors)):
        points.append(
            {
                "id": i,
                "vector": vector,
                "payload": {
                    "text": chunk,
                    "source": source_name,
                    "chunk_index": i,
                },
            }
        )

    client.upsert(collection_name=collection_name, points=points)
    print(
        f"Ingested {len(points)} chunks from '{source_name}' into "
        f"'{collection_name}' at {qdrant_host}:{qdrant_port}"
    )


if __name__ == "__main__":
    main()
