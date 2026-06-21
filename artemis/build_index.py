"""Build the Chroma vector index from player_stats.csv. Run once after data changes."""

import csv

import chromadb
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from artemis import config


def _team(row: dict) -> str:
    return row.get("Team") or row.get("Country") or ""


def player_to_text(row: dict) -> str:
    circuit_line = f"Circuits: {row['Circuit']}\n" if row.get("Circuit") else ""
    region_line = f"Region: {row['Region']}\n" if row.get("Region") else ""
    return (
        f"Player: {row['Player Name']}\n"
        f"Team: {_team(row)}\n"
        f"{circuit_line}"
        f"{region_line}"
        f"Agents: {row['Agents']}\n"
        f"Rounds played: {row['Rounds']}\n"
        f"Rating: {row['Rating']}\n"
        f"ACS: {row['ACS']}\n"
        f"K:D: {row['K:D']}\n"
        f"ADR: {row['ADR']}\n"
        f"KAST: {row['KAST']}\n"
        f"KPR: {row['KPR']}\n"
        f"APR: {row['APR']}\n"
        f"FKPR (first kills per round): {row['FKPR']}\n"
        f"FDPR (first deaths per round): {row['FDPR']}"
    )


def load_documents() -> list[Document]:
    documents = []
    with open(config.PLAYER_STATS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            documents.append(
                Document(
                    text=player_to_text(row),
                    metadata={
                        "player": row["Player Name"],
                        "team": _team(row),
                        "agents": row["Agents"],
                        "rating": str(row["Rating"]),
                        "region": row.get("Region", ""),
                        "circuit": row.get("Circuit", ""),
                    },
                )
            )
    return documents


def build_index() -> None:
    Settings.llm = OpenAI(model=config.LLM_MODEL, api_key=config.OPENAI_API_KEY)
    Settings.embed_model = OpenAIEmbedding(
        model=config.EMBED_MODEL, api_key=config.OPENAI_API_KEY
    )

    documents = load_documents()
    print(f"Loaded {len(documents)} players from {config.PLAYER_STATS_CSV}")

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    db = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        db.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = db.create_collection(config.COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    count = collection.count()
    if count == 0:
        raise RuntimeError("Index build finished but collection is empty.")
    print(f"Index built at {config.CHROMA_DIR} ({config.COLLECTION_NAME}, {count} docs)")


if __name__ == "__main__":
    build_index()
