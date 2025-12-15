# agents/Knowledge_agent.py
import pandas as pd
import chromadb
from chromadb.config import Settings

class KnowledgeAgent:
    def __init__(self):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name="compliance_store_knowledge")

    def df_to_text(self, df, title: str) -> str:
        """Convert a dataframe into a text block for RAG storage."""
        return f"{title}\n" + "\n".join([
            " | ".join(map(str, row)) for _, row in df.fillna("").iterrows()
        ])

    def run(self, context: dict) -> dict:
        docs, ids, metas = [], [], []

        # Collect all structured sheets from context
        sources = {
            "compliance_notes": context.get("compliance_notes"),
            "service_periods": context.get("service_periods"),
            "store_configs": context.get("store_configs"),
            "basic_params": context.get("basic_params"),
            "fixed_hours": context.get("fixed_hours"),
            "shift_codes": context.get("shift_codes"),
            "mgmt_roster": context.get("mgmt_roster"),
        }

        # Add each sheet if present
        for i, (name, df) in enumerate(sources.items(), start=1):
            if df is not None and isinstance(df, pd.DataFrame):
                docs.append(self.df_to_text(df, name.replace("_", " ").title()))
                ids.append(f"doc{i}")
                metas.append({"source": name})

        # Add to Chroma collection
        if docs:
            self.collection.add(documents=docs, metadatas=metas, ids=ids)

        # Expose query function
        def rag_query(q: str, n_results: int = 3):
            res = self.collection.query(query_texts=[q], n_results=n_results)
            return list(zip(res["documents"][0], res["metadatas"][0]))

        context["rag_query_fn"] = rag_query
        return context
