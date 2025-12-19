import json
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from knowledge_agent_rag import KnowledgeAgentRAG   # <-- your RAG agent


class ExplanationAgent:
    """
    Generates a human‑readable, auditor‑friendly explanation of:
    - compliance violations
    - applied swaps
    - direct fixes
    - why rules exist (via RAG knowledge)
    - any skipped or unconfirmed items
    """

    def __init__(self, compliance_path: str, manifest_path: str, rules_path: str = None):
        # Load compliance report
        with open(compliance_path, "r") as f:
            self.compliance_report = json.load(f)

        # Load final roster manifest
        with open(manifest_path, "r") as f:
            self.final_manifest = json.load(f)

        # Load optional rules JSON
        self.rules = {}
        if rules_path:
            with open(rules_path, "r") as f:
                self.rules = json.load(f)

        
        # Initialize RAG knowledge agent
        # -----------------------------
        self.rag = KnowledgeAgentRAG(
            persistence_dir="./chroma_knowledge",
            collection_name="compliance_knowledge"
        )

        
        # Initialize Flan‑T5 LLM
        # ----------------------
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512
        )

        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

        
        # Prompt template
        # ---------------
        self.prompt = PromptTemplate(
            input_variables=["compliance", "manifest", "rules", "rag_context"],
            template=(
                "You are an ExplanationAgent. Your job is to explain compliance outcomes.\n\n"
                "Compliance Report:\n{compliance}\n\n"
                "Final Roster Manifest:\n{manifest}\n\n"
                "Compliance Rules:\n{rules}\n\n"
                "Relevant Knowledge (RAG):\n{rag_context}\n\n"
                "Generate a clear, human-readable explanation of:\n"
                "- Which swaps were suggested and which were applied\n"
                "- What direct fixes were applied (meal breaks, penalties)\n"
                "- Why these rules exist (use RAG context when relevant)\n"
                "- Any skipped or unconfirmed items\n"
                "Write in a structured, auditor-friendly style."
            )
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    
    # Generate explanation
    # --------------------
    def generate_explanation(self, out_json="explanation_report.json"):
        # Query RAG for relevant knowledge
        rag_query = "Explain the Fair Work rules behind these compliance violations."
        rag_result = self.rag.answer(rag_query, k=6)
        rag_context = "\n".join(rag_result.get("contexts", []))

        # Run LLM chain
        explanation_text = self.chain.run(
            compliance=json.dumps(self.compliance_report, indent=2),
            manifest=json.dumps(self.final_manifest, indent=2),
            rules=json.dumps(self.rules, indent=2),
            rag_context=rag_context
        )

        explanation = {
            "compliance_file_used": self.final_manifest.get("compliance_file_used"),
            "rag_sources": rag_result.get("sources", []),
            "explanation_text": explanation_text
        }

        with open(out_json, "w") as f:
            json.dump(explanation, f, indent=2)

        return explanation

    
    # Human-readable output
    # ---------------------
    def human_report(self) -> str:
        return self.generate_explanation()["explanation_text"]

