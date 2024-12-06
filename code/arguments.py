from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LLMArguments:
    model_name: Optional[str] = field(
        default="google/gemma-2-9b-it",
        metadata={"help":"The name of llm."})
    device: Optional[str] = field(
        default='cuda:0'
    )
    use_8bit: Optional[bool] = field(
        default=False,
        metadata={"help":"Whether use 8 bits or not."}
    )
    use_wandb: Optional[bool] = field(
        default=False,
        metadata={"help":"Whether use wandb to track or not."}
    )

@dataclass
class RAGArguments:
    embedding_model: Optional[str] = field(
        default="BAAI/bge-base-en-v1.5",
        metadata={"help":"The model name of embedding model"}
    )
    seed: Optional[int] = field(
        default=42
    )
    top_k: Optional[int] = field(
        default=8
    )
    order: Optional[str] = field(
        default="similar_at_top"
    )

@dataclass
class ClassificationArguments:
    max_tokens: Optional[int] = field(
        default=16
    )

@dataclass
class SQLGenerationArguments:
    max_tokens: Optional[int] = field(
        default=512
    )

if __name__ == "__main__":
    test_LLMArguments = LLMArguments()
    test_RAGArguments = RAGArguments()
    combined_args = {**test_LLMArguments.__dict__, **test_RAGArguments.__dict__}
    print(combined_args)