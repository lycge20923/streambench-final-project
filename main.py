from base import Agent
from execution_pipeline import main
from argparse import ArgumentParser
from execution_pipeline import main
from datetime import datetime
from types import SimpleNamespace

from code.arguments import LLMArguments, RAGArguments, ClassificationArguments, SQLGenerationArguments
class ClassificationAgent(Agent):
    """
    An agent that classifies text into one of the labels in the given label set.
    """
    def __init__(self, config: dict) -> None:
        """
        Initialize your LLM here
        """
        # TODO
        raise NotImplementedError

    def __call__(
        self,
        label2desc: dict[str, str],
        text: str
    ) -> str:
        """
        Classify the text into one of the labels.

        Args:
            label2desc (dict[str, str]): A dictionary mapping each label to its description.
            text (str): The text to classify.

        Returns:
            str: The label (should be a key in label2desc) that the text is classified into.

        For example:
        label2desc = {
            "apple": "A fruit that is typically red, green, or yellow.",
            "banana": "A long curved fruit that grows in clusters and has soft pulpy flesh and yellow skin when ripe.",
            "cherry": "A small, round stone fruit that is typically bright or dark red.",
        }
        text = "The fruit is red and about the size of a tennis ball."
        label = "apple" (should be a key in label2desc, i.e., ["apple", "banana", "cherry"])
        """
        # TODO
        raise NotImplementedError

    def update(self, correctness: bool) -> bool:
        """
        Update your LLM agent based on the correctness of its own prediction at the current time step.

        Args:
            correctness (bool): Whether the prediction is correct.

        Returns:
            bool: Whether the prediction is correct.
        """
        # TODO
        raise NotImplementedError

class SQLGenerationAgent(Agent):
    """
    An agent that generates SQL code based on the given table schema and the user query.
    """
    def __init__(self, config: dict) -> None:
        """
        Initialize your LLM here
        """
        # TODO
        raise NotImplementedError

    def __call__(
        self,
        table_schema: str,
        user_query: str
    ) -> str:
        """
        Generate SQL code based on the given table schema and the user query.

        Args:
            table_schema (str): The table schema.
            user_query (str): The user query.

        Returns:
            str: The SQL code that the LLM generates.
        """
        # TODO: Note that your output should be a valid SQL code only.
        raise NotImplementedError

    def update(self, correctness: bool) -> bool:
        """
        Update your LLM agent based on the correctness of its own SQL    code at the current time step.
        """
        # TODO
        raise NotImplementedError
        
if __name__ == "__main__":
    # from argparse import ArgumentParser
    # from execution_pipeline import main
    
    # declare required arguments for the main.py
    parser = ArgumentParser()
    parser.add_argument('--bench_name', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    
    # pass for function main
    bench_cfg = {
        'bench_name': args.bench_name, 
        'output_path': args.output_path
    }
    
    # pass for the model
    if args.bench_name.startswith("classification"):
        agent_name = ClassificationAgent
        models_args = SimpleNamespace(**LLMArguments().__dict__, **ClassificationArguments().__dict__)
    elif args.bench_name.startswith("sql_generation"):
        agent_name = SQLGenerationAgent
        models_args = SimpleNamespace(**LLMArguments().__dict__, **SQLGenerationArguments().__dict__)
    else:
        raise ValueError(f"Invalid benchmark name: {args.bench_name}")
    
    RAG_args = RAGArguments()
    
    implement_time = datetime.now()
    implement_time = implement_time.strftime("%Y%m%d-%H%M%S")
    
    config = {
        'model_name': models_args.model_name,
        'exp_name': f"{implement_time}_{bench_cfg['bench_name']}_{bench_cfg['output_path']}",
        'bench_name': bench_cfg['bench_name'],
        'max_tokens': models_args.max_tokens,
        'do_sample': False,
        'device': models_args.device,
        'use_8bit': models_args.use_8bit,
        'rag': {
            'embedding_model': RAG_args.embedding_model,
            'seed': RAG_args.seed,
            "top_k": RAG_args.top_k,
            "order": RAG_args.order
        }
    }
    agent = agent_name(config)
    main(agent, bench_cfg)
