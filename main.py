import os
from argparse import ArgumentParser
from datetime import datetime
from types import SimpleNamespace
from dotenv import load_dotenv
from huggingface_hub import login

from execution_pipeline import main
from code.arguments import LLMArguments, RAGArguments, ClassificationArguments, SQLGenerationArguments
from code.ClassificationAgent import ClassificationAgent
from code.SQLGenerationAgent import SQLGenerationAgent

def _parse_args():
    parser = ArgumentParser()
    parser.add_argument('--bench_name', type=str, required=True, choices=["classification_public", "sql_generation_public"])
    parser.add_argument('--output_path', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    # access through api 
    load_dotenv()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    login(token)
    
    # record implement time
    implement_time = datetime.now()
    implement_time = implement_time.strftime("%Y%m%d-%H%M%S")
    
    # initialization
    bench_args = _parse_args()
    if not os.path.dirname(bench_args.output_path):
        raise ValueError('You have to declare the directory!')
    bench_cfg = {
        'bench_name': bench_args.bench_name, 
        'output_path': bench_args.output_path
    }
    if bench_args.bench_name.startswith("classification"):
        agent_name = ClassificationAgent
        agent_args = SimpleNamespace(**LLMArguments().__dict__, **ClassificationArguments().__dict__)
    elif bench_args.bench_name.startswith("sql_generation"):
        agent_name = SQLGenerationAgent
        agent_args = SimpleNamespace(**LLMArguments().__dict__, **SQLGenerationArguments().__dict__)
    else:
        raise ValueError(f"Invalid benchmark name: {bench_args.bench_name}")
    
    rag_args = RAGArguments()
    config = {
        'model_name': agent_args.model_name,
        'exp_name': f"{implement_time}_{bench_cfg['bench_name']}_{bench_cfg['output_path']}",
        'bench_name': bench_cfg['bench_name'],
        'max_tokens': agent_args.max_tokens,
        'do_sample': False,
        'device': agent_args.device,
        'use_8bit': agent_args.use_8bit,
        'rag': {
            'embedding_model': rag_args.embedding_model,
            'seed': rag_args.seed,
            "top_k": rag_args.top_k,
            "order": rag_args.order,
        }
    }
    
    agent = agent_name(config)
    main(agent, bench_cfg)
