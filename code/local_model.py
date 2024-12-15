import os, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from colorama import Fore, Style
from abc import abstractmethod

sys.path.append(os.getcwd())
from base import Agent
from utils import RAG, strip_all_lines

class LLMModelAgent(Agent):
    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.config = config
        self.model = self._initialize_model()
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.rag = RAG(self.config["rag"])
        
        # store for prompt
        self.question, self.answer = "", ""
        
        # set the max token for the output
        self.max_token = self.config["max_tokens"]
        
    def _initialize_model(self) -> AutoModelForCausalLM:
        """
        Initialize the LLM model
        """
        kwargs = {
            "device_map": self.config["device"], 
            "pretrained_model_name_or_path": self.config["model_name"]
        }
        if self.config["use_8bit"]:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_has_fp16_weight=False
            )
        else:
            kwargs["torch_dtype"] = torch.float16
        return AutoModelForCausalLM.from_pretrained(**kwargs)

    @abstractmethod 
    def design_prompt(self, few_shot:bool=False) ->str:
        '''
        Write a function to do the prompt designing
        '''
        raise NotImplementedError
    
    @staticmethod
    def get_shot_template() -> str:
        '''
        Store the questions and answers for RAG
        '''
        prompt = f"""\
        Question: {{question}}
        Answer: 
        {{answer}}"""
        return strip_all_lines(prompt)
    
    def generate_response(self, messages: list) -> str:
        text_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_chat], return_tensors="pt").to(self.model.device)
        
        # output the tokenized ids 
        # it would include the problem and the answer
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.config["max_tokens"],
            do_sample=False
        )
        # extract only the output content
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def update(self, correctness: bool) -> bool:
        '''
        store the positive response in RAG
        '''
        if correctness:
            chunk = self.get_shot_template().format(question=self.question, answer=self.answer)
            self.rag.insert(key=self.question, value=chunk)
        return correctness

    def get_llm_response(self, query:str) -> str:
        # design the prompt
        prompt_zero_shot = self.design_prompt(few_shot=False)
        prompt_few_shot = self.design_prompt(few_shot=True)
        
        # extract from RAG
        shots = self.rag.retrieve(query=query, top_k=self.rag.top_k) if self.rag.insert_acc > 0 else []
        # print(shots)
        
        # set the user prompt for each case, depends on some conditions
        if len(shots):
            few_shot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
            try:
                user_prompt = re.sub(pattern=r"\{few_shot_text\}",  repl=few_shot_text, string=prompt_few_shot)
                # print(user_prompt)
            except Exception as e:
                error_msg = f"Error ```{e}``` caused by these shots. Using the zero-shot prompt."
                print(Fore.RED + error_msg + Fore.RESET)
                user_prompt = prompt_zero_shot
        else:
            print(Fore.YELLOW + "No RAG shots found. Using zeroshot prompt." + Fore.RESET)
            user_prompt = prompt_zero_shot
        
        # modified for llm
        messages = [{"role": "user", "content": user_prompt}]
        
        # get prediction
        response = self.generate_response(messages)
        
        # update log
        self.update_log_info(log_data={
            "num_input_tokens": len(self.tokenizer.encode(user_prompt)),
            "num_output_tokens": len(self.tokenizer.encode(response)),
            "num_shots": str(len(shots)),
            "input_pred": user_prompt,
            "output_pred": response,
        })
        
        return response