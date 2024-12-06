import os, sys, random, re
from colorama import Fore, Style
import torch

sys.path.append(os.getcwd())
from code.local_model import LLMModelAgent
from utils import strip_all_lines

class ClassificationAgent(LLMModelAgent):
    """
    An agent that classifies text into one of the labels in the given label set.
    """
    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.disease_options = ''
        self.symptom = ''
    
    def design_prompt(self, few_shot:bool=False) -> str:
        if not self.cot_example or not self.cot_result:
            prompt = f'''
            You have the following list of disease options:
            {self.disease_options}
            
            You have the following patient information:
            {self.symptom}
            
            Please analyze step-by-step, explain your reasoning:
            A. ...
            B. ...
            Finally, make a final prediction(in the format of  <number>. <diagnosis>):
            '''
        else:
            initial_text = "You are a professional medical doctor. Your work is analyze patient's symptom and give diagnosis."
            middle_text = ''
            if few_shot:
                middle_text = f'''
                Below are some simple examples of successful previous predictions.
            
                {{few_shot_text}}
                '''
            main_text = f'''
            All possible diagnoses for you to choose from are as follows (one diagnosis per line, in the format of <number>. <diagnosis>):
            {self.disease_options}
            
            Here is a case:
            {self.cot_example}
            
            The analysis and the prediction is:
            {self.cot_result}
            
            Here is another case:
            {self.symptom}
            
            After analysis(don't print the process, just think in your mind), the prediction is (in the format of  <number>. <diagnosis>):
            '''
            prompt = initial_text + middle_text + main_text
        
        return strip_all_lines(prompt.strip())
    
    @staticmethod
    def extract_label(pred_text: str, label2desc: dict[str, str]) -> str:
        numbers = re.findall(pattern=r"(\d+)", string=pred_text)
        if len(numbers) == 1:
            number = numbers[0]
            if int(number) in label2desc:
                prediction = number
            else:
                print(Fore.RED + f"Prediction {pred_text} not found in the label set. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))
        else:
            if len(numbers) > 1:
                print(Fore.YELLOW + f"Extracted numbers {numbers} is not exactly one. Select the first one." + Style.RESET_ALL)
                prediction = numbers[0]
            else:
                print(Fore.RED + f"Prediction {pred_text} has no extracted numbers. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))
        return str(prediction)

    def __call__(self, label2desc: dict[str, str], text:str) -> str:
        self.reset_log_info()
        self.disease_options = '\n'.join([f"{str(k)}. {v}" for k, v in label2desc.items()])
        self.symptom = text

        # get the response from the prompt, 
        # which has been designed in the design_prompt
        response = self.get_llm_response(query=self.symptom)
        
        # mapping
        prediction = self.extract_label(response, label2desc)
        
        # generate for update
        self.question = self.symptom
        self.answer = f"{str(prediction)}. {label2desc[int(prediction)]}"
        
        # update the example and result for cot(chain of thought)
        if not self.cot_example or not self.cot_result:
            self.cot_example = self.symptom
            self.cot_result = response
        torch.cuda.empty_cache()
        return prediction

