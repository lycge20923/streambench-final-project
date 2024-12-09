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

            Please analyze briefly step-by-step. Provide only the essential reasoning in short sentences:
            A. ...
            B. ...
            Finally, make a final prediction in the format: <number>. <diagnosis>
            '''

            
        else:
            initial_text = "You are a professional medical doctor. Your task is to analyze the patient's symptoms and provide a diagnosis."
            middle_text = ''
            if few_shot:
                middle_text = f'''
                Below are some simple examples of successful previous predictions with analysis.
            
                {{few_shot_text}}
                '''
            main_text = f'''
            All possible diagnoses for you to choose from are listed below (one diagnosis per line, in the format of <number>. <diagnosis>):
            {self.disease_options}
            
            Here is the case you need to diagnose:
            {self.symptom}
            
            Please follow a concise analysis process, using short sentences for key reasoning steps, and provide the final prediction in this format: <number>. <diagnosis>.
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
                print(Fore.YELLOW + f"Extracted numbers {numbers} is not exactly one. Select the last one." + Style.RESET_ALL)
                prediction = numbers[-1]
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
        self.answer = response + f"\nFinal Answer: {str(prediction)}. {label2desc[int(prediction)]}"
        
        # update the example and result for cot(chain of thought)
        if not self.cot_example or not self.cot_result:
            self.cot_example = self.symptom
            self.cot_result = response
        torch.cuda.empty_cache()
        return prediction

