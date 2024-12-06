import os, sys, re
from colorama import Fore, Style
import torch

sys.path.append(os.getcwd())
from utils import strip_all_lines
from code.local_model import LLMModelAgent

class SQLGenerationAgent(LLMModelAgent):
    """
    An agent that generates SQL code based on the given table schema and the user query.
    """
    def __init__(self, config):
        super().__init__(config)
        self.table_schema = ''
        self.user_query = ''

    def design_prompt(self, few_shot = False):
        if not self.cot_example or not self.cot_result:
            prompt = f'''
            You are performing prompt-to-SQL task
            The Table Schema is:
            {self.table_schema}
            
            The user query is:{self.user_query}
            
            You have to 
            1. analyze step-by-step, explain your reasoning
            A. ...
            B. ...
            2. Then, generate the SQL code in the format of ```sql\n<your_SQL_code>\n```:
            '''
        else:
            initial_text = 'You are a professional SQL programmer. Your job is to generate SQL code based on user query.'
            middle_text = ''
            if few_shot:
                middle_text = f'''
                Below are some simple examples of successful previous predictions.
                
                {{few_shot_text}}
                '''
            main_text = f'''
            Here is a case:
            {self.cot_example}
            
            The analysis and the prediction is:
            {self.cot_result}
            
            Here is another case:
            1. Table Schema:
            {self.table_schema}
            2. User query: {self.user_query}
            
            After analysis(don't print the process, just think in your mind), Generate the correct SQL code directly in the following format:```sql\n<your_SQL_code>\n```
            '''
            prompt = initial_text + middle_text + main_text
        return strip_all_lines(prompt.strip())

    @staticmethod
    def parse_sql(pred_text: str) -> str:
        """
        Parse the SQL code from the LLM's response.
        """
        pattern = r"```sql([\s\S]*?)```"
        match = re.search(pattern, pred_text)
        if match:
            sql_code = match.group(1)
            sql_code = sql_code.strip()
            return sql_code
        else:
            print(Fore.RED + "No SQL code found in the response" + Style.RESET_ALL)
            sql_code = pred_text
        return sql_code
    
    def __call__(self, table_schema: str, user_query: str) -> str:
        self.reset_log_info()
        self.table_schema = table_schema
        self.user_query = user_query
        
        # get the response from the prompt, 
        # which has been designed in the design_prompt
        response = self.get_llm_response(query=user_query)
        
        # mapping 
        sql_code = self.parse_sql(response)
        
        # generate for update
        self.question = user_query
        self.answer = f"```sql\n{sql_code}\n```"
        if not self.cot_example or not self.cot_result:
            self.cot_example = f'''
                1. Table Schema:
                {table_schema}
                2. User query: {user_query}
                '''
            self.cot_result = response
        torch.cuda.empty_cache()
        return sql_code
