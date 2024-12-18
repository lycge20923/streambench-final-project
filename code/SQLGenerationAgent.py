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
        if not few_shot:
            prompt = f"""\
            {self.table_schema}
            
            -- Using valid SQLite, answer the following question for the tables provided above.
            -- Question: {self.user_query}
            
            Now, generate the correct SQL code directly in the following format:
            ```sql\n<your_SQL_code>\n```"""
        else:
            prompt = f'''
            You are performing the text-to-SQL task. Here are some examples:
            
            {{few_shot_text}}
            
            Now it's your turn.
            
            -- SQL schema: {self.table_schema}
            -- Using valid SQLite, answer the following question for the SQL schema provided above.
            -- Question: {self.user_query}
            
            Now, generate the correct SQL code directly in the following format:
            ```sql\n<your_SQL_code>\n```
            '''
        return strip_all_lines(prompt)

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
        torch.cuda.empty_cache()
        return sql_code
