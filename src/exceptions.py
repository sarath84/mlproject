import sys
import logging
from src.logger import logging

def error_message_details(error,error_detail:sys):
    
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error ocured in the python script[{0}] , Errror occured in the line number[{1}] , Errror message [{2}] ".format(
       file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message

class CustomException:
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,error_detail=error_details)
    def __str__(self):
       return self.error_message
   
      
