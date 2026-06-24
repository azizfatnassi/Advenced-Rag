
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.rag.extraction import extract_financial_data

context="""  Tesla's total revenue in fiscal year 2023 was $96,773 million, an increase from
the prior year. Tesla spent $3,969 million on research and development in 2023.
Net income for the year ended December 31, 2023 was $14,974 million."""

result=extract_financial_data(context)
print(result)
print(result.model_dump())