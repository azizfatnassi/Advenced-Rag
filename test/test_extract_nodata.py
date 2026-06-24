import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.rag.extraction import extract_financial_data


context = """
Tesla's headquarters are located in Austin, Texas. The company was founded in 2003
and is led by CEO Elon Musk. Tesla operates several Gigafactories around the world.
"""

result = extract_financial_data(context)
print(result)
print(result.model_dump())