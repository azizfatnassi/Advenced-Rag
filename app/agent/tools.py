


from langchain_core.tools import tool
from app.dependencies import get_vectordb
from app.rag.reranker import rerank
import numexpr 


_current_trace=None

def set_trace(trace):
    global _current_trace
    _current_trace= trace


@tool
def search_documents(query:str)-> str:
    """
     Search the financial documents database for relevant information.
     Use this when you need to find specific financial data, numbers, facts,
     or context from compnay annual reports or 10-K fillings .
     Input should be a specific search query about financial data.
    """
    span= None
    try: 

      
        if _current_trace :
         span=_current_trace.span(name="tool-search_documents",input= query) 

        vectordb=get_vectordb()
        chunks= vectordb.similarity_search(query,k=4)
        reranked=rerank(query,chunks,top_k=3)

        if not reranked:
            return "No relevant documents for this query"
        
        results=[]
        for i, chunk in enumerate(reranked):
            company= chunk.metadata.get("company","unknown")
            year= chunk.metadata.get("year","unknown")
            results.append(f"[Source {i+1} - {company} {year}\n{chunk.page_content}]")
        output= "\n\n".join(results)
        if span:
         span.end(output=output)
        return output
    except Exception as e :
        return f"Error searching documents : {str(e)}"
    
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    Use this when you need to calculate growth rates, percentages, ratios,
    or any arithmetic on financial numbers.
    Input must be a valid math expression like '(96773 - 53823) / 53823 * 100'
    Do not include units or currency symbols in the expression."""
    span= None
    try:
        # numexpr is safe — it only evaluates math, not arbitrary Python
        if _current_trace:
            span=_current_trace.span(name="tool-calculate",input=expression)
        result = numexpr.evaluate(expression)
        final= f"{float(result):.4f}"
        if span:
            span.end(output=final)
        return final
    
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}. Make sure the expression contains only numbers and math operators."

@tool 
def get_stock_price(ticker:str)->str:
    """Getthe cureent stock price for company using its ticker symbol.
    Use this when user asks about current stock price or market data,
    Input must be valid stock ticker symbol like 'TSLA','AAPL'
    this tool fetches live data from internet."""
    span=None
    try:
        import yfinance as yf
        if _current_trace:
            span=_current_trace.span(name="tool-price",input=ticker)
        stock=yf.Ticker(ticker.upper().strip())
        info = stock.info

        price=info.get("currentPrice") or info.get("regularMarketPrice")
        currency= info.get("currency","USD")
        name= info.get("shortName",ticker)
        market_cap =info.get("marketCap")

        if not price:
             result=  f"Could not retrieve price for ticker '{ticker}'. Make sure the ticker symbol is correct."
        else:
          result= f"{name} ({ticker.upper()})\n"
          result += f"Current price : {currency} {price}\n"

          if market_cap:
             result += f"Market Cap: {currency} {market_cap:,.0f}"
        if span:
            span.end(output=result)

        return result
    except ImportError:
        return "yfinance library not installed. Run: pip install yfinance"
    except Exception as e:
        return f"Error fetching stock price for '{ticker}': {str(e)}"

