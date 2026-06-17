import streamlit as st
import requests

API_URL="http://127.0.0.1:8001"

st.set_page_config(page_title="Finance RAG",layout="wide")
st.title("Finance Document Assistant")

if "messages" not in st.session_state:
    st.session_state.messages=[]
if "session_id" not in st.session_state:
    st.session_state.session_id= "default"

with st.sidebar:
    st.header("Upload Document")

    company= st.text_input("Company name",value=" Tesla")
    year = st.text_input("Year",value="2023")

    uploaded_file= st.file_uploader("Choose a PDF ", type="pdf")

    if uploaded_file and st.button("Upload & Ingest"):
        with st.spinner("Ingesting Document"):
            response= requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file,"application/pdf")},
                params={"company": company, "year":year}
                )
            
            if response.status_code==200:
                st.success("Document uploaded successfully")
            else:
                st.error(f"Upload failed: {response.text}")

st.divider()
st.header("Session")
st.write(f"Session ID : '{st.session_state.session_id}'")

new_session= st.text_input("Change session ID", value = "default")
if st.button("Apply Session"):
    st.session_state.session_id= new_session
    st.session_state.messages= []
    st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask about your document ..."):

   st.session_state.messages.append({"role":"user","content":prompt})
   with st.chat_message("user"):
       st.write(prompt)

   
   with st.spinner("Thinking .."):
       response = requests.post(f"{API_URL}/chat/memory",
                                params={
                                    "question":prompt,
                                    "session_id":st.session_state.session_id
                                }
       )

   if response.status_code== 200:
       data=response.json()
       answer= data["answer"]
       sources= data.get("sources",[])

       st.session_state.messages.append({"role": "assistant", "content": answer})
       with st.chat_message("assistant"):
            st.write(answer)
            
            if sources: 
                with st.expander("View Sources"):
                  for i,source in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}** — {source['company']} {source['year']} (page {source['page']})")
                    st.caption(source["content"])
                    st.divider()

   else:
        st.error("Something went wrong. Is FastAPI running?")   


if st.button("Clear Chat"):
    requests.delete(f"{API_URL}/chat/{st.session_state.session_id}")
    st.session_state.messages=[]
    st.rerun()
       