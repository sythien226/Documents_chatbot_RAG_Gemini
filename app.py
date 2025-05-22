# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import PromptTemplate
# #from langchain.chat_models import ChatGoogleGenerativeAI
# from langchain_google_genai import ChatGoogleGenerativeAI
# #from langchain.embeddings import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from htmlTemplates1 import css, bot_template, user_template
# import os

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n", " ", ""],
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vectorstore(text_chunks, api_key):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore

# def get_conversation_chain(vectorstore, api_key):
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3, google_api_key=api_key)

#     memory = ConversationBufferMemory(memory_key="chat_history", return_message=True)

#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": prompt}  # Áp dụng prompt template ở đây
#     )
#     return conversation_chain

# # def handle_userinput(user_question):
# #     response = st.session_state.conversation.invoke({'question': user_question})
# #     st.write(response)
# #     st.session_state.chat_history = response['chat_history']
# #     for i, message in enumerate(st.session_state.chat_history):
# #         if i % 2 == 0:
# #             st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
# #         else:
# #             st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# def handle_userinput(user_question):
#     # Gọi chain, chain tự dùng ConversationBufferMemory để lưu lịch sử
#     response = st.session_state.conversation({'question': user_question})

#     # Hiển thị câu trả lời
#     answer = response.get('answer', "Answer not available in context.")

#     # Lấy lịch sử chat từ ConversationBufferMemory
#     messages = st.session_state.conversation.memory.chat_memory.messages

#     for message in messages:
#         if message.type == "human":
#             st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# # def main():
# #     load_dotenv()
# #     api_key = os.getenv("GOOGLE_API_KEY")
# #     st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
# #     st.write(css, unsafe_allow_html=True)

# #     if "conversation" not in st.session_state:
# #         st.session_state.conversation = None
# #     if "chat_history" not in st.session_state:
# #         st.session_state.chat_history = None

# #     st.header("Chat with multiple PDFs :books:")
# #     user_question = st.text_input("Ask a question about your documents:")
# #     if user_question:
# #         handle_userinput(user_question)

# #     with st.sidebar:
# #         st.subheader("Your documents")
# #         pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
# #         if st.button("Process"):
# #             with st.spinner("Processing..."):
# #                 raw_text = get_pdf_text(pdf_docs)
# #                 text_chunks = get_text_chunks(raw_text)
# #                 vectorstore = get_vectorstore(text_chunks, api_key)
# #                 st.session_state.conversation = get_conversation_chain(vectorstore, api_key)
# #                 st.success("Done")
# # if __name__ == '__main__':
# #     main()

# def main():
#     load_dotenv()
#     api_key = os.getenv("GOOGLE_API_KEY")

#     st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
#     st.write(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None

#     st.header("Chat with multiple PDFs :books:")
#     user_question = st.text_input("Ask a question about your documents:")

#     if user_question:
#         if st.session_state.conversation is not None:
#             handle_userinput(user_question)
#         else:
#             st.warning("Please upload and process your PDFs first.")

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
#         if st.button("Process"):
#             if pdf_docs:
#                 with st.spinner("Processing..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_text_chunks(raw_text)
#                     vectorstore = get_vectorstore(text_chunks, api_key)
#                     st.session_state.conversation = get_conversation_chain(vectorstore, api_key)
#                     st.success("Done")
#             else:
#                 st.warning("Please upload at least one PDF file.")

# if __name__ == '__main__':
#     main()

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import messages_from_dict
import os
from htmlTemplates1 import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],  # bỏ "" separator không hợp lệ
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, api_key):
    prompt_template = """
Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
provided context just say, "answer is not available in the context", don't provide the wrong answer
Context:\n {context}?\n
Question: \n{question}\n

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3, google_api_key=api_key)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return conversation_chain

def handle_userinput_with_manual_history(user_question):
    # Khởi tạo nếu chưa có lịch sử dạng dict
    if "chat_history_dict" not in st.session_state:
        st.session_state.chat_history_dict = []

    # Thêm câu hỏi mới
    st.session_state.chat_history_dict.append({"type": "human", "data": {"content": user_question}})

    # Chuyển lịch sử dict thành list message object
    chat_history_messages = messages_from_dict(st.session_state.chat_history_dict)

    # Gọi chain, truyền question + chat_history dạng message object
    response = st.session_state.conversation.invoke({
        "question": user_question,
        "chat_history": chat_history_messages
    })

    answer = response.get("answer", "Answer not available in context.")

    # Thêm câu trả lời vào lịch sử
    st.session_state.chat_history_dict.append({"type": "ai", "data": {"content": answer}})

    # Hiển thị toàn bộ lịch sử
    for msg in st.session_state.chat_history_dict:
        if msg["type"] == "human":
            st.write(user_template.replace("{{MSG}}", msg["data"]["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg["data"]["content"]), unsafe_allow_html=True)

def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history_dict" not in st.session_state:
        st.session_state.chat_history_dict = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        if st.session_state.conversation is not None:
            handle_userinput_with_manual_history(user_question)
        else:
            st.warning("Please upload and process your PDFs first.")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks, api_key)
                    st.session_state.conversation = get_conversation_chain(vectorstore, api_key)
                    # Reset lịch sử chat khi nạp tài liệu mới
                    st.session_state.chat_history_dict = []
                    st.success("Processing done! You can ask questions now.")
            else:
                st.warning("Please upload at least one PDF file before processing.")

if __name__ == '__main__':
    main()
