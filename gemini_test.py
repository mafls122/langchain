import streamlit as st
import tiktoken
from loguru import logger
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def main():
    st.set_page_config(
    page_title="enAI_TF3",
    page_icon=":balloon:")

    st.title("엔코아 AI 프로젝트 :blue[@TF3팀(gemini pro)]:balloon:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
        
    #왼쪽 사이드바
    with st.sidebar:
        uploaded_files =  st.file_uploader("참고 파일을 업로드 해주세요",type=['pdf','docx','pptx'],accept_multiple_files=True)
        api_key = st.text_input("Google API Key 입력", key="chatbot_api_key", type="password")
        pinecone_api_key = st.text_input("Pinecone API Key 입력", key="pinecone_api_key", type="password")
        process = st.button("Process")
    if process:
        if not api_key:
            st.info("API Key를 입력해주세요.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        # vetorestore = get_vectorstore(text_chunks)
        vetorestore = get_pinecone_vectorstore(text_chunks, api_key, pinecone_api_key)

        st.session_state.conversation = get_conversation_chain(vetorestore,api_key)

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금한 부분을 질문 해주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("첨부한 문서에서 찾아보고 있습니다!"):
                result = chain({"question": query})

                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']

                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    st.markdown(source_documents[3].metadata['source'], help = source_documents[3].page_content)



# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):

    doc_list = []

    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                       model_name="jhgan/ko-sroberta-multitask",
                                       model_kwargs={'device': 'cpu'},
                                       encode_kwargs={'normalize_embeddings': True}
                                        )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_pinecone_vectorstore(text_chunks, llm_api_key, pinecone_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=llm_api_key)
    
    pinecone.init(
        api_key=pinecone_key,
        environment="gcp-starter"
    )

    # 생성할 인덱스명
    idx_nm = "testp"

    if idx_nm not in pinecone.list_indexes():
        pinecone.create_index(name=idx_nm, metric="cosine", dimension=768)
        # 파라미터 : 인덱스명, 계산방법, 차원

    vectordb = Pinecone.from_documents(text_chunks, embeddings, index_name=idx_nm)

    return vectordb

def get_conversation_chain(vetorestore, api_key):

    llm = ChatGoogleGenerativeAI(google_api_key=api_key,model="gemini-pro",temperature = 0,convert_system_message_to_human=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="refine", #stuff
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True),
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )


    return conversation_chain

if __name__ == '__main__':
    main()
