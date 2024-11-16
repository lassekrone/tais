import streamlit as st

from knowledge_gpt.components.sidebar import sidebar

from knowledge_gpt.ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)

from knowledge_gpt.core.caching import bootstrap_caching

from knowledge_gpt.core.parsing import read_file
from knowledge_gpt.core.chunking import chunk_file
from knowledge_gpt.core.embedding import embed_files
from knowledge_gpt.core.qa import query_folder
from knowledge_gpt.core.utils import get_llm

from dotenv import load_dotenv
import os


EMBEDDING = "openai"
VECTOR_STORE = "faiss"

# Uncomment to enable debug mode
# MODEL_LIST.insert(0, "debug")

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


st.set_page_config(page_title="T(AI)S", page_icon="📖", layout="wide")
# st.header("T(AI)S")

# Enable caching for expensive functions
bootstrap_caching()


if not openai_api_key:
    st.error("OpenAI API key not found in .env file")

uploaded_file = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf", "docx", "txt"],
    help="Scanned documents are not supported yet!",
)

model = 'gpt-4' 

# with st.expander("Advanced Options"):
#     return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
#     show_full_doc = st.checkbox("Show parsed contents of the document")

show_full_doc = False

if not uploaded_file:
    st.stop()

try:
    file = read_file(uploaded_file)
except Exception as e:
    display_file_read_error(e, file_name=uploaded_file.name)

chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

if not is_file_valid(file):
    st.stop()

if not is_open_ai_key_valid(openai_api_key, model):
    st.stop()

llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)

with st.spinner("Indexing document... This may take a while⏳"):
    folder_index = embed_files(
        files=[chunked_file],
        embedding=EMBEDDING if model != "debug" else "debug",
        vector_store=VECTOR_STORE if model != "debug" else "debug",
        openai_api_key=openai_api_key,
    )
    
    company_result = query_folder(
        query="What company information can you find? Focus on the company's background, capabilities, and relevant experience.",
        folder_index=folder_index,
        llm=llm,
    )
    
    summary_result = query_folder(
        query="Provide a clear and concise summary of the document's main topic and purpose.",
        folder_index=folder_index,
        llm=llm,
    )
    
    requirements_result = query_folder(
        query="You are a legal expert analyzing an RFP document. Your task is to: \
            - Extract **all** requirements, including those that are implied or stated indirectly. \
            - Organize the requirements by their respective topics as structured in the document. \
            - Present the information in clear, concise bullet points. \
            - Ensure that no requirement is overlooked. \
            - If any statements are ambiguous but may contain requirements, please include them and mark them as 'Potential Requirement - Needs Clarification.'",
        folder_index=folder_index,
        llm=llm,
    )

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Company")
        st.markdown(company_result.answer)
        # with st.expander("Sources"):
        #     for source in company_result.sources:
        #         st.markdown(source.page_content)
        #         formatted_source = source.metadata["source"].split('-')[0]
        #         st.markdown('p. ' + formatted_source)
        #         st.markdown("---")

    with col2:
        st.markdown("### Summary")
        st.markdown(summary_result.answer)
        # with st.expander("Sources"):
        #     for source in summary_result.sources:
        #         st.markdown(source.page_content)
        #         formatted_source = source.metadata["source"].split('-')[0]
        #         st.markdown('p. ' + formatted_source)
        #         st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Tender requirements")
        st.markdown(requirements_result.answer)

    with col4:
        st.markdown("#### Sources")
        for source in requirements_result.sources:
            st.markdown(source.page_content)
            formmated_source = source.metadata["source"].split('-')[0]
            st.markdown('p. ' + formmated_source)
            st.markdown("---")


with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")


if show_full_doc:
    with st.expander("Document"):
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)

if submit:
    if not is_query_valid(query):
        st.stop()    

    # Output Columns
    answer_col, sources_col = st.columns(2)

    with st.spinner("Processing your question... ⏳"):
        
        result = query_folder(
            folder_index=folder_index,
            query=query,
            return_all=False,
            llm=llm,
        )

            
        with answer_col:
            st.markdown("#### Answer")
            st.markdown(result.answer)

        with sources_col:
            st.markdown("#### Sources")
            for source in result.sources:
                st.markdown(source.page_content)
                formmated_source = source.metadata["source"].split('-')[0]
                st.markdown('p. ' + formmated_source)
                st.markdown("---")
