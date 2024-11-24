import streamlit as st

# from knowledge_gpt.components.sidebar import sidebar

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

load_dotenv()

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
openai_api_key = os.getenv("OPENAI_API_KEY")

# Enable caching for expensive functions
bootstrap_caching()

if not openai_api_key:
    st.error("OpenAI API key not found in .env file")

model = "gpt-4"

if not is_open_ai_key_valid(openai_api_key, model):
    st.stop()

llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)


@st.cache_resource(show_spinner=False)
def load_data(uploaded_file):
    file = read_file(uploaded_file)
    chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

    with st.spinner("⏳ Indexing document..."):
        folder_index = embed_files(
            files=[chunked_file],
            embedding=EMBEDDING if model != "debug" else "debug",
            vector_store=VECTOR_STORE if model != "debug" else "debug",
            openai_api_key=openai_api_key,
        )
    return folder_index


@st.cache_resource(show_spinner=False)
def prepare_default_answers(_indexed_doc):
    requirements_result = query_folder(
        query="You are a legal expert analyzing an RFP document. Your task is to: \
            - Extract **all** requirements, including those that are implied or stated indirectly. \
            - Organize the requirements by their respective topics as structured in the document. \
            - Present the information in clear, concise bullet points. \
            - Ensure that no requirement is overlooked. \
            - If any statements are ambiguous but may contain requirements, please include them and mark them as 'Potential Requirement - Needs Clarification.'",
        folder_index=_indexed_doc,
        llm=llm,
    )

    company_result = query_folder(
        query="Who is the buyer? What company information can you find of the buyer? Focus on the company's background, capabilities, and relevant experience.",
        folder_index=_indexed_doc,
        llm=llm,
    )

    summary_result = query_folder(
        query="Provide a clear and concise summary of the document's main topic and purpose.",
        folder_index=_indexed_doc,
        llm=llm,
    )

    return [requirements_result, company_result, summary_result]


# Uncomment to enable debug mode
# MODEL_LIST.insert(0, "debug")
st.set_page_config(page_title="T(AI)S", page_icon="📖", layout="wide")

st.title("Tender-GPT")

tab1, tab2, tab3 = st.tabs(["🔼 Upload material", "📃 Tender Overview", "💬 Chat"])

with tab1:
    st.header("Uploaded Files")
    uploaded_file = st.file_uploader(
        "Upload a pdf, docx, or txt file",
        # accept_multiple_files=True,
        type=["pdf", "docx", "txt"],
    )

    # show_full_doc = False

    if not uploaded_file:
        st.stop()

indexed_data = load_data(uploaded_file)
if indexed_data:
    with tab1:
        st.markdown(f"**Uploaded file:** {uploaded_file.name}")

company_info = prepare_default_answers(indexed_data)[1]
requirements = prepare_default_answers(indexed_data)[0]
summary = prepare_default_answers(indexed_data)[2]


# with st.expander("Advanced Options"):
#     return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
#     show_full_doc = st.checkbox("Show parsed contents of the document")


####


with tab2:
    if not uploaded_file:
        st.write(
            "Upload a file in the first tab, and get back to see what the tender is about!"
        )
    else:
        col1, col2 = st.columns(2)

        # with col1:
        #     st.markdown("### Company")
        #     st.markdown(company_info.answer)

        with col1:
            st.markdown("### Summary")
            st.markdown(summary.answer)

        with col2:
            st.markdown("### Tender requirements")
            st.markdown(requirements.answer)

        # with col4:
        #     st.markdown("#### Sources")
        #     for source in requirements.sources:
        #         st.markdown(source.page_content)
        #         formmated_source = source.metadata["source"].split("-")[0]
        #         st.markdown("p. " + formmated_source)
        #         st.markdown("---")
with tab3:
    st.header("Chat with Tender material 📑")

if not uploaded_file:
    st.write("Please upload file.")
else:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Ask me a question about the tender material.",
            }
        ]
    if prompt := st.chat_input("What is this tender about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        response = query_folder(
            folder_index=indexed_data,
            query=prompt,
            return_all=False,
            llm=llm,
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": response.answer}
        )
        st.chat_message("assistant").write(response.answer)

    # if prompt == "Who is the buyer?":
    #     response = query_folder(
    #         query="What company information can you find? Focus on the company's background, capabilities, and relevant experience. ALWAYS begin with with format: The Buyer: [insert company name or organisation name]\n\n Description of buyer: [insert company information].",
    #         folder_index=folder_index,
    #         llm=llm,
    #     ).answer
    # # elif prompt == "What are the requirements of the tender?":
    # #     response = requirements_result
    # elif prompt == "Summarize the content of the file":
    #     response = query_folder(
    #         query="Provide a clear and concise summary of the document's main topic and purpose.",
    #         folder_index=folder_index,
    #         llm=llm,
    #     ).answer

    # response = query_folder(
    #     folder_index=folder_index,
    #     query=prompt,
    #     return_all=False,
    #     llm=llm,
    # ).answer

    # query_folder(query=st.session_state.messages, folder_index=folder_index, llm=llm)
    # st.session_state.messages.append({"role": "assistant", "content": response})

    # user_input = st.text_input("Ask a question about the tender material:")
    # if user_input:
    #     # Placeholder for chatbot logic
    #     st.write("You asked:", user_input)
    #     st.spinner("Looking in the tender material... Hang on.")
    #     company_result = query_folder(
    #         query="What company information can you find? Focus on the company's background, capabilities, and relevant experience.",
    #         folder_index=folder_index,
    #         llm=llm,
    #         )

    # with st.spinner("Indexing document... This may take a while⏳"):
    #     folder_index = embed_files(
    #         files=[chunked_file],
    #         embedding=EMBEDDING if model != "debug" else "debug",
    #         vector_store=VECTOR_STORE if model != "debug" else "debug",
    #         openai_api_key=openai_api_key,
    #     )

    #     company_result = query_folder(
    #         query="What company information can you find? Focus on the company's background, capabilities, and relevant experience.",
    #         folder_index=folder_index,
    #         llm=llm,
    #     )

    # summary_result = query_folder(
    #     query="Provide a clear and concise summary of the document's main topic and purpose.",
    #     folder_index=folder_index,
    #     llm=llm,
    # )

    # requirements_result = query_folder(
    #     query="You are a legal expert analyzing an RFP document. Your task is to: \
    #         - Extract **all** requirements, including those that are implied or stated indirectly. \
    #         - Organize the requirements by their respective topics as structured in the document. \
    #         - Present the information in clear, concise bullet points. \
    #         - Ensure that no requirement is overlooked. \
    #         - If any statements are ambiguous but may contain requirements, please include them and mark them as 'Potential Requirement - Needs Clarification.'",
    #     folder_index=folder_index,
    #     llm=llm,
    # )

    #     col1, col2 = st.columns(2)

    #     with col1:
    #         st.markdown("### Company")
    #         st.markdown(company_result.answer)
    #         # with st.expander("Sources"):
    #         #     for source in company_result.sources:
    #         #         st.markdown(source.page_content)
    #         #         formatted_source = source.metadata["source"].split('-')[0]
    #         #         st.markdown('p. ' + formatted_source)
    #         #         st.markdown("---")

    #     with col2:
    #         st.markdown("### Summary")
    #         st.markdown(summary_result.answer)
    #         # with st.expander("Sources"):
    #         #     for source in summary_result.sources:
    #         #         st.markdown(source.page_content)
    #         #         formatted_source = source.metadata["source"].split('-')[0]
    #         #         st.markdown('p. ' + formatted_source)
    #         #         st.markdown("---")

    # col1, col2 = st.columns(2)

    # with col1:
    #     st.markdown("### Tender requirements")
    #     st.markdown(requirements_result.answer)

    # with col2:
    #     st.markdown("#### Sources")
    #     for source in requirements_result.sources:
    #         st.markdown(source.page_content)
    #         formmated_source = source.metadata["source"].split("-")[0]
    #         st.markdown("p. " + formmated_source)
    #         st.markdown("---")


# with st.form(key="qa_form"):
#     query = st.text_area("Ask a question about the document")
#     submit = st.form_submit_button("Submit")


# if show_full_doc:
#     with st.expander("Document"):
#         # Hack to get around st.markdown rendering LaTeX
#         st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)

# if submit:
#     if not is_query_valid(query):
#         st.stop()

#     # Output Columns
#     answer_col, sources_col = st.columns(2)

#     with st.spinner("Processing your question... ⏳"):

#         result = query_folder(
#             folder_index=folder_index,
#             query=query,
#             return_all=False,
#             llm=llm,
#         )

#         with answer_col:
#             st.markdown("#### Answer")
#             st.markdown(result.answer)

#         with sources_col:
#             st.markdown("#### Sources")
#             for source in result.sources:
#                 st.markdown(source.page_content)
#                 formmated_source = source.metadata["source"].split("-")[0]
#                 st.markdown("p. " + formmated_source)
#                 st.markdown("---")
