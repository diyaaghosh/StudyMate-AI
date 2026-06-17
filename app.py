import streamlit as st

from rag_engine import Advanced_Rag
from retriever import RAGRetriever
from VectorStore import vector_store
from embedding import Embedding_Manager
from llm import get_llm
from ingestion import process_pdfs



st.set_page_config(
    page_title="PDF AI Assistant",
    layout="wide"
)



# ==========================
# LOAD PIPELINE
# ==========================


@st.cache_resource
def load_pipeline():


    embedding_manager = Embedding_Manager()


    store = vector_store()


    retriever = RAGRetriever(

        store,

        embedding_manager

    )


    llm = get_llm()



    rag = Advanced_Rag(

        rag_retriever=retriever,

        llm=llm

    )


    return rag



rag = load_pipeline()




# ==========================
# NEW SESSION RESET
# ==========================


if "initialized" not in st.session_state:


    st.session_state.initialized = True


    rag.clear_history()


    try:

        rag.rag_retriever.vector_store.clear()

        print(
            "Old vectors cleared"
        )


    except Exception as e:

        print(
            "Clear skipped:",
            e
        )





# ==========================
# SIDEBAR UPLOAD
# ==========================


st.sidebar.title(
    "📂 Upload PDFs"
)



pdfs = st.sidebar.file_uploader(

    "Upload PDF files",

    type=["pdf"],

    accept_multiple_files=True

)




if pdfs:


    if st.sidebar.button(
        "Process Documents"
    ):


        with st.spinner(
            "Creating embeddings..."
        ):


            process_pdfs(

                pdfs,

                rag.rag_retriever

            )


        st.sidebar.success(

            f"{len(pdfs)} PDFs processed"

        )






# ==========================
# MAIN
# ==========================


st.title(
    "📚 PDF AI Assistant"
)



tabs = st.tabs(

    [
        "💬 Chat",
        "📄 Summary",
        "📝 Quiz"
    ]

)





# ==========================
# CHAT
# ==========================


with tabs[0]:


    question = st.chat_input(

        "Ask from your PDFs..."

    )



    if question:


        with st.chat_message(
            "user"
        ):

            st.write(question)



        with st.chat_message(
            "assistant"
        ):



            result = rag.query(

                question,

                top_k=5,

                stream=False,

                # summarize=False

            )



            st.write(

                result["Answer"]

            )



            with st.expander(
                "📚 Sources"
            ):



                for src in result["Sources"]:


                    st.write(

f"""
📄 {src['source']}

Page:
{src['page']}

Similarity:
{src['score']:.3f}


{src['preview']}
"""

                    )










# ==========================
# SUMMARY
# ==========================

with tabs[1]:


    topic = st.text_input(

        "Enter topic or PDF name",

        key="summary_topic"

    )


    if st.button(

        "Generate Summary",

        key="summary_btn"

    ):


        with st.spinner(
            "Creating summary..."
        ):


            result = rag.summarize_document(

                topic,

                top_k=20

            )



        st.subheader(
            "Summary"
        )


        st.write(

            result["summary"]

        )
# ==========================
# QUIZ
# ==========================


with tabs[2]:


    quiz_topic = st.text_input(

        "Quiz topic",

        key="quiz_topic"

    )



    difficulty = st.selectbox(

        "Difficulty",

        [
            "easy",
            "medium",
            "hard"
        ],

        key="difficulty"

    )



    if st.button(

        "Generate Quiz",

        key="quiz_btn"

    ):



        quiz = rag.generate_quiz(

            topic=quiz_topic,

            num_questions=10,

            difficulty=difficulty

        )


        st.session_state.quiz = quiz





    # display quiz if exists

    if "quiz" in st.session_state:


        quiz = st.session_state.quiz



        if not quiz["quiz"]:


            st.error(
                "No quiz content found in documents."
            )



        else:


            st.subheader(

                f"{quiz['topic']} Quiz"

            )



            for i,q in enumerate(

                quiz["quiz"]

            ):



                st.markdown(

                    f"### Question {i+1}"

                )


                answer = st.radio(

                    q["question"],

                    [

                        "A) " + q["options"]["A"],

                        "B) " + q["options"]["B"],

                        "C) " + q["options"]["C"],

                        "D) " + q["options"]["D"]

                    ],

                    key=f"question_{i}"

                )




                if st.button(

                    "Check Answer",

                    key=f"check_{i}"

                ):



                    selected = answer[0]



                    if selected == q["answer"]:


                        st.success(
                            "Correct ✅"
                        )


                    else:


                        st.error(

                            f"Wrong ❌ Correct answer: {q['answer']}"

                        )



                    st.info(

                        q["explanation"]

                    )