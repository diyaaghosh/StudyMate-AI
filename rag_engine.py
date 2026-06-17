import time
import json
import re
from typing import Any



class Advanced_Rag:


    def __init__(
        self,
        rag_retriever,
        llm
    ):

        self.rag_retriever = rag_retriever
        self.llm = llm
        self.history = []



    def clear_history(self):

        self.history = []



    # ==========================
    # QUESTION ANSWERING
    # ==========================

    def query(
        self,
        question:str,
        top_k:int=5,
        score_threshold:float=0.0,
        stream:bool=True
    ) -> dict[str,Any]:


        results = self.rag_retriever.retrieve(
            question,
            top_k,
            score_threshold
        )


        if not results:

            return {

                "Query":question,
                "Answer":
                "I cannot find this in the documents.",
                "Sources":[]

            }



        context="\n\n".join(

            doc["content"]

            for doc in results

        )



        sources=[]


        for doc in results:

            sources.append({

                "source":
                doc["metadata"].get(
                    "source_file",
                    "unknown"
                ),

                "page":
                doc["metadata"].get(
                    "page",
                    "unknown"
                ),

                "score":
                doc.get(
                    "cosine_similarity",
                    0
                ),

                "preview":
                doc["content"][:300]+"..."

            })




        prompt=f"""

You are an educational AI assistant.
Answer the question using the document context.
Instructions:
- First identify the relevant section of the document.
- Understand the concept before answering.
- Answer in your own words.
- Do not copy sentences blindly.
- Use related explanations from the document.
- Do not mix unrelated topics.
Answer style:
- Definition questions:
  Give a clear definition first, then explain.
- Difference questions:
  Use a comparison table.
- Step questions:
  Give ordered steps.
- Example questions:
  Give examples only if available or logically explained from the document.
Important:
The document may explain a concept without using the exact wording of the question.
In that case, infer the answer from the relevant section.
Only say:
"I cannot find this in the documents."
when the concept is not covered at all.
Context:
{context}
Question:
{question}
Answer:

"""


        response=self.llm.invoke(prompt)

        answer=response.content



        if stream:

            for i in range(0,len(answer),20):

                print(
                    answer[i:i+20],
                    end="",
                    flush=True
                )

                time.sleep(0.02)



        return {

            "Query":question,

            "Answer":answer,

            "Sources":sources

        }





    # ==========================
    # SUMMARY
    # ==========================


    def summarize_document(
        self,
        topic:str,
        top_k:int=20
    ):


        results=self.rag_retriever.retrieve(

            topic,

            top_k,

            0.0

        )


        if not results:

            results=self.rag_retriever.retrieve(

                topic,

                40,

                -1

            )



        if not results:

            return {

                "topic":topic,

                "summary":
                "I cannot find this topic in the documents."

            }



        context="\n\n".join(

            doc["content"]

            for doc in results

        )



        prompt=f"""
You are an educational summarizer.
Summarize the given document content.
Rules:
- Use only provided content.
- Do not add outside knowledge.
- Include important definitions.
- Include important points.
- Keep it exam friendly.
- Keep steps in order.
- Keep comparisons if present.
Topic:
{topic}
Content:
{context}
Summary:

"""


        response=self.llm.invoke(prompt)



        return {

            "topic":topic,

            "summary":response.content

        }





    # ==========================
    # QUIZ
    # ==========================


    def generate_quiz(
        self,
        topic:str,
        num_questions:int=10,
        difficulty:str="medium"
    ):


        results=self.rag_retriever.retrieve(

            topic,

            top_k=20,

            score_threshold=0.0

        )



        if not results:


            results=self.rag_retriever.retrieve(

                topic,

                top_k=50,

                score_threshold=-1

            )



        if not results:

            return {

                "topic":topic,

                "difficulty":difficulty,

                "quiz":[]

            }



        context="\n\n".join(

            doc["content"]

            for doc in results

        )




        prompt=f"""

Create {num_questions} MCQ quiz.

Topic:
{topic}

Difficulty:
{difficulty}


Use ONLY this content.


Return ONLY JSON.

Format:

[
{{
"question":"",
"options":
{{
"A":"",
"B":"",
"C":"",
"D":""
}},
"answer":"A",
"explanation":""
}}
]


Content:

{context}

"""


        response=self.llm.invoke(prompt)

        raw=response.content



        quiz=[]


        try:


            match=re.search(

                r"\[.*\]",

                raw,

                re.DOTALL

            )


            if match:

                quiz=json.loads(
                    match.group()
                )


        except Exception as e:

            print(
                "Quiz JSON error:",
                e
            )

            quiz=[]



        return {

            "topic":topic,

            "difficulty":difficulty,

            "quiz":quiz

        }