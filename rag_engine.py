import time
import json
import re
from typing import Any
from urllib import response



class Advanced_Rag:
    def __init__(self,rag_retriever,llm):
        self.rag_retriever = rag_retriever
        self.llm = llm
        self.history = []

    def clear_history(self):
        self.history = []


 # Answers question from the context

    def query(self,question:str,top_k:int=5,score_threshold:float=0.0,stream:bool=True) -> dict[str,Any]:
        results = self.rag_retriever.retrieve(question,top_k,score_threshold)
        if not results:
            return {
                "Query":question,
                "Answer":
                "I cannot find this in the documents.",
                "Sources":[]
            }

        context="\n\n".join(doc["content"]for doc in results)
        sources=[]
        for doc in results:
            sources.append({
                "source":doc["metadata"].get("source_file","unknown"), # if source or source file found then ok otherwise unknown
                "page":doc["metadata"].get("page","unknown"),
                "score":doc.get( "cosine_similarity", 0),
                "preview":doc["content"][:300]+"..."
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
            for i in range(0,len(answer),20): # streaming ans
                print(answer[i:i+20],end="",flush=True)
                time.sleep(0.02)
        return {
            "Query":question,
            "Answer":answer,
            "Sources":sources
        }

 # summary


    def summarize_document(self,topic:str,top_k:int=20):
        results=self.rag_retriever.retrieve(topic,top_k,0.0)
        if not results:
            results=self.rag_retriever.retrieve(topic,40,-1)
            return {
                "topic":topic,
                "summary":
                "I cannot find this topic in the documents."
            }
        context="\n\n".join(doc["content"]for doc in results) # join the context
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
        - kep it accuarate .
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
# quiz


    def generate_quiz(self,topic: str,num_questions: int = 5,difficulty: str = "medium"): # to reduce the token ussage we use 5 questions
        results = self.rag_retriever.retrieve(topic,top_k=20,score_threshold=0.0)
        if not results:
            results = self.rag_retriever.retrieve(topic,top_k=50,score_threshold=-1 )
        print(f"Quiz Topic: {topic}")
        print(f"Retrieved Chunks: {len(results)}")
        if not results:
            return {
                "topic": topic,
                "difficulty": difficulty,
                "quiz": []
            }

        context = "\n\n".join( doc["content"] for doc in results)
        prompt = f"""
        You are an educational quiz generator.

        Task:
        Generate exactly {num_questions} multiple-choice questions from the provided context.

        Topic: {topic}
        Difficulty: {difficulty}

        STRICT FACTUAL RULES:

        1. Use ONLY information explicitly stated in the provided context.
        2. Do NOT use outside knowledge, assumptions, or textbook knowledge.
        3. Every question must be answerable using only the context.
        4. The correct answer must be directly supported by the context.
        5. There must be EXACTLY ONE correct answer.
        6. Before generating a question, verify that:

        * The correct answer is explicitly stated in the context.
        * None of the other options can be considered correct.
        * No reasonable interpretation makes another option partially correct.
        7. If multiple options could be correct, DO NOT generate that question.
        8. Never create questions based on inferred, implied, or commonly known facts or from examples stated there.
        9. If the context contains conflicting information, skip that topic.
        10. If a statement is not explicitly present in the context, treat it as unknown.
        11. Prefer definition, fact, concept, process, formula, and example-based questions that have unambiguous answers.

        ANTI-HALLUCINATION CHECK:

        For every question internally verify:

        * Evidence exists in the context for the correct answer.
        * Evidence does NOT exist for any incorrect option.
        * Another textbook, edition, or external source would not change the answer.
        * The answer remains correct when considering only the provided context.

        If any check fails, discard the question and create a new one.

        QUALITY CHECK:

        Reject questions like:

        "Where can you find information about schedules in database systems?"

        A) Database System Concepts 6th Edition
        B) Database System Concepts 7th Edition
        C) Recovery Techniques
        D) Security Methods

        Reason:
        Multiple options may be correct outside the provided context.

        Instead generate:

        "According to the provided content, which edition discusses schedules in database systems?"

        This anchors the answer directly to the source material.

        Output:
        Return ONLY valid JSON.
        Format:

        [
        {{
            "question": "Question text",
            "options": {{
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
                "D": "Option D"
            }},
            "answer": "A",
            "explanation": "Explanation based only on the provided context."
        }}
        ]

        Context:
        {context}

    """

        quiz = []

       
        for attempt in range(3): # retry for 3 times
            try:
                response = self.llm.invoke(prompt)
                raw = response.content.strip()
                print(f"\nAttempt {attempt+1}")
                print("RAW RESPONSE:")
                print(raw[:1000])
                raw = raw.replace("```json", "")
                raw = raw.replace("```", "")
                raw = raw.strip()
                match = re.search(r"\[\s*\{.*\}\s*\]", raw,re.DOTALL)
                if match:
                    json_text = match.group()
                else:
                    json_text = raw

                quiz = json.loads(json_text)
                if isinstance(quiz, list) and len(quiz) > 0:
                    print(f"Successfully parsed {len(quiz)} questions")
                    break
                quiz = []
            except Exception as e:
                print(f"Quiz Parse Error (Attempt {attempt+1}):",e)
                quiz = []
        return {
            "topic": topic,
            "difficulty": difficulty,
            "quiz": quiz
        }