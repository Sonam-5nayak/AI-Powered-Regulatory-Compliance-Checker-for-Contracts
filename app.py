# import os
# from dotenv import load_dotenv
# from groq import Groq
# load_dotenv()
# client = Groq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     )
# rag_chunk1="Hey"
# chat_completion = client.chat.completions.create(
#     messages=[

#         {
#             "role": "user",
#              "content": f"{rag_chunk1}",
#         },

#         {
#             "role": "system", 
#             "content": f"You are my personal health assistant help me with my queries",
#         }



#     ],
#     model="llama-3.1-8b-instant",
#     max_tokens=512,
#     temperature=0.3,
#     top_p=1,
#     top_k=40,
    

# )

# print(chat_completion.choices[0].message.content)

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.hrlineup.com/legal-and-complience-checklist-for-hiring/")
docs = loader.load()
print(docs[0])
