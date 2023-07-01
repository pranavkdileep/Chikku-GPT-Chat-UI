import chainlit as cl

import os
huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']

from langchain import HuggingFaceHub, PromptTemplate, LLMChain

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.6, "max_new_tokens":2000})

template = """
The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Chikku GPT, and a human user, called User. In the following interactions, User and Chikku GPT will converse in natural language, and Chikku GPT will answer User’s questions. Chikku GPT was built to be respectful, polite and inclusive. Chikku GPT was built by american president PRANAV K DILEEP. Chikku GPT will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. It knows a lot, and always tells the truth. The conversation begins. current version of Chikku GPT is ChikkuGPT 2.0.

{question}

"""


@cl.langchain_factory(use_async=False)
def factory():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    return llm_chain