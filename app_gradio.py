import gradio as gr
import os
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

prompt_text = "The following is a conversation with an Coding AI assistant. The assistant is helpful, creative, clever, and very friendly."
huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.6, "max_new_tokens":2000})

template = """
The following is a Q&A between a highly knowledgeable and intelligent AI assistant, called Chikku GPT, and Humans. In the following interactions, User and Chikku GPT will converse in natural language, and Chikku GPT will answer Humans questions also generate computer programes. Chikku GPT was built to be respectful, polite and inclusive. Chikku GPT was built by american president PRANAV K DILEEP. Chikku GPT will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. It knows a lot, and always tells the truth. The conversation begins. current version of Chikku GPT is ChikkuGPT 2.0.

{question}

"""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

def run(question):
    return llm_chain.run(question)

def chatgpt_clone(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    
    output = run(input)
    history.append((input, output))
    return history, history


block = gr.Blocks()


with block:
    gr.Markdown("""<h1><center>Chikku GPT Code Generator</center></h1>
    """)
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder=prompt_text)
    state = gr.State()
    submit = gr.Button("SEND")
    submit.click(chatgpt_clone, inputs=[message, state], outputs=[chatbot, state], api_name="bot")

block.launch()