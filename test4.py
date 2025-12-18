from langgraph.graph import StateGraph, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import getpass
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
# 1. Define your state
class State(dict):
    input:str
    output:str
    pass

# 2. Initialize DeepSeek model
load_dotenv()
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")
# print(  os.environ["HUGGINGFACEHUB_API_TOKEN"])

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1-Terminus",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

model = ChatHuggingFace(llm=llm)

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that can chat with me about the finacial reporting ",
#     ),
#     ("human", "My salary was 52000 and now salary is 10000. Can you tell the increasing ratio of this"),
# ]
# ai_msg = chat_model.invoke(messages)
# print(ai_msg.content)
class LLMState(TypedDict):

    question: str
    answer: str

def llm_qa(state: LLMState) -> LLMState:

    # extract the question from state
    question = state['question']

    # form a prompt
    prompt = f'Answer the following question {question}'

    # ask that question to the LLM
    answer = model.invoke(prompt).content

    # update the answer in the state
    state['answer'] = answer

    return state
# create our graph

graph = StateGraph(LLMState)

# add nodes
graph.add_node('llm_qa', llm_qa)

# add edges
graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa', END)

# compile
workflow = graph.compile()
# execute

intial_state = {'question': """ May I upload file here? 
"""}

final_state = workflow.invoke(intial_state)

print(final_state['answer'])