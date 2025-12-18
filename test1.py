from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()
model =  ChatOpenAI(model="gpt-5")

# create a state

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

intial_state = {'question': 'Company Description** - Get initial profile of the company from its website: "(https://www.bestholdings.com.bd/)" - Provide a brief history of the company, including year of establishment, core business activities, and mission.'}

final_state = workflow.invoke(intial_state)

print(final_state['answer'])