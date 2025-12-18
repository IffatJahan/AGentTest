from typing import Annotated
from dotenv import load_dotenv
load_dotenv()
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1-Terminus",
    # repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)
# model =  ChatOpenAI(model="gpt-5")
model = ChatHuggingFace(llm=llm)

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

intial_state = {'question': 'distance between MOON and earth'
                    }

# final_state = workflow.invoke(intial_stat)
#
# print(final_state['answer'])
# Use the stream method to get incremental states
for message_chunk, metadata in workflow.stream(
    intial_state,
    stream_mode="messages",
):
    if message_chunk.content:
        print(message_chunk.content, end=" ", flush=True)

print("\n--- Stream complete ---")