
import mlflow
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv
import os
load_dotenv()
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("AgenticAI-LLM-Experiment1")
mlflow.langchain.autolog()
# Load model
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1-Terminus",
    # repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)
model = ChatHuggingFace(llm=llm)
with mlflow.start_run():
    result = model.invoke("Hello")

    print(result)

# import mlflow
# import openai
# import os
# import pandas as pd
# import dagshub
# dagshub.init(repo_owner='krishnaik06', repo_name='MLfLow', mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/krishnaik06/MLfLow.mlflow")
# eval_data = pd.DataFrame(
#     {
#         "inputs": [
#             "What is MLflow?",
#             "What is Spark?",
#         ],
#         "ground_truth": [
#             "MLflow is an open-source platform for managing the end-to-end machine learning (ML) "
#             "lifecycle. It was developed by Databricks, a company that specializes in big data and "
#             "machine learning solutions. MLflow is designed to address the challenges that data "
#             "scientists and machine learning engineers face when developing, training, and deploying "
#             "machine learning models.",
#             "Apache Spark is an open-source, distributed computing system designed for big data "
#             "processing and analytics. It was developed in response to limitations of the Hadoop "
#             "MapReduce computing model, offering improvements in speed and ease of use. Spark "
#             "provides libraries for various tasks such as data ingestion, processing, and analysis "
#             "through its components like Spark SQL for structured data, Spark Streaming for "
#             "real-time data processing, and MLlib for machine learning tasks",
#         ],
#     }
# )
# mlflow.set_experiment("LLM Evaluation")
# with mlflow.start_run() as run:
#     system_prompt = "Answer the following question in two sentences"
#     # Wrap "gpt-4" as an MLflow model.
#     logged_model_info = mlflow.openai.log_model(
#         model="gpt-4",
#         task=openai.chat.completions,
#         artifact_path="model",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": "{question}"},
#         ],
#     )
#
#     # Use predefined question-answering metrics to evaluate our model.
#     results = mlflow.evaluate(
#         logged_model_info.model_uri,
#         eval_data,
#         targets="ground_truth",
#         model_type="question-answering",
#         extra_metrics=[mlflow.metrics.toxicity(), mlflow.metrics.latency(),mlflow.metrics.genai.answer_similarity()]
#     )
#     print(f"See aggregated evaluation results below: \n{results.metrics}")
#
#     # Evaluation result for each data record is available in `results.tables`.
#     eval_table = results.tables["eval_results_table"]
#     df=pd.DataFrame(eval_table)
#     df.to_csv('eval.csv')
#     print(f"See evaluation table below: \n{eval_table}")