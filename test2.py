import requests

url = "http://localhost:11434/api/generate"
# data = {
#     "model": "llama3",
#     "prompt": "Hi!",
#     "stream": False
# }
with open("E:\Doc\Bank Credit Rating AI Prompt\Bank Credit Rating AI Prompt\\1. Annual_Report_2024.pdf", "r") as f:
    file_content = f.read()
data = {
    "model": "llama3",
    "prompt": f"Summarize this:\n\n{file_content}",
    "stream": False
}
# data = {
#     "model": "deepseek-v3.1:671b-cloud",
#     "prompt": "Hi"
# }
response = requests.post(url, json=data, stream=True)
print(response.json()["response"])
# for line in response.iter_lines():
#     if line:
#         print(line.decode())
