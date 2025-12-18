
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(

  model="openai/gpt-oss-20b",
  messages=[
    {
      "role": "user",
      "content": "A text file with a summary or notes  "
    }
  ]
)

print(completion.choices[0].message.content)
