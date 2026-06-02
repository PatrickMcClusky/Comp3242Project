from ollama import chat
import json
import modelModify

something = '{"postcode_mlm": {"prediction": {"deleted_at": "assATckkFsm", "email": "4347"}, "version": {"endpoint": "uuid", "data": 630.54}}, "source": [{"embedding_key": "nHizAypMYxXz"}, null]}'

response = chat(
    model="deepseek-r1:1.5b",
    messages=[
        {
            "role": "user",
            "content": "with minimal thinking summarize the following json in english: " + something
        }
    ]
)

chatResponse = response.message.content


content = f"""
You are a JSON generator.

Return ONLY JSON.

Do NOT write any comments.

Return ONLY JSON.

Task:
Turn the description of a json file into valid json
{chatResponse}
"""
modelModify.create_valid_json(content)
