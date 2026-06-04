from ollama import chat
import json
import modelModify
import jsonGenerator

while True:
    try:
        something = json.dumps(jsonGenerator.generate_complex_json(num_samples=1, max_depth=2, max_keys=4)[0])

        print("Original json: " + something)
        response = chat(
            model="deepseek-r1:1.5b",
            messages=[
                {
                    "role": "user",
                    "content": "Describe the following json in english in detail: " + something
                }
            ]
        )

        chatResponse = response.message.content

        print("Described json: " + chatResponse)

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
    except:
        pass
