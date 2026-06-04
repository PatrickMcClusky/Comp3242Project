import json
from typing import Any, Dict, Optional, Tuple

import ollama
from ollama import chat


# --------------------------------------------------
# Ollama Client
# --------------------------------------------------

client = ollama.Client()


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def extract_text(response: Dict[str, Any]) -> str:
    return response["response"].strip()


def try_parse_json(text: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, e

def find_json(text: str) -> str:
    for marker in ("```json", "```"):
        start = text.find(marker)
        if start != -1:
            break
    else:
        return text

    start = text.find("\n", start)
    if start == -1:
        return text

    start += 1

    end = text.find("```", start)
    if end == -1:
        return text

    return text[start:end].strip()
    
# --------------------------------------------------
# Dataset Saving
# --------------------------------------------------

def save_example(
    broken_json: str,
    fixed_json: str,
    path: str = "json_repair_dataset.jsonl"
):
    """
    Save successful repair examples for later training.
    """

    record = {
        "broken_json": broken_json,
        "fixed_json": fixed_json
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------
# Stage 1 - Generate JSON
# --------------------------------------------------

def generate_json(task_prompt: str) -> str:
    """
    Generate JSON using DeepSeek-R1 1.5B.
    """

    response = chat(
        model="deepseek-r1:1.5b",
        messages=[
            {
                "role": "user",
                "content": f"""
You are a JSON generator.

Return ONLY JSON.

Do NOT write any comments.

Return ONLY JSON.

Task:
{task_prompt}
"""
            }
        ]
    )

    return response.message.content


# --------------------------------------------------
# Stage 2 - Repair JSON
# --------------------------------------------------

def repair_json(
    broken_json: str,
    error_message: str,
    original_task: str
) -> str:
    """
    Ask DeepSeek-R1 8B to repair invalid JSON.
    """

    response = chat(
        model="deepseek-r1:1.5b",
        messages=[
            {
                "role": "user",
                "content": f"""
You are a JSON repair system.

Original task:
{original_task}

Parser error:
{error_message}

Current invalid JSON:
{broken_json}

Requirements:
- Make the MINIMUM changes necessary.
- Preserve all keys.
- Preserve all values.
- Preserve all information.
- Do not invent content.
- Fix syntax only.
- Return ONLY valid JSON.
"""
            }
        ]
    )

    return response.message.content


# --------------------------------------------------
# Iterative Repair Loop
# --------------------------------------------------

def repair_until_valid(
    broken_json: str,
    original_task: str,
    max_attempts: int = 10
):
    """
    Attempts repeated repairs using the 8B model.

    Each failure is fed back into the model with
    the latest parser error.

    Returns:
        (fixed_json, parsed_json)
        or
        (None, None)
    """

    current_json = broken_json

    for attempt in range(1, max_attempts + 1):

        current_json = find_json(current_json)


        parsed, error = try_parse_json(current_json)

        print(
            f"Repair attempt {attempt}/{max_attempts} "
            f"error: {error}"
        )
        print("current broken json: " + current_json)
        # Success
        if parsed is not None:
            print(f"Repair succeeded on attempt {attempt - 1}")
            return current_json, parsed

        current_json = repair_json(
            broken_json=current_json,
            error_message=error,
            original_task=original_task
        )

    current_json = find_json(current_json)


    # One final parse check after the last repair
    parsed, error = try_parse_json(current_json)

    if parsed is not None:
        print(f"Repair succeeded on final attempt")
        return current_json, parsed

    print("Repair failed after all attempts")

    return None, None


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------

def generate_valid_json(
    task_prompt: str,
    dataset_path: str = "json_repair_dataset.jsonl",
    max_repair_attempts: int = 10
):
    """
    Pipeline:

    1. Generate JSON with 1.5B model.
    2. Parse.
    3. If invalid:
         repair with 8B.
    4. Retry repair up to N times.
    5. Save successful repair pairs.

    Returns:
    {
        success: bool,
        repaired: bool,
        parsed_json: dict/list/None,
        generated_json: str,
        fixed_json: str|None,
        error: str|None
    }
    """

    # --------------------------------------
    # Generate with 1.5B
    # --------------------------------------

    generated_json = generate_json(task_prompt)

    generated_json = find_json(generated_json)

    parsed_json, parse_error = try_parse_json(generated_json)

    # --------------------------------------
    # Generation already valid
    # --------------------------------------

    if parsed_json is not None:

        return {
            "success": True,
            "repaired": False,
            "parsed_json": parsed_json,
            "generated_json": generated_json,
            "fixed_json": None,
            "error": None
        }

    print("Initial JSON failed parsing.")
    print(parse_error)

    # --------------------------------------
    # Repair Loop
    # --------------------------------------

    fixed_json, repaired_parsed = repair_until_valid(
        broken_json=generated_json,
        original_task=task_prompt,
        max_attempts=max_repair_attempts
    )

    # --------------------------------------
    # Repair Success
    # --------------------------------------

    if repaired_parsed is not None:

        save_example(
            broken_json=generated_json,
            fixed_json=fixed_json,
            path=dataset_path
        )

        return {
            "success": True,
            "repaired": True,
            "parsed_json": repaired_parsed,
            "generated_json": generated_json,  # original broken JSON
            "fixed_json": fixed_json,          # final repaired JSON
            "error": None
        }

    # --------------------------------------
    # Total Failure
    # --------------------------------------

    return {
        "success": False,
        "repaired": False,
        "parsed_json": None,
        "generated_json": generated_json,
        "fixed_json": None,
        "error": f"Repair failed after {max_repair_attempts} attempts"
    }


# --------------------------------------------------
# Example
# --------------------------------------------------

def create_valid_json(task: str):

    result = generate_valid_json(
        task_prompt=task,
        max_repair_attempts=10
    )

    print("\n========================")
    print("RESULT")
    print("========================")

    print("Success:", result["success"])
    print("Repaired:", result["repaired"])

    if result["generated_json"] is not None:
        print("\nOriginal Generated JSON:")
        print(result["generated_json"])

    if result["fixed_json"] is not None:
        print("\nFinal Repaired JSON:")
        print(result["fixed_json"])

    if result["parsed_json"] is not None:
        print("\nParsed Object:")
        print(json.dumps(result["parsed_json"], indent=2))

    if result["error"] is not None:
        print("\nError:")
        print(result["error"])

if __name__ == "__main__":
    task = """
Generate a JSON object representing a user profile.
"""

    create_valid_json(task)