from pathlib import Path
import pandas as pd
import csv
import time
import subprocess

# Load prompt dataset
dataset_path = "statistical_prompts_dataset.csv"
df = pd.read_csv(dataset_path, delimiter=';')

# Define list of local LLM models served by Ollama
# ollama_models = ["qwen3:8b", "deepseek-r1:latest"]
ollama_models = ["qwen3:8b", "deepseek-v2:latest"]

# Prepare output CSV path
output_path = "llm_ollama_responses.csv"

# Function to query a single prompt to a local model via Ollama
def query_ollama(model_name: str, prompt: str) -> str:
    try:
        prompt = prompt +" /no_think"
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt.encode("utf-8") ,
            #input=prompt + " /no_think",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )
        return result.stdout.decode("utf-8").strip()
        #return result.stdout
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR] {str(e)}"

# Loop through all prompts and models
results = []
for _, row in df.iterrows():
    prompts = {
        "original": row["original_prompt"],
        "paraphrase": row["paraphrase"],
        "contradictory": row["contradictory_prompt"]
    }
    for model in ollama_models:
        for prompt_type, prompt_text in prompts.items():
            print(f"Querying model {model} with {prompt_type} prompt ID {row['id']}")
            response = query_ollama(model, prompt_text)
            results.append({
                "id": row["id"],
                "model": model,
                "prompt_type": prompt_type,
                "prompt": prompt_text,
                "response": response
            })
            time.sleep(1)  # avoid overloading the system

# Save results to CSV
keys = results[0].keys()
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=keys, delimiter=';')
    writer.writeheader()
    writer.writerows(results)

output_path
