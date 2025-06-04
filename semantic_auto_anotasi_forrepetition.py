from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load model semantic embedding
model_embed = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
questions_df = pd.read_csv('dataset/statistical_prompts_dataset.csv', sep=';')
responses_df = pd.read_csv('llm_ollama_repetition_responses.csv', sep=';')

def semantic_similarity(a, b):
    embeddings = model_embed.encode([str(a), str(b)], convert_to_tensor=True)
    return float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))

SIMILAR_THRESHOLD = 0.75
DIFF_THRESHOLD = 0.75

results = []

for i, row in questions_df.iterrows():
    original = row['original_prompt']
    repetition_1 = row['original_prompt']
    repetition_2 = row['original_prompt']
    

    for model in responses_df['model'].unique():
        res_ori = responses_df.query("model == @model and prompt_type == 'original' and prompt == @original ")['response']
        res_rep1 = responses_df.query("model == @model and prompt_type == 'repetition_1' and prompt == @repetition_1 ")['response']
        res_rep2 = responses_df.query("model == @model and prompt_type == 'repetition_2' and prompt == @repetition_2 ")['response']

        if res_ori.empty or res_rep1.empty or res_rep2.empty:
            continue

        sim_ori_rep1 = semantic_similarity(res_ori.values[0], res_rep1.values[0])
        sim_ori_rep2 = semantic_similarity(res_ori.values[0], res_rep2.values[0])
        sim_ori_rep3 = semantic_similarity(res_rep1.values[0], res_rep2.values[0])

        results.append({
            'id': row['id'],
            'model_name': model,
            'original_prompt': original,
            'repetition1_prompt': repetition_1,
            'repetition2_prompt': repetition_2,
            'response_original': res_ori.values[0],
            'response_repetition_1': res_rep1.values[0],
            'response_repetition_2': res_rep2.values[0],
            'semantic_similarity_ori_rep1': sim_ori_rep1,
            'semantic_similarity_ori_rep2': sim_ori_rep2,
            'semantic_similarity_ori_rep3': sim_ori_rep3,
            'is_consistent_with_repetition_1': sim_ori_rep1 >= SIMILAR_THRESHOLD,
            'is_consistent_with_repetition_2': sim_ori_rep2 >= SIMILAR_THRESHOLD,
            'is_consistent_with_repetition_3': sim_ori_rep3 >= SIMILAR_THRESHOLD
        })
    if i==7:
        break

# Simpan anotasi
annotated_df = pd.DataFrame(results)
annotated_df.to_csv('anotasi_konsistensi_semantik_repetisi.csv', index=False, sep=';')

print("✅ Anotasi semantik untuk repetisi selesai dan disimpan di 'anotasi_konsistensi_semantik_repetisi.csv'")
