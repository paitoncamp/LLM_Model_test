from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load model semantic embedding
model_embed = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
questions_df = pd.read_csv('dataset/statistical_prompts_dataset.csv', sep=';')
responses_df = pd.read_csv('dataset/llm_ollama_responses_50.csv', sep=';')

def semantic_similarity(a, b):
    embeddings = model_embed.encode([str(a), str(b)], convert_to_tensor=True)
    return float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))

SIMILAR_THRESHOLD = 0.75
DIFF_THRESHOLD = 0.4

results = []

for i, row in questions_df.iterrows():
    original = row['original_prompt']
    paraphrase = row['paraphrase']
    contradictory = row['contradictory_prompt']

    for model in responses_df['model_name'].unique():
        res_ori = responses_df.query("model_name == @model and prompt == @original")['response']
        res_para = responses_df.query("model_name == @model and prompt == @paraphrase")['response']
        res_contra = responses_df.query("model_name == @model and prompt == @contradictory")['response']

        if res_ori.empty or res_para.empty or res_contra.empty:
            continue

        sim_ori_para = semantic_similarity(res_ori.values[0], res_para.values[0])
        sim_ori_contra = semantic_similarity(res_ori.values[0], res_contra.values[0])

        results.append({
            'id': row['id'],
            'model_name': model,
            'original_prompt': original,
            'paraphrase_prompt': paraphrase,
            'contradictory_prompt': contradictory,
            'response_original': res_ori.values[0],
            'response_paraphrase': res_para.values[0],
            'response_contradictory': res_contra.values[0],
            'semantic_similarity_ori_para': sim_ori_para,
            'semantic_similarity_ori_contra': sim_ori_contra,
            'is_consistent_with_paraphrase': sim_ori_para >= SIMILAR_THRESHOLD,
            'is_inconsistent_with_contradictory': sim_ori_contra <= DIFF_THRESHOLD
        })

# Simpan anotasi
annotated_df = pd.DataFrame(results)
annotated_df.to_csv('anotasi_konsistensi_semantik.csv', index=False, sep=';')

print("✅ Anotasi semantik selesai dan disimpan di 'anotasi_konsistensi_semantik.csv'")
