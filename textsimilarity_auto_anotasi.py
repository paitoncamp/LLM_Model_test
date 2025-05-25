import pandas as pd
from difflib import SequenceMatcher

# Load daftar pertanyaan
questions_df = pd.read_csv('dataset/statistical_prompts_dataset.csv', sep=';')

# Load hasil respons model
responses_df = pd.read_csv('dataset/llm_ollama_responses_50.csv', sep=';')

# Fungsi kemiripan antar respons (berbasis kemiripan karakter)
def response_similarity(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

# Threshold untuk deteksi konsistensi
SIMILAR_THRESHOLD = 0.75  # untuk original vs paraphrase
DIFF_THRESHOLD = 0.4      # untuk original vs contradictory

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

        sim_ori_para = response_similarity(res_ori.values[0], res_para.values[0])
        sim_ori_contra = response_similarity(res_ori.values[0], res_contra.values[0])
        
        results.append({
            'id': row['id'],
            'model_name': model,
            'original_prompt': original,
            'paraphrase_prompt': paraphrase,
            'contradictory_prompt': contradictory,
            'response_original': res_ori.values[0],
            'response_paraphrase': res_para.values[0],
            'response_contradictory': res_contra.values[0],
            'similarity_ori_para': sim_ori_para,
            'similarity_ori_contra': sim_ori_contra,
            'is_consistent_with_paraphrase': sim_ori_para >= SIMILAR_THRESHOLD,
            'is_inconsistent_with_contradictory': sim_ori_contra <= DIFF_THRESHOLD
        })

# Simpan hasil anotasi
annotated_df = pd.DataFrame(results)
annotated_df.to_csv('anotasi_konsistensi_model.csv', index=False, sep=';')

print("✅ Anotasi otomatis selesai dan disimpan di 'anotasi_konsistensi_model.csv'")