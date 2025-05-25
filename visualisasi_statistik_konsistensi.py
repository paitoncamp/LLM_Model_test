import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data hasil anotasi semantik
df = pd.read_csv('anotasi_konsistensi_semantik.csv', sep=';')

# Buat metrik konsistensi total per model
summary = df.groupby('model_name').agg(
    total_pertanyaan=('id', 'count'),
    konsisten_paraphrase=('is_consistent_with_paraphrase', 'sum'),
    konsisten_kontradiksi=('is_inconsistent_with_contradictory', 'sum')
).reset_index()

summary['persen_konsisten_paraphrase'] = 100 * summary['konsisten_paraphrase'] / summary['total_pertanyaan']
summary['persen_konsisten_kontradiktif'] = 100 * summary['konsisten_kontradiksi'] / summary['total_pertanyaan']

# Visualisasi 1: Konsistensi terhadap Paraphrase
plt.figure(figsize=(10, 5))
sns.barplot(data=summary, x='model_name', y='persen_konsisten_paraphrase', color='skyblue')
plt.title('Konsistensi Jawaban terhadap Pertanyaan Paraphrase')
plt.ylabel('Persentase Konsisten (%)')
plt.xlabel('Model LLM')
plt.ylim(0, 100)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Visualisasi 2: InKonsistensi terhadap Kontradiksi (semakin tinggi, semakin baik)
plt.figure(figsize=(10, 5))
sns.barplot(data=summary, x='model_name', y='persen_konsisten_kontradiktif', color='salmon')
plt.title('Ketidaksesuaian Jawaban terhadap Pertanyaan Kontradiktif (Semakin Tinggi = Semakin Baik)')
plt.ylabel('Persentase Tidak Konsisten terhadap Kontradiksi (%)')
plt.xlabel('Model LLM')
plt.ylim(0, 100)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# (Optional) Tampilkan tabel ringkasan
display(summary)
