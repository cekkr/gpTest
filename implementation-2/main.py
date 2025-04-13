import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Controlla se MPS è disponibile (per dispositivi Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Utilizzo accelerazione MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Utilizzo accelerazione CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("Utilizzo CPU (nessuna accelerazione hardware)")

# Carica il modello per l'embedding (leggero e performante)
embedding_model = SentenceTransformer('/Users/riccardo/Sources/Models/all-MiniLM-L6-v2')
# Sposta il modello di embedding sul device appropriato
embedding_model.to(device)

# Carica un modello per la generazione/ricostruzione del testo
tokenizer = AutoTokenizer.from_pretrained("gpt2")
generation_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)


def testo_a_vettori(testo, chunk_size=128):
    """
    Converte il testo in vettori embedded suddividendolo in chunk logici.

    Args:
        testo: Testo da convertire in vettori
        chunk_size: Lunghezza massima di ogni chunk in caratteri

    Returns:
        Lista di vettori numpy e lista dei chunk di testo corrispondenti
    """
    # Suddividi il testo in frasi o paragrafi
    chunks = []
    current_chunk = ""

    for sentence in testo.replace("\n", " ").split(". "):
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Crea embedding per ogni chunk
    embeddings = embedding_model.encode(chunks)

    return embeddings, chunks


def trova_testo_simile(vettore_query, vettori_riferimento, testi_riferimento, top_k=5):
    """
    Trova i testi più simili a un vettore query

    Args:
        vettore_query: Vettore di query
        vettori_riferimento: Array di vettori di riferimento
        testi_riferimento: Lista di testi corrispondenti ai vettori
        top_k: Numero di risultati da restituire

    Returns:
        Lista dei testi più simili
    """
    # Calcola similarità coseno
    similarities = []
    for vec in vettori_riferimento:
        sim = np.dot(vettore_query, vec) / (np.linalg.norm(vettore_query) * np.linalg.norm(vec))
        similarities.append(sim)

    # Ordina per similarità
    sorted_indices = np.argsort(similarities)[::-1][:top_k]

    return [testi_riferimento[i] for i in sorted_indices]


def ricostruisci_da_vettore(vettore, vettori_riferimento, testi_riferimento, max_length=100):
    """
    Tenta di ricostruire un testo da un vettore embedding

    Args:
        vettore: Vettore da cui ricostruire il testo
        vettori_riferimento: Array di vettori di riferimento
        testi_riferimento: Lista di testi corrispondenti ai vettori
        max_length: Lunghezza massima del testo generato

    Returns:
        Testo ricostruito basato sul contesto più simile
    """
    # Trova il testo di riferimento più simile
    testi_simili = trova_testo_simile(vettore, vettori_riferimento, testi_riferimento, top_k=1)
    contesto = testi_simili[0]

    # Usa il contesto come prompt per generare testo simile
    inputs = tokenizer(contesto, return_tensors="pt").to(device)
    with torch.no_grad():
        output_sequences = generation_model.generate(
            input_ids=inputs['input_ids'],
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    testo_generato = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return testo_generato


# Esempio di utilizzo
if __name__ == "__main__":
    # Testo di esempio
    testo_originale = """L'intelligenza artificiale sta trasformando molti settori.
    Gli algoritmi di machine learning consentono di analizzare grandi quantità di dati.
    I modelli di linguaggio possono generare testo coerente e rispondere a domande."""

    # Converti in vettori
    vettori, chunks = testo_a_vettori(testo_originale)
    print(f"Creati {len(vettori)} vettori di dimensione {vettori[0].shape[0]}")

    # Ricostruisci il testo dal primo vettore
    testo_ricostruito = ricostruisci_da_vettore(vettori[0], vettori, chunks)

    print("\nTesto originale primo chunk:")
    print(chunks[0])
    print("\nTesto ricostruito:")
    print(testo_ricostruito)