import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import os
from sklearn.metrics.pairwise import cosine_similarity

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

# Usa un modello più leggero ma efficace per l'embedding: distilbert-base-uncased
model_name = "/Users/riccardo/Sources/Models/distilbert-base-uncased"
tokenizer_embedding = AutoTokenizer.from_pretrained(model_name)
model_embedding = AutoModel.from_pretrained(model_name).to(device)

# Per la ricostruzione, usiamo un modello più piccolo di GPT-2: distilgpt2
gen_model_name = "/Users/riccardo/Sources/Models/distilgpt2"
tokenizer_gen = AutoTokenizer.from_pretrained(gen_model_name)
model_gen = AutoModelForCausalLM.from_pretrained(gen_model_name).to(device)


# Funzione per creare embedding utilizzando media degli ultimi hidden states
def crea_embedding(testi, batch_size=8):
    """
    Crea embedding per una lista di testi utilizzando un modello transformer

    Args:
        testi: Lista di testi da convertire in embedding
        batch_size: Numero di testi da processare contemporaneamente

    Returns:
        numpy array di embedding
    """
    embeddings = []

    for i in range(0, len(testi), batch_size):
        batch = testi[i:i + batch_size]

        # Tokenizza i testi
        inputs = tokenizer_embedding(batch, padding=True, truncation=True,
                                     max_length=512, return_tensors="pt").to(device)

        # Non calcolare gradienti per risparmiare memoria
        with torch.no_grad():
            # Ottieni gli output del modello
            outputs = model_embedding(**inputs)

            # Usa l'ultimo hidden state
            last_hidden_states = outputs.last_hidden_state

            # Crea una maschera di attenzione per escludere i padding tokens
            attention_mask = inputs['attention_mask'].unsqueeze(-1)

            # Calcola la media degli hidden states, escludendo i padding tokens
            embeddings_batch = torch.sum(last_hidden_states * attention_mask, 1) / torch.sum(attention_mask, 1)

            # Converte a numpy e normalizza
            embeddings_batch = embeddings_batch.cpu().numpy()
            # Normalizza gli embedding (per calcolare più facilmente la similarità coseno)
            norms = np.linalg.norm(embeddings_batch, axis=1, keepdims=True)
            embeddings_batch = embeddings_batch / norms

            embeddings.extend(embeddings_batch)

    return np.array(embeddings)


def dividi_in_paragrafi(testo, max_chars=200):
    """
    Divide un testo in paragrafi di dimensioni ragionevoli

    Args:
        testo: Testo da dividere
        max_chars: Numero massimo di caratteri per paragrafo

    Returns:
        Lista di paragrafi
    """
    paragrafi = []
    # Sostituisci le interruzioni di riga con spazi
    testo = testo.replace("\n", " ")

    # Dividi per punti
    frasi = testo.split(". ")

    paragrafo_corrente = ""
    for frase in frasi:
        # Se il paragrafo corrente è già troppo lungo, aggiungi il punto e salvalo
        if len(paragrafo_corrente) + len(frase) > max_chars and paragrafo_corrente:
            paragrafi.append(paragrafo_corrente.strip() + ".")
            paragrafo_corrente = frase
        else:
            # Altrimenti, aggiungi la frase al paragrafo corrente
            if paragrafo_corrente:
                paragrafo_corrente += ". " + frase
            else:
                paragrafo_corrente = frase

    # Aggiungi l'ultimo paragrafo, se necessario
    if paragrafo_corrente:
        paragrafi.append(paragrafo_corrente.strip() + ".")

    return paragrafi


def testo_a_vettori(testo, max_chars=200):
    """
    Converte un testo in vettori embedding

    Args:
        testo: Testo da convertire
        max_chars: Numero massimo di caratteri per ogni chunk

    Returns:
        Tuple di (array di vettori, lista di chunk di testo)
    """
    # Dividi il testo in paragrafi
    paragrafi = dividi_in_paragrafi(testo, max_chars)

    # Crea embedding per i paragrafi
    embeddings = crea_embedding(paragrafi)

    return embeddings, paragrafi


def trova_testo_simile(vettore_query, vettori_riferimento, testi_riferimento, top_k=3):
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
    # Calcola la similarità coseno tra il vettore query e tutti i vettori di riferimento
    similarita = cosine_similarity([vettore_query], vettori_riferimento)[0]

    # Trova gli indici dei top_k vettori più simili
    indici_top = np.argsort(similarita)[::-1][:top_k]

    # Restituisci i testi corrispondenti
    return [testi_riferimento[i] for i in indici_top]


def ricostruisci_da_vettore(vettore, vettori_riferimento, testi_riferimento, max_length=100):
    """
    Tenta di ricostruire un testo da un vettore embedding

    Args:
        vettore: Vettore da cui ricostruire il testo
        vettori_riferimento: Array di vettori di riferimento
        testi_riferimento: Lista di testi corrispondenti ai vettori
        max_length: Lunghezza massima del testo generato

    Returns:
        Testo ricostruito
    """
    # Trova il testo più simile al vettore
    testi_simili = trova_testo_simile(vettore, vettori_riferimento, testi_riferimento, top_k=1)
    contesto = testi_simili[0]

    # Usa il contesto come prompt per generare un testo simile
    inputs = tokenizer_gen(contesto, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            # Genera il testo
            output_sequences = model_gen.generate(
                input_ids=inputs['input_ids'],
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        # Decodifica il testo generato
        testo_generato = tokenizer_gen.decode(output_sequences[0], skip_special_tokens=True)
        return testo_generato
    except Exception as e:
        print(f"Errore durante la generazione: {e}")
        return contesto  # In caso di errore, restituisci il contesto originale


# Esempio di utilizzo
if __name__ == "__main__":
    # Testo di esempio
    testo_originale = """L'intelligenza artificiale sta rivoluzionando molti campi della tecnologia moderna.
    Gli algoritmi di machine learning permettono di analizzare enormi quantità di dati e trovare pattern nascosti.
    I modelli linguistici sono in grado di comprendere e generare testo in modo sempre più sofisticato.
    Le applicazioni pratiche spaziano dalla medicina alla finanza, dal marketing all'automotive.
    Ciao come stai? Io mi chiamo Riccardo."""

    print("Elaborazione del testo in corso...")

    # Converti il testo in vettori
    vettori, paragrafi = testo_a_vettori(testo_originale)

    print(f"Creati {len(vettori)} vettori di dimensione {vettori[0].shape[0]}")

    # Stampa i paragrafi
    print("\nParagrafi estratti:")
    for i, p in enumerate(paragrafi):
        print(f"{i + 1}. {p}")

    # Ricostruisci il testo dal primo vettore
    print("\nRicostruzione del primo paragrafo:")
    testo_ricostruito = ricostruisci_da_vettore(vettori[0], vettori, paragrafi)

    print("\nVettori effettivi: ")
    print(vettori)
    print(paragrafi)

    print(f"\nOriginale: {paragrafi[0]}")
    print(f"\nRicostruito: {testo_ricostruito}")

    # Test di similarità
    print("\nTest di similarità semantica:")
    # Crea un testo di test con concetti simili ma parole diverse
    testo_test = "L'AI sta cambiando il modo in cui funziona la tecnologia in molti settori."
    vettore_test, _ = testo_a_vettori(testo_test)

    # Trova i paragrafi più simili
    paragrafi_simili = trova_testo_simile(vettore_test[0], vettori, paragrafi, top_k=2)

    print(f"\nTesto di query: {testo_test}")
    print("\nParagrafi più simili:")
    for i, p in enumerate(paragrafi_simili):
        print(f"{i + 1}. {p}")