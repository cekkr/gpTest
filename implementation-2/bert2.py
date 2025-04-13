import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import os
from torch import nn

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

# Modello per embedding
model_name = "/Users/riccardo/Sources/Models/distilbert-base-uncased"
tokenizer_embedding = AutoTokenizer.from_pretrained(model_name)
model_embedding = AutoModel.from_pretrained(model_name).to(device)

# Modello per generazione
gen_model_name = "/Users/riccardo/Sources/Models/distilgpt2"
tokenizer_gen = AutoTokenizer.from_pretrained(gen_model_name)
model_gen = AutoModelForCausalLM.from_pretrained(gen_model_name).to(device)

# Dimensioni del vettore di embedding e dell'input del decodificatore
embedding_dim = model_embedding.config.hidden_size  # Dimensione dell'embedding (768 per DistilBERT)
decoder_dim = model_gen.config.n_embd  # Dimensione dell'input del generatore (768 per DistilGPT2)


# Classe decoder per convertire vettori in input adatti al modello di linguaggio
class VettoreATestoDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Rete neurale semplice per proiettare l'embedding nel formato corretto per il decoder
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.projection(x)


# Inizializza il decoder
decoder = VettoreATestoDecoder(embedding_dim, decoder_dim).to(device)


# Funzione per ottimizzare il decoder
def addestra_decoder(esempi_testo, learning_rate=1e-4, epochs=100):
    """
    Addestra il decoder sui testi di esempio

    Args:
        esempi_testo: Lista di testi per l'addestramento
        learning_rate: Tasso di apprendimento
        epochs: Numero di epoche di addestramento
    """
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Estrai gli embedding e i token di input per il modello di generazione
    with torch.no_grad():
        # Ottieni gli embedding dei testi
        inputs_emb = tokenizer_embedding(esempi_testo, padding=True, truncation=True,
                                         max_length=512, return_tensors="pt").to(device)
        outputs_emb = model_embedding(**inputs_emb)
        embeddings = outputs_emb.last_hidden_state[:, 0, :]  # Usa il token [CLS]

        # Ottieni gli initial hidden states del modello di generazione
        inputs_gen = tokenizer_gen(esempi_testo, padding=True, truncation=True,
                                   return_tensors="pt").to(device)
        # Per GPT2, vogliamo l'hidden state iniziale che userà per generare
        # Possiamo ottenerlo eseguendo un forward pass e guardando i hidden states
        outputs_gen = model_gen(inputs_gen.input_ids, output_hidden_states=True)
        targets = outputs_gen.hidden_states[0][:, 0, :]  # Primo hidden state

    print(f"Addestramento decoder: {embeddings.shape} -> {targets.shape}")

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward
        output = decoder(embeddings)
        loss = criterion(output, targets)

        # Backward
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoca {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

    # Salva il modello
    torch.save(decoder.state_dict(), "vector_to_text_decoder.pt")
    print("Decoder addestrato e salvato")


# Funzione per creare embedding utilizzando il token [CLS]
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

            # Usa il token [CLS] come rappresentazione dell'intero testo
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # Normalizza gli embedding
            norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
            cls_embeddings = cls_embeddings / norms

            embeddings.extend(cls_embeddings)

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


def ricostruisci_da_vettore(vettore, max_length=100, temperatura=0.7):
    """
    Ricostruisce un testo direttamente da un vettore embedding senza usare testi di riferimento

    Args:
        vettore: Vettore da cui ricostruire il testo (numpy array)
        max_length: Lunghezza massima del testo generato
        temperatura: Temperatura per la generazione (più alta = più casuale)

    Returns:
        Testo ricostruito
    """
    # Converti il vettore in un tensor
    vettore_tensor = torch.tensor(vettore).float().to(device)

    # Converti il vettore in un formato adatto al modello di linguaggio
    with torch.no_grad():
        # Usa il decoder per trasformare l'embedding in un hidden state per il generatore
        hidden_state = decoder(vettore_tensor.unsqueeze(0))

    try:
        # Prepara un token iniziale come input
        input_ids = torch.tensor([[tokenizer_gen.bos_token_id]]).to(device)

        # Configura gli hidden states iniziali del modello
        past = (hidden_state.unsqueeze(0).repeat(model_gen.config.n_layer, 1, 1, 1),)

        # Genera il testo
        with torch.no_grad():
            output_sequence = model_gen.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperatura,
                top_p=0.9,
                do_sample=True,
                past_key_values=past,
                num_return_sequences=1
            )

        # Decodifica il testo generato
        testo_generato = tokenizer_gen.decode(output_sequence[0], skip_special_tokens=True)
        return testo_generato
    except Exception as e:
        print(f"Errore durante la generazione: {e}")
        # Fallback: genera senza condizionamento iniziale
        return genera_testo_semplice(max_length, temperatura)


def genera_testo_semplice(max_length=100, temperatura=0.7):
    """
    Genera testo senza condizionamento (metodo di fallback)

    Args:
        max_length: Lunghezza massima del testo generato
        temperatura: Temperatura per la generazione

    Returns:
        Testo generato
    """
    try:
        with torch.no_grad():
            # Genera il testo con un prompt vuoto
            input_ids = tokenizer_gen(tokenizer_gen.bos_token, return_tensors="pt").input_ids.to(device)
            output_sequence = model_gen.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperatura,
                top_p=0.9,
                do_sample=True
            )

        # Decodifica il testo generato
        testo_generato = tokenizer_gen.decode(output_sequence[0], skip_special_tokens=True)
        return testo_generato
    except Exception as e:
        print(f"Errore anche nella generazione semplice: {e}")
        return "Errore nella generazione del testo."


# Funzione principale
def main():
    """Funzione principale per testare il codice"""
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

    # Addestra il decoder
    print("\nAddestramento del decoder...")
    addestra_decoder(paragrafi)

    # Ricostruisci il testo dal primo vettore
    print("\nRicostruzione del primo paragrafo:")
    testo_ricostruito = ricostruisci_da_vettore(vettori[0])

    print(f"\nOriginale: {paragrafi[0]}")
    print(f"\nRicostruito: {testo_ricostruito}")

    # Test con un nuovo testo
    nuovo_testo = "Le reti neurali profonde hanno rivoluzionato il campo della visione artificiale."
    print(f"\nTest con nuovo testo: {nuovo_testo}")

    # Crea embedding per il nuovo testo
    nuovo_vettore = crea_embedding([nuovo_testo])[0]

    # Ricostruisci dal nuovo vettore
    testo_ricostruito = ricostruisci_da_vettore(nuovo_vettore)
    print(f"Ricostruito: {testo_ricostruito}")


if __name__ == "__main__":
    main()