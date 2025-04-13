import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import os
from torch import nn
import time
import json
from pathlib import Path

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

# Directory per salvare i checkpoint
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Modello per embedding
model_name = "/Users/riccardo/Sources/Models/distilbert-base-uncased"
tokenizer_embedding = AutoTokenizer.from_pretrained(model_name)
model_embedding = AutoModel.from_pretrained(model_name).to(device)

# Modello per generazione
gen_model_name = "/Users/riccardo/Sources/Models/distilgpt2"
tokenizer_gen = AutoTokenizer.from_pretrained(gen_model_name)
# Aggiungi un token di padding a GPT-2 (necessario per il batch processing)
tokenizer_gen.pad_token = tokenizer_gen.eos_token
model_gen = AutoModelForCausalLM.from_pretrained(gen_model_name).to(device)
# Assicurati che il modello conosca il token di padding
model_gen.config.pad_token_id = tokenizer_gen.pad_token_id

# Dimensioni del vettore di embedding e dell'input del decodificatore
embedding_dim = model_embedding.config.hidden_size  # Dimensione dell'embedding (768 per DistilBERT)
decoder_dim = model_gen.config.n_embd  # Dimensione dell'input del generatore (768 per DistilGPT2)


# Classe decoder per convertire vettori in input adatti al modello di linguaggio
class VettoreATestoDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Rete neurale per proiettare l'embedding nel formato corretto per il decoder
        # Aggiunti più strati per una migliore rappresentazione
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.projection(x)


# Inizializza il decoder
decoder = VettoreATestoDecoder(embedding_dim, decoder_dim).to(device)


# Funzioni per gestire i checkpoint
def salva_checkpoint(model, optimizer, epoch, loss, nome_file=None):
    """
    Salva il checkpoint del modello

    Args:
        model: Il modello da salvare
        optimizer: L'ottimizzatore da salvare
        epoch: L'epoca corrente
        loss: La loss corrente
        nome_file: Nome del file di checkpoint (se None, usa timestamp)
    """
    if nome_file is None:
        nome_file = f"checkpoint_{int(time.time())}.pt"

    checkpoint_path = CHECKPOINT_DIR / nome_file

    # Crea il dizionario del checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'timestamp': time.time()
    }

    # Salva il checkpoint
    torch.save(checkpoint, checkpoint_path)

    # Aggiorna il file metadata.json con l'ultimo checkpoint
    metadata = {
        'ultimo_checkpoint': str(checkpoint_path),
        'epoca': epoch,
        'loss': loss,
        'timestamp': time.time()
    }

    with open(CHECKPOINT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Checkpoint salvato: {checkpoint_path}")
    return str(checkpoint_path)


def carica_ultimo_checkpoint(model, optimizer=None):
    """
    Carica l'ultimo checkpoint disponibile

    Args:
        model: Il modello in cui caricare i pesi
        optimizer: L'ottimizzatore in cui caricare lo stato (opzionale)

    Returns:
        Dizionario con informazioni sul checkpoint, o None se non trovato
    """
    metadata_path = CHECKPOINT_DIR / "metadata.json"

    if not metadata_path.exists():
        print("Nessun checkpoint trovato")
        return None

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        ultimo_checkpoint = metadata.get('ultimo_checkpoint')

        if not ultimo_checkpoint or not Path(ultimo_checkpoint).exists():
            print("Il file di checkpoint non esiste più")
            return None

        # Carica il checkpoint
        checkpoint = torch.load(ultimo_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint caricato: {ultimo_checkpoint} (Epoca {checkpoint['epoch']})")
        return checkpoint

    except Exception as e:
        print(f"Errore durante il caricamento del checkpoint: {e}")
        return None


def trova_miglior_checkpoint(criterio='loss'):
    """
    Trova il miglior checkpoint in base al criterio specificato

    Args:
        criterio: 'loss' per il checkpoint con la loss più bassa, 'epoca' per il più recente

    Returns:
        Path del miglior checkpoint, o None se non trovato
    """
    checkpoint_files = list(CHECKPOINT_DIR.glob("checkpoint_*.pt"))

    if not checkpoint_files:
        return None

    best_checkpoint = None
    best_value = float('inf') if criterio == 'loss' else -1

    for cp_file in checkpoint_files:
        try:
            checkpoint = torch.load(cp_file, map_location='cpu')

            if criterio == 'loss' and checkpoint['loss'] < best_value:
                best_value = checkpoint['loss']
                best_checkpoint = cp_file
            elif criterio == 'epoca' and checkpoint['epoch'] > best_value:
                best_value = checkpoint['epoch']
                best_checkpoint = cp_file

        except Exception:
            continue

    return best_checkpoint


# Funzione per ottimizzare il decoder
def addestra_decoder(esempi_testo, learning_rate=1e-4, epochs=100, batch_size=4, checkpoint_interval=10):
    """
    Addestra il decoder sui testi di esempio con supporto per checkpoint

    Args:
        esempi_testo: Lista di testi per l'addestramento
        learning_rate: Tasso di apprendimento
        epochs: Numero di epoche di addestramento
        batch_size: Dimensione del batch per l'addestramento
        checkpoint_interval: Intervallo per salvare i checkpoint
    """
    # Inizializza optimizer
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Carica l'ultimo checkpoint se disponibile
    start_epoch = 0
    best_loss = float('inf')
    checkpoint = carica_ultimo_checkpoint(decoder, optimizer)

    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Ripresa addestramento dall'epoca {start_epoch}")

    # Per facilitare il training, creiamo i dataset di embedding e target una volta sola
    embedding_dataset = []
    target_dataset = []

    print("Preparazione dataset...")
    for i in range(0, len(esempi_testo), batch_size):
        batch_testi = esempi_testo[i:i + batch_size]

        with torch.no_grad():
            # Ottieni gli embedding dei testi
            inputs_emb = tokenizer_embedding(batch_testi, padding=True, truncation=True,
                                             max_length=512, return_tensors="pt").to(device)
            outputs_emb = model_embedding(**inputs_emb)
            batch_embeddings = outputs_emb.last_hidden_state[:, 0, :]  # Usa il token [CLS]

            # Ottieni gli initial hidden states del modello di generazione
            inputs_gen = tokenizer_gen(batch_testi, padding=True, truncation=True,
                                       return_tensors="pt").to(device)
            # Per GPT2, vogliamo l'hidden state iniziale che userà per generare
            outputs_gen = model_gen(inputs_gen.input_ids, output_hidden_states=True)
            batch_targets = outputs_gen.hidden_states[0][:, 0, :]  # Primo hidden state

            embedding_dataset.append(batch_embeddings)
            target_dataset.append(batch_targets)

    total_batches = len(embedding_dataset)
    print(f"Dataset preparato: {total_batches} batch")

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0

        for batch_idx in range(total_batches):
            optimizer.zero_grad()

            # Forward
            embeddings = embedding_dataset[batch_idx]
            targets = target_dataset[batch_idx]
            output = decoder(embeddings)
            loss = criterion(output, targets)

            # Backward
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / total_batches
        print(f"Epoca {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.6f}")

        # Salva checkpoint se è il migliore finora o a intervalli regolari
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            salva_checkpoint(decoder, optimizer, epoch, avg_epoch_loss, "best_checkpoint.pt")

        if (epoch + 1) % checkpoint_interval == 0:
            salva_checkpoint(decoder, optimizer, epoch, avg_epoch_loss)

    # Salva il modello finale
    salva_checkpoint(decoder, optimizer, epochs - 1, avg_epoch_loss, "final_decoder.pt")
    print("Addestramento completato")


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
    testo = testo.replace("\n", " ").strip()

    # Se il testo è già più corto di max_chars, restituisci semplicemente
    if len(testo) <= max_chars:
        return [testo]

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

    # Filtra paragrafi troppo corti (meno di 20 caratteri)
    paragrafi = [p for p in paragrafi if len(p) > 20]

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


# Funzioni per operazioni vettoriali
def calcola_similarita_coseno(vettore1, vettore2):
    """
    Calcola la similarità coseno tra due vettori

    Args:
        vettore1: Primo vettore
        vettore2: Secondo vettore

    Returns:
        Valore di similarità coseno (da -1 a 1)
    """
    # Normalizza i vettori
    vettore1_norm = vettore1 / np.linalg.norm(vettore1)
    vettore2_norm = vettore2 / np.linalg.norm(vettore2)

    # Calcola similarità coseno
    return np.dot(vettore1_norm, vettore2_norm)


def interpola_vettori(vettore1, vettore2, peso=0.5):
    """
    Interpola tra due vettori con il peso specificato

    Args:
        vettore1: Primo vettore
        vettore2: Secondo vettore
        peso: Peso per l'interpolazione (0 = solo vettore1, 1 = solo vettore2)

    Returns:
        Vettore interpolato
    """
    vettore_interpolato = (1 - peso) * vettore1 + peso * vettore2
    # Normalizza il risultato
    return vettore_interpolato / np.linalg.norm(vettore_interpolato)


def rumore_vettore(vettore, scala=0.1):
    """
    Aggiunge rumore gaussiano a un vettore

    Args:
        vettore: Vettore di input
        scala: Scala del rumore

    Returns:
        Vettore con rumore aggiunto
    """
    rumore = np.random.normal(0, scala, vettore.shape)
    vettore_disturbato = vettore + rumore
    # Normalizza il risultato
    return vettore_disturbato / np.linalg.norm(vettore_disturbato)


def ricostruisci_da_vettore(vettore, max_length=100, temperatura=0.7, num_campioni=1, prompt=None):
    """
    Ricostruisce un testo direttamente da un vettore embedding senza usare testi di riferimento

    Args:
        vettore: Vettore da cui ricostruire il testo (numpy array)
        max_length: Lunghezza massima del testo generato
        temperatura: Temperatura per la generazione (più alta = più casuale)
        num_campioni: Numero di varianti da generare
        prompt: Prompt opzionale per indirizzare la generazione

    Returns:
        Testo ricostruito o lista di testi se num_campioni > 1
    """
    # Converti il vettore in un tensor
    vettore_tensor = torch.tensor(vettore).float().to(device)

    # Converti il vettore in un formato adatto al modello di linguaggio
    with torch.no_grad():
        # Usa il decoder per trasformare l'embedding in un hidden state per il generatore
        hidden_state = decoder(vettore_tensor.unsqueeze(0))

    try:
        # Prepara l'input iniziale
        if prompt:
            input_tokens = tokenizer_gen(prompt, return_tensors="pt").to(device)
            input_ids = input_tokens.input_ids
        else:
            input_ids = torch.tensor([[tokenizer_gen.bos_token_id]]).to(device)

        # Configura gli hidden states iniziali del modello
        # Prepara il formato corretto per past_key_values in GPT-2
        # past_key_values è una tupla di tuple: (key_layers, value_layers) per ogni layer
        # Ogni key/value layer ha dimensione [batch, heads, seq_len, head_dim]
        batch_size = 1
        n_heads = model_gen.config.n_head
        head_dim = model_gen.config.n_embd // n_heads
        seq_len = 1

        # Gli hidden states che abbiamo calcolato devono essere trasformati nel formato corretto
        # Creiamo dei valori fittizi per key e value per simulare un contesto
        reshaped_hidden = hidden_state.view(batch_size, 1, -1)

        # Creiamo tensori vuoti per simulare past_key_values
        past_key_values = []
        for _ in range(model_gen.config.n_layer):
            # Per ogni layer, creiamo key e value tensors
            key_layer = torch.zeros(batch_size, n_heads, seq_len, head_dim).to(device)
            value_layer = reshaped_hidden.repeat(1, n_heads, 1).view(batch_size, n_heads, seq_len, head_dim)
            past_key_values.append((key_layer, value_layer))

        # Trasforma in tupla di tuple
        past_key_values = tuple(past_key_values)

        # Genera il testo
        with torch.no_grad():
            output_sequences = model_gen.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperatura,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=num_campioni,
                past_key_values=past_key_values if num_campioni == 1 else None,  # Past values solo per singolo campione
                no_repeat_ngram_size=3,  # Evita ripetizioni di trigrammi
                early_stopping=True  # Ferma la generazione quando è logico
            )

        # Decodifica il testo generato
        if num_campioni == 1:
            testo_generato = tokenizer_gen.decode(output_sequences[0], skip_special_tokens=True)
            return testo_generato
        else:
            testi_generati = [tokenizer_gen.decode(seq, skip_special_tokens=True) for seq in output_sequences]
            return testi_generati

    except Exception as e:
        print(f"Errore durante la generazione: {e}")
        # Fallback: genera senza condizionamento iniziale
        return genera_testo_semplice(max_length, temperatura, num_campioni, prompt)


def genera_testo_semplice(max_length=100, temperatura=0.7, num_campioni=1, prompt=None):
    """
    Genera testo senza condizionamento (metodo di fallback)

    Args:
        max_length: Lunghezza massima del testo generato
        temperatura: Temperatura per la generazione
        num_campioni: Numero di varianti da generare
        prompt: Prompt opzionale per indirizzare la generazione

    Returns:
        Testo generato o lista di testi se num_campioni > 1
    """
    try:
        with torch.no_grad():
            # Prepara l'input
            if prompt:
                input_ids = tokenizer_gen(prompt, return_tensors="pt").input_ids.to(device)
            else:
                input_ids = tokenizer_gen(tokenizer_gen.bos_token, return_tensors="pt").input_ids.to(device)

            # Genera il testo
            output_sequences = model_gen.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperatura,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=num_campioni,
                no_repeat_ngram_size=3
            )

        # Decodifica il testo generato
        if num_campioni == 1:
            testo_generato = tokenizer_gen.decode(output_sequences[0], skip_special_tokens=True)
            return testo_generato
        else:
            testi_generati = [tokenizer_gen.decode(seq, skip_special_tokens=True) for seq in output_sequences]
            return testi_generati

    except Exception as e:
        print(f"Errore anche nella generazione semplice: {e}")
        return "Errore nella generazione del testo." if num_campioni == 1 else [
                                                                                   "Errore nella generazione del testo."] * num_campioni


# Funzione per testare la ricostruzione
def testa_ricostruzione(testo, temperatura=0.7, num_campioni=3):
    """
    Testa la ricostruzione di un testo

    Args:
        testo: Testo da testare
        temperatura: Temperatura per la generazione
        num_campioni: Numero di varianti da generare
    """
    print(f"\nTest con: \"{testo}\"")

    # Crea embedding per il testo
    vettore = crea_embedding([testo])[0]

    # Ricostruisci dal vettore
    risultati = ricostruisci_da_vettore(vettore, temperatura=temperatura, num_campioni=num_campioni)

    print("Ricostruzioni:")
    if isinstance(risultati, list):
        for i, r in enumerate(risultati):
            print(f"{i + 1}. {r}")
    else:
        print(risultati)

    return vettore, risultati


# Funzione principale
def main():
    """Funzione principale per testare il codice"""
    import argparse

    parser = argparse.ArgumentParser(description="Vector-to-Text Reconstruction")
    parser.add_argument("--train", action="store_true", help="Addestra il modello")
    parser.add_argument("--test", action="store_true", help="Testa il modello")
    parser.add_argument("--epochs", type=int, default=100, help="Numero di epoche per l'addestramento")
    parser.add_argument("--batch-size", type=int, default=4, help="Dimensione batch per l'addestramento")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--input-text", type=str, help="Testo di input per il test")
    parser.add_argument("--temperatura", type=float, default=0.7, help="Temperatura per la generazione")

    args = parser.parse_args()

    # Se non vengono specificate opzioni, abilita entrambe
    if not (args.train or args.test):
        args.train = True
        args.test = True

    # Testo di esempio per addestramento
    testo_originale = """L'intelligenza artificiale sta rivoluzionando molti campi della tecnologia moderna.
    Gli algoritmi di machine learning permettono di analizzare enormi quantità di dati e trovare pattern nascosti.
    I modelli linguistici sono in grado di comprendere e generare testo in modo sempre più sofisticato.
    Le applicazioni pratiche spaziano dalla medicina alla finanza, dal marketing all'automotive.
    I sistemi di computer vision possono riconoscere oggetti e persone nelle immagini.
    Le reti neurali profonde hanno trasformato il campo della percezione artificiale.
    I transformer hanno permesso di creare modelli di linguaggio con una capacità di comprensione senza precedenti.
    Gli assistenti virtuali diventano sempre più utili e naturali nell'interazione.
    L'apprendimento per rinforzo ha consentito di creare sistemi in grado di giocare e battere gli umani.
    L'etica dell'intelligenza artificiale è un tema sempre più importante nel dibattito pubblico.
    """

    print("Elaborazione del testo in corso...")

    # Converti il testo in vettori
    vettori, paragrafi = testo_a_vettori(testo_originale)

    if args.train:
        # Addestra il decoder
        print(f"\nAddestramento del decoder...")
        print(f"- Epoche: {args.epochs}")
        print(f"- Batch size: {args.batch_size}")
        print(f"- Learning rate: {args.learning_rate}")

        addestra_decoder(
            paragrafi,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

    if args.test:
        # Carica l'ultimo checkpoint se non è stato eseguito l'addestramento
        if not args.train:
            carica_ultimo_checkpoint(decoder)

        # Usa il testo di input specificato o altrimenti il primo paragrafo
        if args.input_text:
            testa_ricostruzione(args.input_text, temperatura=args.temperatura)
        else:
            # Test con alcuni paragrafi di esempio
            print("\nTest con paragrafi originali:")
            testo_ricostruito = ricostruisci_da_vettore(vettori[0], temperatura=args.temperatura)

            print(f"\nOriginale: {paragrafi[0]}")
            print(f"Ricostruito: {testo_ricostruito}")

            # Test con nuovi testi
            testi_test = [
                "Le reti neurali profonde hanno rivoluzionato il campo della visione artificiale.",
                "L'elaborazione del linguaggio naturale è diventata molto più precisa grazie ai transformer.",
                "Gli algoritmi di deep learning richiedono grandi quantità di dati per l'addestramento."
            ]

            for test_text in testi_test:
                testa_ricostruzione(test_text, temperatura=args.temperatura)

            # Test di interpolazione tra vettori
            print("\nTest di interpolazione tra vettori:")
            indice1, indice2 = 0, 2  # Indici di due paragrafi

            # Interpola tra i due vettori
            vettore_interpolato = 0.5 * vettori[indice1] + 0.5 * vettori[indice2]
            # Normalizza il vettore
            vettore_interpolato = vettore_interpolato / np.linalg.norm(vettore_interpolato)

            print(f"Paragrafo 1: {paragrafi[indice1]}")
            print(f"Paragrafo 2: {paragrafi[indice2]}")
            print("Testo interpolato:")

            testo_interpolato = ricostruisci_da_vettore(vettore_interpolato, temperatura=args.temperatura)
            print(testo_interpolato)


if __name__ == "__main__":
    main()