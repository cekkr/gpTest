import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Tenta di importare moduli opzionali
try:
    from transformers import AutoTokenizer, AutoModel

    transformers_available = True
except ImportError:
    transformers_available = False
    print("Transformers non disponibile - utilizzo modalità base")

try:
    import spacy

    spacy_available = True
except ImportError:
    spacy_available = False
    print("SpaCy non disponibile - utilizzo modalità base")

try:
    import nltk
    from nltk.corpus import wordnet as wn

    nltk.download('wordnet', quiet=True)
    wordnet_available = True
except ImportError:
    wordnet_available = False
    print("WordNet non disponibile - utilizzo sinonimi limitati")


class RobustConceptExtractor:
    def __init__(self, use_transformer=True, model_name="distilbert-base-multilingual-cased"):
        """
        Inizializza l'estrattore robusto e semplificato di concetti.

        Args:
            use_transformer: Se utilizzare un modello transformer per gli embedding (se disponibile)
            model_name: Nome del modello transformer da utilizzare
        """
        print("Inizializzazione estrattore robusto di concetti")

        # Stato delle librerie
        self.transformers_available = transformers_available and use_transformer
        self.spacy_available = spacy_available
        self.wordnet_available = wordnet_available

        # Imposta il dispositivo (GPU o CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'mps' if torch.mps.is_available() else self.device
        print(f"Utilizzando dispositivo: {self.device}")

        # Tenta di caricare il modello transformer
        if self.transformers_available:
            try:
                print(f"Caricamento modello transformer: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(self.device)
                print("Modello transformer caricato con successo")
            except Exception as e:
                print(f"Errore nel caricamento del modello transformer: {e}")
                self.transformers_available = False

        # Tenta di caricare SpaCy
        if self.spacy_available:
            try:
                print("Caricamento modello SpaCy")
                self.nlp = spacy.load("en_core_web_sm")

                # Aggiungi il componente sentencizer per i confini di frase
                if "sentencizer" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("sentencizer")
                print("Modello SpaCy caricato con successo")
            except Exception as e:
                print(f"Errore nel caricamento del modello SpaCy: {e}")
                self.spacy_available = False

        # Dizionari per la normalizzazione
        self.possessive_mapping = {
            "mio": "io", "tuo": "tu", "suo": "lui/lei",
            "nostro": "noi", "vostro": "voi", "loro": "loro",
            "my": "io", "your": "tu", "his": "lui", "her": "lei",
            "our": "noi", "their": "loro"
        }

        # Mappe di sinonimi predefinite
        self.synonym_mapping = {
            "gatto": ["cat", "felino", "micio", "kitten", "kitty", "feline"],
            "cane": ["dog", "canino", "cucciolo", "puppy", "canine"],
            "casa": ["home", "abitazione", "appartamento", "apartment", "dwelling"],
            "macchina": ["car", "auto", "automobile", "vehicle", "veicolo"],
            "possedere": ["avere", "have", "own", "possiede", "possess"],
            "essere": ["è", "be", "is", "am", "are", "essere", "sia"]
        }

        # Aggiungi la mappa inversa per i sinonimi
        inverse_synonyms = {}
        for word, synonyms in self.synonym_mapping.items():
            for syn in synonyms:
                if syn not in inverse_synonyms:
                    inverse_synonyms[syn] = []
                inverse_synonyms[syn].append(word)

        # Unisci alle mappe esistenti
        for word, syns in inverse_synonyms.items():
            if word not in self.synonym_mapping:
                self.synonym_mapping[word] = syns

    def get_embeddings(self, text):
        """Ottiene embedding per una frase o parola"""
        if not self.transformers_available:
            # Fallback a un embedding semplice basato su caratteri
            return np.array([ord(c) for c in text[:100]]).reshape(1, -1)

        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(
                self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Prendi l'embedding del [CLS] token
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embeddings
        except Exception as e:
            print(f"Errore nel calcolo degli embedding: {e}")
            # Fallback
            return np.array([ord(c) for c in text[:100]]).reshape(1, -1)

    def get_word_similarity(self, word1, word2):
        """Calcola la similarità semantica tra due parole"""
        # Controlla sinonimi predefiniti
        if word1 in self.synonym_mapping and word2 in self.synonym_mapping[word1]:
            return 0.9
        if word2 in self.synonym_mapping and word1 in self.synonym_mapping[word2]:
            return 0.9

        # Prova con WordNet se disponibile
        if self.wordnet_available:
            try:
                synsets1 = wn.synsets(word1)
                synsets2 = wn.synsets(word2)

                if synsets1 and synsets2:
                    max_sim = max([s1.path_similarity(s2) or 0
                                   for s1 in synsets1
                                   for s2 in synsets2])
                    if max_sim > 0:
                        return max_sim
            except Exception as e:
                print(f"Errore in WordNet: {e}")

        # Fallback a embeddings
        emb1 = self.get_embeddings(word1)
        emb2 = self.get_embeddings(word2)

        return cosine_similarity(emb1, emb2)[0][0]

    def normalize_concept(self, word):
        """Normalizza un concetto alla sua forma base"""
        # Controlla se è un possessivo
        for poss, entity in self.possessive_mapping.items():
            if poss.lower() == word.lower() or self.get_word_similarity(word, poss) > 0.8:
                return entity

        # Controlla sinonimi
        for base_word, synonyms in self.synonym_mapping.items():
            if word.lower() == base_word.lower() or word.lower() in [s.lower() for s in synonyms]:
                return base_word

        return word

    def extract_concepts(self, text, normalize=True):
        """
        Estrae concetti da una frase di testo.

        Args:
            text: Testo da analizzare
            normalize: Se normalizzare i concetti

        Returns:
            Lista di concetti estratti
        """
        all_concepts = []

        # Estrae concetti utilizzando diverse strategie

        # 1. Pattern di identità (A è B)
        identity_patterns = [
            r'\b(\w+)\s+(è|sono|essere|is|are|be)\s+(\w+)\b',
            r'\b(\w+)\s+(è|sono|essere|is|are|be)\s+(?:un|una|a|an)?\s+(\w+)\b',
            r'\b(\w+)\s+(?:è|sono|essere|is|are|be)\s+(?:il|la|lo|i|gli|le|the)\s+(\w+)\s+(?:di|of)\s+(\w+)\b'
        ]

        for pattern in identity_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                groups = match.groups()
                if len(groups) >= 3:
                    subject = groups[0]
                    relation = groups[1]
                    attribute = groups[2]

                    if normalize:
                        subject = self.normalize_concept(subject)
                        relation = "identità"
                        attribute = self.normalize_concept(attribute)

                    all_concepts.append([subject, relation, attribute])

        # 2. Pattern di possesso (A ha B, B di A)
        possession_patterns = [
            r'\b(\w+)\s+(ha|hanno|ho|hai|possiede|possiedono|possiedo|possiedi|have|has|own|owns)\s+(?:un|una|a|an)?\s+(\w+)\b',
            r'\b(\w+)\s+(?:di|del|dello|della|dei|degli|delle|of)\s+(\w+)\b'
        ]

        for pattern in possession_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    if "di" in pattern or "of" in pattern:
                        # Struttura "B di A" → A ha B
                        object = groups[0]
                        subject = groups[1]
                    else:
                        # Struttura "A ha B"
                        subject = groups[0]
                        object = groups[-1]

                    if normalize:
                        subject = self.normalize_concept(subject)
                        relation = "possesso"
                        object = self.normalize_concept(object)
                    else:
                        relation = groups[1] if "di" not in pattern and "of" not in pattern else "possesso"

                    all_concepts.append([subject, relation, object])

        # 3. Pattern possessivi (il mio X)
        possessive_patterns = [
            r'\b(il\s+mio|la\s+mia|i\s+miei|le\s+mie|mio|mia|miei|mie|my)\s+(\w+)\b',
            r'\b(il\s+tuo|la\s+tua|i\s+tuoi|le\s+tue|tuo|tua|tuoi|tue|your)\s+(\w+)\b'
        ]

        for pattern in possessive_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                possessive = match.group(1)
                object = match.group(2)

                if normalize:
                    if possessive in self.possessive_mapping or any(p in possessive for p in self.possessive_mapping):
                        # Estrai la persona dal possessivo
                        for p, person in self.possessive_mapping.items():
                            if p in possessive:
                                subject = person
                                break
                    else:
                        subject = possessive

                    relation = "possesso"
                    object = self.normalize_concept(object)
                else:
                    subject = possessive
                    relation = "possesso"

                all_concepts.append([subject, relation, object])

        # 4. Pattern per nomi (X si chiama Y, X di nome Y)
        name_patterns = [
            r'\b(\w+)\s+(?:si\s+chiama|è\s+chiamato|è\s+chiamata|is\s+called|is\s+named)\s+(\w+)\b',
            r'\b(\w+)\s+(?:di\s+nome|named)\s+(\w+)\b',
            r'(?:il|la|lo|un|una|the|a|an)\s+(\w+)[,\s]+(\w+)[,\s]+'
        ]

        for pattern in name_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                entity = match.group(1)
                name = match.group(2)

                # Verifica se il nome è un nome proprio (prima lettera maiuscola)
                if name[0].islower() and name in text:
                    # Cerca nel testo originale
                    for word in text.split():
                        if word.lower() == name and word[0].isupper():
                            name = word
                            break

                if normalize:
                    entity = self.normalize_concept(entity)
                    relation = "nome"
                else:
                    relation = "nome"

                all_concepts.append([entity, relation, name])

        # 5. "Neve è mio" pattern specifico
        is_mine_pattern = r'\b(\w+)\s+(?:è|essere|is|be)\s+(mio|mia|tuo|tua|suo|sua|nostro|nostra|vostro|vostra|loro|my|your|his|her|our|their)\b'
        matches = re.finditer(is_mine_pattern, text.lower())

        for match in matches:
            object = match.group(1)  # Neve
            possessive = match.group(2)  # mio

            if normalize:
                subject = self.possessive_mapping.get(possessive, possessive)
                relation = "possesso"
                object = self.normalize_concept(object)
            else:
                subject = possessive
                relation = "essere"

            # Inverti qui perché "Neve è mio" significa "io possiedo Neve"
            all_concepts.append([subject, relation, object])

        # Usa anche SpaCy se disponibile
        if self.spacy_available and not all_concepts:
            try:
                spacy_concepts = self._extract_spacy_concepts(text, normalize)
                all_concepts.extend(spacy_concepts)
            except Exception as e:
                print(f"Errore nell'estrazione SpaCy: {e}")

        return all_concepts

    def _extract_spacy_concepts(self, text, normalize):
        """Estrae concetti usando SpaCy"""
        concepts = []
        doc = self.nlp(text)

        for token in doc:
            # Relazioni soggetto-verbo-oggetto
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subject = None
                object = None

                for child in token.children:
                    if child.dep_ == "nsubj":
                        subject = child.text
                    elif child.dep_ in ["dobj", "obj", "attr"]:
                        object = child.text

                if subject and object:
                    if normalize:
                        subj_norm = self.normalize_concept(subject)
                        rel_norm = self.normalize_concept(token.lemma_)
                        obj_norm = self.normalize_concept(object)
                        concepts.append([subj_norm, rel_norm, obj_norm])
                    else:
                        concepts.append([subject, token.lemma_, object])

        return concepts

    def get_synonyms(self, word, threshold=0.7):
        """Ottiene sinonimi di una parola"""
        synonyms = set()

        # Sinonimi predefiniti
        if word.lower() in self.synonym_mapping:
            synonyms.update(self.synonym_mapping[word.lower()])

        # Controlla parole simili nelle mappe di sinonimi
        for base_word, syns in self.synonym_mapping.items():
            if self.get_word_similarity(word, base_word) > threshold:
                synonyms.add(base_word)
                synonyms.update(syns)

        # WordNet
        if self.wordnet_available:
            try:
                for syn in wn.synsets(word):
                    for lemma in syn.lemmas():
                        if lemma.name() != word:
                            synonyms.add(lemma.name())
            except Exception as e:
                print(f"Errore WordNet: {e}")

        return list(synonyms)


# Esempio di utilizzo
if __name__ == "__main__":
    extractor = RobustConceptExtractor()

    test_phrases = [
        "Il mio gatto, Neve, è un siamese",
        "Ho un cane di nome Rex che è un pastore tedesco",
        "Maria possiede una macchina rossa",
        "La casa è grande e ha un giardino",
        "Neve è mio",
        "Neve essere mio",
        "My cat is white and fluffy"
    ]

    for phrase in test_phrases:
        print(f"\nAnalisi di: '{phrase}'")

        # Estrai concetti normalizzati
        concepts = extractor.extract_concepts(phrase, normalize=True)
        print("Concetti semantici:")
        for concept in concepts:
            print(f"  {concept}")

        # Estrai concetti non normalizzati
        raw_concepts = extractor.extract_concepts(phrase, normalize=False)
        print("Concetti testuali:")
        for concept in raw_concepts:
            print(f"  {concept}")

        # Dimostra analisi sinonimi
        main_words = [word for word in phrase.split() if len(word) > 3]
        if main_words:
            word = main_words[0]
            synonyms = extractor.get_synonyms(word)
            print(f"Sinonimi di '{word}': {synonyms}")

# python -m spacy download it_core_news_sm
# python -m spacy download xx_ent_wiki_sm
# python -m spacy download en_core_web_sm

'''
pip install torch transformers spacy nltk scikit-learn
python -m spacy download xx_ent_wiki_sm
python -m nltk.downloader wordnet

self.device = 'mps' if torch.mps.is_available() else self.device
'''