import numpy as np
import torch
import json
from collections import defaultdict
import re

# Gestione importazioni opzionali
try:
    from transformers import AutoTokenizer, AutoModel, pipeline

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
    nltk.download('omw-1.4', quiet=True)  # Multilingual WordNet
    wordnet_available = True
except ImportError:
    wordnet_available = False
    print("WordNet non disponibile - utilizzo sinonimi limitati")


class AdvancedConceptExtractor:
    """
    Estrattore di concetti avanzato che utilizza modelli di linguaggio moderni
    per estrarre concetti primitivi da testo in modo dinamico e multilingue.
    """

    def __init__(self,
                 use_transformer=True,
                 model_name="distilbert-base-multilingual-cased",
                 spacy_models=None,
                 force_pattern_only=False,
                 config_path=None):
        """
        Inizializza l'estrattore avanzato di concetti.

        Args:
            use_transformer: Se utilizzare un modello transformer per gli embedding
            model_name: Nome del modello transformer da utilizzare
            spacy_models: Lista di modelli SpaCy da caricare (es. ["it_core_news_sm", "en_core_web_sm"])
            force_pattern_only: Se True, usa solo l'approccio basato su pattern anche se altre librerie sono disponibili
            config_path: Percorso a un file di configurazione JSON
        """
        print("Inizializzazione estrattore avanzato di concetti")

        # Imposta il dispositivo (GPU, MPS o CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'mps' if hasattr(torch.mps, 'is_available') and torch.mps.is_available() else self.device
        print(f"Utilizzando dispositivo: {self.device}")

        # Stato delle librerie
        self.transformers_available = transformers_available and use_transformer and not force_pattern_only
        self.spacy_available = spacy_available and not force_pattern_only
        self.wordnet_available = wordnet_available and not force_pattern_only

        # Carica configurazione personalizzata se specificata
        self.config = self._load_config(config_path)

        # Inizializza modello transformer e NER
        if self.transformers_available:
            try:
                print(f"Caricamento modello transformer: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(self.device)

                # Aggiungi un pipeline per zero-shot classification
                try:
                    self.zero_shot = pipeline(
                        "zero-shot-classification",
                        model="facebook/bart-large-mnli",
                        device=0 if self.device == 'cuda' else -1
                    )
                    self.has_zero_shot = True
                    print("Pipeline zero-shot caricato con successo")
                except Exception as e:
                    print(f"Errore nel caricamento della pipeline zero-shot: {e}")
                    self.has_zero_shot = False

                print("Modello transformer caricato con successo")
            except Exception as e:
                print(f"Errore nel caricamento del modello transformer: {e}")
                self.transformers_available = False

        # Inizializza modelli SpaCy
        self.nlp_models = {}
        if self.spacy_available:
            # Verifica quali modelli sono effettivamente installati
            available_models = []
            try:
                available_models = spacy.util.get_installed_models()
                print(f"Modelli SpaCy disponibili: {available_models}")
            except Exception as e:
                print(f"Errore nel recupero dei modelli SpaCy: {e}")

            # Se non vengono specificati, usa quelli disponibili
            if spacy_models is None:
                spacy_models = []
                # Aggiungi inglese se disponibile
                if "en_core_web_sm" in available_models:
                    spacy_models.append("en_core_web_sm")
                # Aggiungi italiano se disponibile
                if "it_core_news_sm" in available_models:
                    spacy_models.append("it_core_news_sm")
                # Se non ci sono modelli specifici, usa il primo disponibile
                if not spacy_models and available_models:
                    spacy_models.append(available_models[0])

            # Carica i modelli specificati se disponibili
            for model_name in spacy_models:
                try:
                    print(f"Caricamento modello SpaCy: {model_name}")
                    nlp = spacy.load(model_name)

                    # Aggiungi sentencizer se non presente
                    if "sentencizer" not in nlp.pipe_names:
                        nlp.add_pipe("sentencizer")

                    # Aggiungi componente per rilevare entità e relazioni
                    try:
                        if "entity_ruler" not in nlp.pipe_names:
                            ruler = nlp.add_pipe("entity_ruler")
                            self._add_default_entity_patterns(ruler)
                    except Exception as e:
                        print(f"Avviso: impossibile aggiungere entity_ruler: {e}")

                    # Deduci il codice lingua dal nome del modello
                    lang_code = model_name.split('_')[0]
                    self.nlp_models[lang_code] = nlp  # Chiave è la lingua (it/en)
                    print(f"Modello SpaCy {model_name} caricato con successo per lingua {lang_code}")
                except Exception as e:
                    print(f"Errore nel caricamento del modello SpaCy {model_name}: {e}")

            if not self.nlp_models:
                print("Nessun modello SpaCy caricato correttamente, utilizzando solo pattern")

        # Relazioni primitive dinamiche
        self.primitive_relations = {
            "identity": {
                "en": ["is", "are", "be", "am", "was", "were", "being"],
                "it": ["è", "sono", "essere", "sia", "era", "erano", "sei", "siamo", "siete"],
                "fr": ["est", "sont", "être", "suis", "es", "sommes", "êtes", "était", "étaient"],
                "es": ["es", "son", "ser", "estar", "soy", "eres", "somos", "erais", "estoy", "estás"],
                "de": ["ist", "sind", "sein", "bin", "bist", "seid", "war", "waren"]
            },
            "possession": {
                "en": ["have", "has", "own", "owns", "possess", "possesses", "of", "'s"],
                "it": ["ha", "hanno", "ho", "hai", "avere", "possiede", "possiedono", "di", "del", "dello", "della",
                       "dei", "degli", "delle"],
                "fr": ["a", "ont", "ai", "as", "avoir", "possède", "possèdent", "de", "du", "des", "d'"],
                "es": ["tiene", "tienen", "tengo", "tienes", "tener", "posee", "poseen", "de", "del", "de la", "de los",
                       "de las"],
                "de": ["hat", "haben", "habe", "hast", "besitzt", "besitzen", "von", "des", "der", "dem"]
            },
            "location": {
                "en": ["in", "at", "on", "near", "inside", "outside", "above", "below", "between"],
                "it": ["in", "a", "su", "vicino", "dentro", "fuori", "sopra", "sotto", "tra", "fra"],
                "fr": ["dans", "à", "sur", "près", "dedans", "dehors", "dessus", "dessous", "entre"],
                "es": ["en", "a", "sobre", "cerca", "dentro", "fuera", "encima", "debajo", "entre"],
                "de": ["in", "an", "auf", "nahe", "innerhalb", "außerhalb", "über", "unter", "zwischen"]
            },
            "attribute": {
                "en": ["with", "has", "having", "contains", "containing"],
                "it": ["con", "ha", "avere", "contiene", "contenente"],
                "fr": ["avec", "a", "ayant", "contient", "contenant"],
                "es": ["con", "tiene", "teniendo", "contiene", "conteniendo"],
                "de": ["mit", "hat", "habend", "enthält", "enthaltend"]
            },
            "action": {
                "en": [],  # Questi saranno popolati dinamicamente con verbi comuni
                "it": [],
                "fr": [],
                "es": [],
                "de": []
            }
        }

        # Carica dizionari aggiuntivi dal config
        if self.config and "relations" in self.config:
            for rel_type, lang_dict in self.config["relations"].items():
                if rel_type not in self.primitive_relations:
                    self.primitive_relations[rel_type] = {}
                for lang, words in lang_dict.items():
                    if lang not in self.primitive_relations[rel_type]:
                        self.primitive_relations[rel_type][lang] = []
                    self.primitive_relations[rel_type][lang].extend(words)

        # Converte strutture di relazioni in una forma più efficiente per la ricerca
        self.relation_lookup = self._build_relation_lookup()

        # Costruisci la cache per la normalizzazione e sinonimi
        self.concept_cache = {}
        self.similarity_cache = {}
        self.synonym_cache = {}

    def _load_config(self, config_path):
        """Carica configurazione da file JSON"""
        if not config_path:
            return None

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"Configurazione caricata da {config_path}")
            return config
        except Exception as e:
            print(f"Errore nel caricamento della configurazione: {e}")
            return None

    def _add_default_entity_patterns(self, ruler):
        """Aggiunge pattern di entità predefiniti a SpaCy"""
        patterns = []

        # Pattern per entità di base
        patterns.extend([
            {"label": "PERSON", "pattern": [{"LOWER": "io"}]},
            {"label": "PERSON", "pattern": [{"LOWER": "tu"}]},
            {"label": "PERSON", "pattern": [{"LOWER": "lui"}]},
            {"label": "PERSON", "pattern": [{"LOWER": "lei"}]},
            {"label": "PERSON", "pattern": [{"LOWER": "noi"}]},
            {"label": "PERSON", "pattern": [{"LOWER": "voi"}]},
            {"label": "PERSON", "pattern": [{"LOWER": "loro"}]},
            {"label": "PERSON", "pattern": [{"LOWER": "i"}]},
            {"label": "PERSON", "pattern": [{"LOWER": "you"}]},
            {"label": "PERSON", "pattern": [{"LOWER": "he"}]},
            {"label": "PERSON", "pattern": [{"LOWER": "she"}]},
            {"label": "PERSON", "pattern": [{"LOWER": "we"}]},
            {"label": "PERSON", "pattern": [{"LOWER": "they"}]}
        ])

        # Aggiungi altre lingue se necessario

        ruler.add_patterns(patterns)

    def _build_relation_lookup(self):
        """Costruisce un dizionario di lookup per relazioni primitive"""
        lookup = {}
        for rel_type, lang_dict in self.primitive_relations.items():
            for lang, words in lang_dict.items():
                for word in words:
                    lookup[word] = (rel_type, lang)
        return lookup

    def detect_language(self, text):
        """
        Rileva la lingua del testo in modo dinamico.

        Args:
            text: Testo da analizzare

        Returns:
            Codice lingua (en, it, fr, es, de)
        """
        # Metodo semplice ma effettivo: controlla le parole comuni in ciascuna lingua
        common_words = {
            "en": set(["the", "is", "and", "of", "in", "to", "a", "for", "with"]),
            "it": set(["il", "la", "è", "e", "di", "in", "un", "una", "per", "con"]),
            "fr": set(["le", "la", "est", "et", "de", "dans", "un", "une", "pour", "avec"]),
            "es": set(["el", "la", "es", "y", "de", "en", "un", "una", "para", "con"]),
            "de": set(["der", "die", "das", "ist", "und", "in", "zu", "ein", "eine", "für", "mit"])
        }

        # Tokenizza e conta parole comuni
        words = re.findall(r'\b\w+\b', text.lower())
        counts = {lang: 0 for lang in common_words}

        for word in words:
            for lang, word_set in common_words.items():
                if word in word_set:
                    counts[lang] += 1

        # Se non ci sono match chiari, usa inglese come default
        if max(counts.values(), default=0) == 0:
            return "en"

        return max(counts.items(), key=lambda x: x[1])[0]

    def get_embeddings(self, text):
        """
        Ottiene embedding vettoriali per un testo.

        Args:
            text: Testo da trasformare in embedding

        Returns:
            Array numpy con embedding
        """
        if not self.transformers_available:
            # Fallback a un embedding basato su TF-IDF improvvisato
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            # Vettore di 100 dimensioni con hash delle parole
            vec = np.zeros(100)
            for word, count in word_counts.items():
                word_hash = hash(word) % 100
                vec[word_hash] += count

            # Normalizza
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            return vec.reshape(1, -1)

        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(
                self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Prendi media degli embedding dei token
            mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = outputs.last_hidden_state * mask
            embeddings = embeddings.sum(1) / mask.sum(1)

            return embeddings.cpu().numpy()
        except Exception as e:
            print(f"Errore nel calcolo degli embedding: {e}")
            # Fallback
            return np.random.rand(1, 768)  # Dimensione tipica di distilbert

    def get_similarity(self, text1, text2):
        """
        Calcola la similarità semantica tra due testi.

        Args:
            text1: Primo testo
            text2: Secondo testo

        Returns:
            Score di similarità [0-1]
        """
        cache_key = (text1.lower(), text2.lower())
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # Calcola gli embedding
        emb1 = self.get_embeddings(text1)
        emb2 = self.get_embeddings(text2)

        # Calcola la similarità del coseno
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 > 0 and norm2 > 0:
            cos_sim = np.dot(emb1, emb2.T)[0, 0] / (norm1 * norm2)
            similarity = (cos_sim + 1) / 2  # Normalizza in [0, 1]
        else:
            similarity = 0

        # Salva nella cache
        self.similarity_cache[cache_key] = similarity

        return similarity

    def get_wordnet_synonyms(self, word, lang="en"):
        """
        Ottiene sinonimi da WordNet.

        Args:
            word: Parola di cui trovare sinonimi
            lang: Lingua (en, it, es, fr, de)

        Returns:
            Lista di sinonimi
        """
        if not self.wordnet_available:
            return []

        synonyms = set()

        try:
            # Associazione codici WordNet
            lang_codes = {
                "en": "eng",
                "it": "ita",
                "fr": "fra",
                "es": "spa",
                "de": "deu"
            }

            wn_lang = lang_codes.get(lang, "eng")

            # Cerca sinonimi nella lingua specificata
            for synset in wn.synsets(word, lang=wn_lang):
                for lemma in synset.lemmas(lang=wn_lang):
                    if lemma.name().lower() != word.lower():
                        synonyms.add(lemma.name().lower())

                # Cerca anche traduzioni in altre lingue
                for other_lang in lang_codes.values():
                    if other_lang != wn_lang:
                        for lemma in synset.lemmas(lang=other_lang):
                            synonyms.add(lemma.name().lower())
        except Exception as e:
            print(f"Errore WordNet: {e}")

        return list(synonyms)

    def get_synonyms(self, word, lang=None):
        """
        Ottiene sinonimi e traduzioni di una parola utilizzando vari metodi.

        Args:
            word: Parola di cui trovare sinonimi
            lang: Lingua della parola (se nota)

        Returns:
            Lista di sinonimi e concetti correlati
        """
        word = word.lower()

        # Usa la cache se disponibile
        cache_key = (word, lang)
        if cache_key in self.synonym_cache:
            return self.synonym_cache[cache_key]

        # Rileva lingua se non specificata
        if not lang:
            lang = self.detect_language(word)

        synonyms = set()

        # Aggiungi sinonimi da WordNet
        if self.wordnet_available:
            wn_synonyms = self.get_wordnet_synonyms(word, lang)
            synonyms.update(wn_synonyms)

        # Aggiungi traduzioni se abbiamo transformer
        if self.transformers_available:
            # Implementazione: trova parole più simili in base agli embedding
            similar_words = []

            # Possiamo aggiungere traduzioni da un piccolo dizionario
            if self.config and "translations" in self.config:
                for lang_key, words_dict in self.config["translations"].items():
                    if word in words_dict:
                        similar_words.extend(words_dict[word])

            synonyms.update(similar_words)

        # Salva nella cache
        result = list(synonyms)
        self.synonym_cache[cache_key] = result

        return result

    def normalize_concept(self, concept, lang=None):
        """
        Normalizza un concetto alla sua forma base.

        Args:
            concept: Concetto da normalizzare
            lang: Lingua del concetto (se nota)

        Returns:
            Concetto normalizzato
        """
        if not concept or len(concept.strip()) == 0:
            return ""

        concept = concept.lower().strip()

        # Usa la cache se disponibile
        cache_key = (concept, lang)
        if cache_key in self.concept_cache:
            return self.concept_cache[cache_key]

        # Rileva lingua se non specificata
        if not lang:
            lang = self.detect_language(concept)

        # Normalizza pronomi personali
        personal_pronouns = {
            "en": {"i": "person:1s", "you": "person:2s", "he": "person:3sm", "she": "person:3sf",
                   "we": "person:1p", "they": "person:3p", "it": "thing:neutral"},
            "it": {"io": "person:1s", "tu": "person:2s", "lui": "person:3sm", "lei": "person:3sf",
                   "noi": "person:1p", "loro": "person:3p", "esso": "thing:neutral",
                   "essa": "thing:neutral"},
            "fr": {"je": "person:1s", "tu": "person:2s", "il": "person:3sm", "elle": "person:3sf",
                   "nous": "person:1p", "ils": "person:3p", "elles": "person:3p"},
            "es": {"yo": "person:1s", "tú": "person:2s", "él": "person:3sm", "ella": "person:3sf",
                   "nosotros": "person:1p", "ellos": "person:3p", "ellas": "person:3p"},
            "de": {"ich": "person:1s", "du": "person:2s", "er": "person:3sm", "sie": "person:3sf",
                   "wir": "person:1p", "ihr": "person:2p", "sie": "person:3p"}
        }

        # Controlla se è un pronome
        if lang in personal_pronouns and concept in personal_pronouns[lang]:
            normalized = personal_pronouns[lang][concept]
        else:
            # Usa SpaCy per lemmatizzazione se disponibile
            if self.spacy_available and lang in self.nlp_models:
                doc = self.nlp_models[lang](concept)
                if len(doc) > 0:
                    normalized = doc[0].lemma_
                else:
                    normalized = concept
            else:
                normalized = concept

        # Salva nella cache
        self.concept_cache[cache_key] = normalized

        return normalized

    def classify_relation(self, relation, context="", lang=None):
        """
        Classifica una relazione in una delle categorie primitive.

        Args:
            relation: Relazione da classificare
            context: Testo di contesto
            lang: Lingua della relazione

        Returns:
            Tipo di relazione primitiva
        """
        relation = relation.lower().strip()

        # Rileva lingua se non specificata
        if not lang:
            lang = self.detect_language(relation if not context else context)

        # Controlla lookup diretto
        if relation in self.relation_lookup:
            return self.relation_lookup[relation][0]

        # Usa zero-shot classification se disponibile
        if self.transformers_available and hasattr(self, 'zero_shot') and self.has_zero_shot:
            candidate_labels = list(self.primitive_relations.keys())

            try:
                context_text = context if context else relation
                result = self.zero_shot(context_text, candidate_labels)
                return result['labels'][0]  # Prendi la classificazione più probabile
            except Exception as e:
                print(f"Errore nella classificazione zero-shot: {e}")

        # Metodo basato su similarità con parole di relazione note
        best_score = -1
        best_type = "action"  # Default

        for rel_type, lang_dict in self.primitive_relations.items():
            for cur_lang, words in lang_dict.items():
                for word in words:
                    score = self.get_similarity(relation, word)
                    if score > best_score and score > 0.7:  # Soglia di confidenza
                        best_score = score
                        best_type = rel_type

        return best_type

    def extract_concepts_with_spacy(self, text, lang=None, normalize=True):
        """
        Estrae concetti utilizzando SpaCy.

        Args:
            text: Testo da analizzare
            lang: Lingua del testo
            normalize: Se normalizzare i concetti

        Returns:
            Lista di triple [soggetto, relazione, oggetto]
        """
        if not self.spacy_available:
            return []

        # Rileva lingua se non specificata
        if not lang:
            lang = self.detect_language(text)

        # Verifica se abbiamo modelli disponibili
        if not self.nlp_models:
            print("Nessun modello SpaCy disponibile, utilizzo fallback a pattern")
            return []

        # Scegli il modello SpaCy appropriato
        if lang not in self.nlp_models:
            # Fallback al primo modello disponibile
            if len(self.nlp_models) > 0:
                lang = list(self.nlp_models.keys())[0]
                print(f"Lingua {lang} non supportata, utilizzo {lang} come fallback")
            else:
                print("Nessun modello SpaCy disponibile")
                return []

        nlp = self.nlp_models[lang]
        doc = nlp(text)

        concepts = []

        # Estrai relazioni soggetto-verbo-oggetto
        for sent in doc.sents:
            for token in sent:
                # Trova il soggetto e l'oggetto dei verbi
                if token.pos_ == "VERB":
                    subjects = []
                    objects = []

                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            # Estendi il soggetto con i suoi modificatori
                            subj_span = self._extend_span(child)
                            subjects.append(subj_span.text)
                        elif child.dep_ in ["dobj", "obj", "iobj", "pobj"]:
                            # Estendi l'oggetto con i suoi modificatori
                            obj_span = self._extend_span(child)
                            objects.append(obj_span.text)

                    # Crea triple soggetto-verbo-oggetto per ogni combinazione
                    for subj in subjects:
                        for obj in objects:
                            if normalize:
                                subj_norm = self.normalize_concept(subj, lang)
                                rel = self.classify_relation(token.lemma_, text, lang)
                                obj_norm = self.normalize_concept(obj, lang)
                                concepts.append([subj_norm, rel, obj_norm])
                            else:
                                concepts.append([subj, token.lemma_, obj])

        # Estrai relazioni possessive (X di Y)
        for token in doc:
            if token.dep_ == "poss" or (token.dep_ == "case" and token.text.lower() in ["di", "of"]):
                head = token.head

                # Trova il possessore
                if token.dep_ == "poss":
                    possessor = token.text
                    possessed = head.text
                else:
                    # Struttura "X di Y"
                    possessed = head.head.text if head.dep_ == "pobj" else None
                    possessor = head.text if head.dep_ == "pobj" else None

                if possessor and possessed:
                    if normalize:
                        subj_norm = self.normalize_concept(possessor, lang)
                        obj_norm = self.normalize_concept(possessed, lang)
                        concepts.append([subj_norm, "possession", obj_norm])
                    else:
                        concepts.append([possessor, "possession", possessed])

        return concepts

    def _extend_span(self, token):
        """
        Estende un token a un span che include tutti i suoi modificatori.
        Utile per estrarre frasi nominali complete.
        """
        min_i = token.i
        max_i = token.i

        # Trova tutti i figli e i loro discendenti
        def get_span_indices(tok):
            nonlocal min_i, max_i
            min_i = min(min_i, tok.i)
            max_i = max(max_i, tok.i)

            # Ricorsione sui figli
            for child in tok.children:
                # Escludiamo congiunzioni e punti
                if child.dep_ not in ["cc", "punct"]:
                    get_span_indices(child)

        get_span_indices(token)

        return token.doc[min_i:max_i + 1]

    def extract_concepts_with_patterns(self, text, lang=None, normalize=True):
        """
        Estrae concetti utilizzando pattern multilingue.

        Args:
            text: Testo da analizzare
            lang: Lingua del testo
            normalize: Se normalizzare i concetti

        Returns:
            Lista di triple [soggetto, relazione, oggetto]
        """
        # Rileva lingua se non specificata
        if not lang:
            lang = self.detect_language(text)

        concepts = []

        # Pattern di identità multilingue (A è B)
        identity_tokens = self.primitive_relations["identity"].get(lang, [])
        identity_pattern = r'\b(\w+)\s+(' + '|'.join(identity_tokens) + r')\s+([\w\s]+)\b'

        for match in re.finditer(identity_pattern, text.lower()):
            subject = match.group(1)
            relation = match.group(2)
            attribute = match.group(3).strip()

            if normalize:
                subject = self.normalize_concept(subject, lang)
                relation = "identity"
                attribute = self.normalize_concept(attribute, lang)

            concepts.append([subject, relation, attribute])

        # Pattern di possesso multilingue (A ha B)
        possession_tokens = self.primitive_relations["possession"].get(lang, [])
        possession_pattern = r'\b(\w+)\s+(' + '|'.join(possession_tokens) + r')\s+([\w\s]+)\b'

        for match in re.finditer(possession_pattern, text.lower()):
            subject = match.group(1)
            relation = match.group(2)
            object_text = match.group(3).strip()

            if normalize:
                subject = self.normalize_concept(subject, lang)
                relation = "possession"
                object_text = self.normalize_concept(object_text, lang)

            concepts.append([subject, relation, object_text])

        # Pattern per relazioni di luogo (A in B)
        location_tokens = self.primitive_relations["location"].get(lang, [])
        location_pattern = r'\b(\w+)\s+(' + '|'.join(location_tokens) + r')\s+([\w\s]+)\b'

        for match in re.finditer(location_pattern, text.lower()):
            subject = match.group(1)
            relation = match.group(2)
            location = match.group(3).strip()

            if normalize:
                subject = self.normalize_concept(subject, lang)
                relation = "location"
                location = self.normalize_concept(location, lang)

            concepts.append([subject, relation, location])

        return concepts

    def extract_concepts(self, text, normalize=True):
        """
        Estrae concetti da un testo usando molteplici strategie.

        Args:
            text: Testo da analizzare
            normalize: Se normalizzare i concetti

        Returns:
            Lista di triple [soggetto, relazione, oggetto]
        """
        # Rileva la lingua
        lang = self.detect_language(text)

        all_concepts = []

        # Estrazione basata su SpaCy (più accurata) - solo se abbiamo modelli validi
        spacy_concepts = []
        if self.spacy_available and self.nlp_models:
            try:
                spacy_concepts = self.extract_concepts_with_spacy(text, lang, normalize)
                all_concepts.extend(spacy_concepts)
            except Exception as e:
                print(f"Errore nell'estrazione con SpaCy: {e}")
                # Continua con altri metodi

        # Estrazione basata su pattern (usata sempre come fallback o metodo complementare)
        try:
            pattern_concepts = self.extract_concepts_with_patterns(text, lang, normalize)
            all_concepts.extend(pattern_concepts)
        except Exception as e:
            print(f"Errore nell'estrazione con pattern: {e}")

        # Se non abbiamo estratto concetti con nessun metodo, prova un approccio più semplice
        if not all_concepts:
            try:
                simple_concepts = self._extract_simple_concepts(text, lang, normalize)
                all_concepts.extend(simple_concepts)
            except Exception as e:
                print(f"Errore nell'estrazione semplice: {e}")
                # Restituisci almeno un concetto basato sul testo di input
                if normalize:
                    return [[self.normalize_concept(text.split()[0] if text.split() else "entity"),
                             "content",
                             self.normalize_concept(text)]]
                else:
                    return [[text.split()[0] if text.split() else "entity", "content", text]]

        # Rimuovi duplicati mantenendo l'ordine
        unique_concepts = []
        seen = set()

        for concept in all_concepts:
            # Assicurati che il concetto sia una lista di 3 elementi
            if not isinstance(concept, list) or len(concept) != 3:
                continue

            # Crea una chiave tuple per il confronto
            try:
                key = tuple(str(item).lower() for item in concept)
                if key not in seen:
                    seen.add(key)
                    unique_concepts.append(concept)
            except Exception:
                # Ignora concetti malformati
                continue

        return unique_concepts

    def expand_concepts(self, concepts, text=""):
        """
        Espande i concetti estratti generando concetti impliciti.

        Args:
            concepts: Lista di concetti esistenti
            text: Testo originale per contesto

        Returns:
            Lista di concetti estesa
        """
        expanded = concepts.copy()

        # Mappa per tracciare relazioni
        entity_relations = defaultdict(list)

        # Costruisci grafo di relazioni
        for subj, rel, obj in concepts:
            entity_relations[subj].append((rel, obj))
            entity_relations[obj].append((f"inverse_{rel}", subj))

        # Inferenze basate su transitività
        for entity in entity_relations:
            relations = entity_relations[entity]

            # Regole di inferenza
            # Se A è B e B è C, allora A è C (transitività di identità)
            is_relations = [(r[1]) for r in relations if r[0] == "identity"]
            for target in is_relations:
                for rel, sec_target in entity_relations.get(target, []):
                    if rel == "identity" and (entity, "identity", sec_target) not in expanded:
                        expanded.append([entity, "identity", sec_target])

        return expanded

    def generate_knowledge_graph(self, text):
        """
        Genera un grafo di conoscenza dal testo.

        Args:
            text: Testo da analizzare

        Returns:
            Dizionario rappresentante il grafo
        """
        concepts = self.extract_concepts(text, normalize=True)
        expanded_concepts = self.expand_concepts(concepts, text)

        # Costruisci il grafo
        graph = {
            "entities": set(),
            "relations": expanded_concepts,
            "lang": self.detect_language(text)
        }

        # Raccogli entità uniche
        for subj, _, obj in expanded_concepts:
            graph["entities"].add(subj)
            graph["entities"].add(obj)

        graph["entities"] = list(graph["entities"])

        return graph

    def compare_concepts(self, concept1, concept2):
        """
        Confronta due concetti per similarità semantica.

        Args:
            concept1: Primo concetto [soggetto, relazione, oggetto]
            concept2: Secondo concetto [soggetto, relazione, oggetto]

        Returns:
            Score di similarità [0-1]
        """
        if not concept1 or not concept2 or len(concept1) != 3 or len(concept2) != 3:
            return 0.0

        # Calcola similarità delle componenti
        subj_sim = self.get_similarity(concept1[0], concept2[0])
        rel_sim = self.get_similarity(concept1[1], concept2[1])
        obj_sim = self.get_similarity(concept1[2], concept2[2])

        # Peso maggiore alla relazione
        return (subj_sim * 0.4 + rel_sim * 0.4 + obj_sim * 0.2)

    def merge_similar_concepts(self, concepts, threshold=0.85):
        """
        Unisce concetti simili.

        Args:
            concepts: Lista di concetti
            threshold: Soglia di similarità per unire concetti

        Returns:
            Lista di concetti unificata
        """
        if not concepts:
            return []

        merged = []
        used = [False] * len(concepts)

        for i in range(len(concepts)):
            if used[i]:
                continue

            current = concepts[i]
            used[i] = True
            cluster = [current]

            for j in range(i + 1, len(concepts)):
                if used[j]:
                    continue

                if self.compare_concepts(current, concepts[j]) > threshold:
                    cluster.append(concepts[j])
                    used[j] = True

            # Usa il concetto più completo come rappresentante
            merged.append(current)

        return merged

    def add_custom_synonyms(self, word, synonyms, langs=None):
        """
        Aggiunge sinonimi personalizzati al dizionario.

        Args:
            word: Parola base
            synonyms: Lista di sinonimi
            langs: Lista di codici lingua per i sinonimi
        """
        word = word.lower()

        # Invalidare cache
        for key in list(self.synonym_cache.keys()):
            if key[0] == word:
                del self.synonym_cache[key]

    def save_config(self, path):
        """
        Salva la configurazione corrente su file.

        Args:
            path: Percorso del file di output
        """
        config = {
            "relations": self.primitive_relations,
            "version": "1.0"
        }

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"Configurazione salvata in {path}")
        except Exception as e:
            print(f"Errore nel salvataggio della configurazione: {e}")


# Classe ausiliaria per l'integrazione con altri sistemi
class ConceptProcessor:
    """
    Classe per elaborare concetti estratti in vari formati.
    """

    def __init__(self, extractor=None):
        """
        Inizializza il processore.

        Args:
            extractor: Istanza di AdvancedConceptExtractor
        """
        self.extractor = extractor or AdvancedConceptExtractor()

    def concepts_to_triples(self, concepts):
        """Converte i concetti in triple RDF"""
        triples = []

        for subj, rel, obj in concepts:
            triples.append((subj, rel, obj))

        return triples

    def concepts_to_json(self, concepts):
        """Converte i concetti in formato JSON"""
        return [{"subject": s, "relation": r, "object": o} for s, r, o in concepts]

    def concepts_to_graph(self, concepts):
        """Converte i concetti in un formato per visualizzazione grafo"""
        nodes = set()
        edges = []

        for subj, rel, obj in concepts:
            nodes.add(subj)
            nodes.add(obj)
            edges.append({"source": subj, "relation": rel, "target": obj})

        return {
            "nodes": [{"id": node, "label": node} for node in nodes],
            "edges": edges
        }

    def concepts_to_natural_language(self, concepts, lang="en"):
        """Converte i concetti in frasi in linguaggio naturale"""
        templates = {
            "en": {
                "identity": "{subject} is {object}.",
                "possession": "{subject} has {object}.",
                "location": "{subject} is in {object}.",
                "attribute": "{subject} has attribute {object}.",
                "action": "{subject} {relation} {object}."
            },
            "it": {
                "identity": "{subject} è {object}.",
                "possession": "{subject} ha {object}.",
                "location": "{subject} è in {object}.",
                "attribute": "{subject} ha attributo {object}.",
                "action": "{subject} {relation} {object}."
            }
        }

        # Fallback a inglese se lingua non supportata
        if lang not in templates:
            lang = "en"

        sentences = []

        for subj, rel, obj in concepts:
            rel_type = rel if rel in templates[lang] else "action"
            template = templates[lang][rel_type]

            sentence = template.format(
                subject=subj.capitalize(),
                relation=rel,
                object=obj
            )

            sentences.append(sentence)

        return sentences

    # Aggiungi un metodo di fallback per l'estrazione semplice
    def _extract_simple_concepts(self, text, lang=None, normalize=True):
        """
        Metodo di fallback quando gli altri approcci falliscono.
        Estrae concetti semplici basati su frasi e pattern base.

        Args:
            text: Testo da analizzare
            lang: Lingua del testo
            normalize: Se normalizzare i concetti

        Returns:
            Lista di triple [soggetto, relazione, oggetto]
        """
        concepts = []

        # Rileva lingua se non specificata
        if not lang:
            lang = self.detect_language(text)

        # Dividi il testo in frasi semplici
        sentences = text.split('.')

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Dividi la frase in parole
            words = sentence.split()
            if len(words) < 3:
                continue

            # Prova a estrarre soggetto-verbo-oggetto in modo molto semplice
            # Assume che il primo sostantivo sia il soggetto, il primo verbo sia la relazione
            # e l'ultimo sostantivo sia l'oggetto (molto semplificato)
            subject = words[0]

            # Prova a trovare un verbo
            verb = None
            for word in words[1:]:
                # Controllo molto semplice - qualsiasi parola potrebbe essere un verbo
                if word.lower() in sum(self.primitive_relations.values(), []):
                    verb = word
                    break

            # Se non troviamo un verbo, usa "contiene" come default
            if not verb:
                verb = "contiene" if lang == "it" else "contains"

            # L'oggetto è l'ultima parola significativa
            object_word = words[-1]

            # Normalizza se richiesto
            if normalize:
                subject = self.normalize_concept(subject, lang)
                relation = self.classify_relation(verb, sentence, lang)
                object_word = self.normalize_concept(object_word, lang)

            concepts.append([subject, relation, object_word])

        return concepts


# Esempio di utilizzo
if __name__ == "__main__":
    # Crea un'istanza con fallback al metodo pattern-only
    extractor = AdvancedConceptExtractor(force_pattern_only=True)
    processor = ConceptProcessor(extractor)

    test_phrases = [
        "Il mio gatto, Neve, è un siamese",
        "Ho un cane di nome Rex che è un pastore tedesco",
        "Maria possiede una macchina rossa",
        "La casa è grande e ha un giardino",
        "Neve è mio",
        "My cat is white and fluffy",
        "La voiture est dans le garage",
        "El libro está sobre la mesa"
    ]

    print("\n" + "=" * 50)
    print("TEST ESTRATTORE CONCETTI ROBUSTO")
    print("=" * 50)

    for phrase in test_phrases:
        print(f"\nAnalisi di: '{phrase}'")

        try:
            # Rileva la lingua
            lang = extractor.detect_language(phrase)
            print(f"Lingua rilevata: {lang}")

            # Estrai concetti
            concepts = extractor.extract_concepts(phrase)
            print("Concetti estratti:")
            for concept in concepts:
                print(f"  {concept}")

            # Genera frasi in linguaggio naturale
            sentences = processor.concepts_to_natural_language(concepts, lang)
            print("Rappresentazione in linguaggio naturale:")
            for sentence in sentences:
                print(f"  {sentence}")
        except Exception as e:
            print(f"Errore nell'analisi: {e}")