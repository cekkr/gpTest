import spacy
import re
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import nltk
from nltk.corpus import wordnet as wn

# Configurazione globale per la scelta del dispositivo di accelerazione
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps" if torch.mps.is_available() else DEVICE
print(f"Utilizzando dispositivo: {DEVICE}")

class AnalizzatoreFrase:
    def __init__(self):
        # Verifica e download delle risorse NLTK necessarie
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Scaricamento di WordNet...")
            nltk.download('wordnet')
            nltk.download('omw-1.4')  # Open Multilingual WordNet

        # Caricamento del modello italiano di spaCy
        try:
            self.nlp = spacy.load("it_core_news_sm")
        except OSError:
            print("Modello italiano non trovato. Installarlo con: python -m spacy download it_core_news_sm")
            raise

        # Caricamento del modello di classificazione semantica (usando un modello multilingue)
        try:
            model_name = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
            self.ner_model = pipeline("ner", 
                                      model=model_name, 
                                      tokenizer=model_name, 
                                      device=0 if DEVICE == "cuda" else -1)
        except Exception as e:
            print(f"Errore nel caricamento del modello di classificazione: {e}")
            print("Proseguimento senza classificazione semantica avanzata.")
            self.ner_model = None

        # Definizione delle categorie grammaticali in italiano
        self.pos_map = {
            "NOUN": "sostantivo",
            "PROPN": "nome proprio",
            "VERB": "verbo",
            "AUX": "verbo ausiliare",
            "ADJ": "aggettivo",
            "ADV": "avverbio",
            "ADP": "preposizione",
            "CCONJ": "congiunzione coordinante",
            "SCONJ": "congiunzione subordinante",
            "PRON": "pronome",
            "DET": "articolo/determinante",
            "NUM": "numero",
            "PART": "particella",
            "INTJ": "interiezione",
            "PUNCT": "punteggiatura",
            "SYM": "simbolo",
            "X": "altro"
        }
        
        # Definizione delle funzioni sintattiche in italiano
        self.dep_map = {
            "ROOT": "predicato",
            "nsubj": "soggetto",
            "obj": "complemento oggetto",
            "iobj": "complemento di termine",
            "obl": "complemento indiretto",
            "amod": "modificatore aggettivale",
            "advmod": "modificatore avverbiale",
            "aux": "ausiliare",
            "cop": "copula",
            "det": "determinante",
            "case": "marca di caso",
            "nmod": "modificatore nominale",
            "appos": "apposizione",
            "conj": "congiunzione",
            "cc": "coordinatore",
            "mark": "marcatore",
            "punct": "punteggiatura",
            "acl": "clausola relativa",
            "advcl": "clausola avverbiale",
            "ccomp": "complemento frasale",
            "xcomp": "complemento predicativo",
            "compound": "composto",
            "fixed": "espressione fissa",
            "flat": "sequenza piatta",
            "parataxis": "paratassi",
            "expl": "espletivo",
            "csubj": "soggetto frasale",
            "csubjpass": "soggetto passivo frasale",
            "nummod": "modificatore numerico"
        }
        
        # Definizione dei tempi verbali in italiano (semplificata)
        self.tempi_verbali = {
            "Mood=Ind|Tense=Pres": "presente indicativo",
            "Mood=Ind|Tense=Past": "passato indicativo",
            "Mood=Ind|Tense=Fut": "futuro indicativo",
            "Mood=Ind|Tense=Imp": "imperfetto indicativo",
            "Mood=Sub|Tense=Pres": "presente congiuntivo",
            "Mood=Sub|Tense=Imp": "imperfetto congiuntivo",
            "Mood=Cnd": "condizionale",
            "VerbForm=Inf": "infinito",
            "VerbForm=Part": "participio",
            "VerbForm=Ger": "gerundio"
        }
        
        # Mappatura delle entità semantiche
        self.entity_map = {
            "PER": "persona",
            "LOC": "luogo",
            "ORG": "organizzazione",
            "MISC": "miscellanea",
            "O": "altro"
        }
        
        # Categorie semantiche predefinite per alcune parole comuni
        self.categorie_semantiche = {
            "gatto": ["animale", "animale domestico", "mammifero", "felino"],
            "cane": ["animale", "animale domestico", "mammifero", "canide"],
            "casa": ["edificio", "struttura", "abitazione", "immobile"],
            "mare": ["corpo d'acqua", "ambiente naturale", "ecosistema"],
            "automobile": ["veicolo", "mezzo di trasporto", "bene materiale"],
            "libro": ["oggetto", "pubblicazione", "opera letteraria", "fonte di informazione"]
            # Altre parole possono essere aggiunte qui
        }

    def ottieni_sinonimi(self, lemma):
        """Ottiene sinonimi per un lemma dato usando WordNet"""
        sinonimi = set()
        
        # Prova con WordNet in italiano
        for synset in wn.synsets(lemma, lang='ita'):
            for lemma_obj in synset.lemmas(lang='ita'):
                sinonimi.add(lemma_obj.name())
                
        # Se non sono stati trovati sinonimi in italiano, prova in inglese e traduci
        if not sinonimi and lemma not in ['il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una']:
            try:
                # Tentativo di traduzione usando WordNet
                for synset in wn.synsets(self.nlp(lemma)[0]._.lemma):
                    for lemma_eng in synset.lemmas():
                        synsets_it = wn.synsets(lemma_eng.name(), lang='ita')
                        for syn_it in synsets_it:
                            for lemma_it in syn_it.lemmas(lang='ita'):
                                sinonimi.add(lemma_it.name())
            except:
                pass
                
        return list(sinonimi)

    def categorizza_semanticamente(self, lemma, pos):
        """Categorizza semanticamente una parola usando WordNet e mappature predefinite"""
        categorie = []
        
        # Usa categorie predefinite se disponibili
        if lemma.lower() in self.categorie_semantiche:
            categorie.extend(self.categorie_semantiche[lemma.lower()])
        
        # Aggiungi categorie da WordNet
        if pos in ["NOUN", "PROPN", "VERB", "ADJ"]:
            try:
                # Cerco synsets in italiano
                synsets_it = wn.synsets(lemma, lang='ita')
                
                # Se non trovo synsets in italiano, provo in inglese
                if not synsets_it:
                    synsets_it = wn.synsets(lemma)
                
                for synset in synsets_it[:3]:  # Limita a 3 synset per evitare troppe categorie
                    # Aggiungi ipernimi (categorie più generali)
                    for hypernym in synset.hypernyms()[:2]:
                        if hypernym.lemmas():
                            try:
                                nome_categoria = hypernym.lemmas()[0].name().replace('_', ' ')
                                categorie.append(nome_categoria)
                            except:
                                continue
            except:
                pass
        
        # Rimuovi duplicati e limita il numero di categorie
        return list(set(categorie))[:5]  # Massimo 5 categorie

    def categorizza_gruppi(self, doc):
        """Identifica e categorizza gruppi di parole (sintagmi)"""
        gruppi = []
        gruppo_corrente = None
        tipo_gruppo_corrente = None
        
        # Identificazione naïve dei sintagmi principali
        for token in doc:
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                # Inizio di un potenziale sintagma nominale
                if gruppo_corrente and tipo_gruppo_corrente != "nominale":
                    gruppi.append({"tipo": tipo_gruppo_corrente, "parole": gruppo_corrente})
                    gruppo_corrente = [token.text]
                    tipo_gruppo_corrente = "nominale"
                elif not gruppo_corrente:
                    gruppo_corrente = [token.text]
                    tipo_gruppo_corrente = "nominale"
                else:
                    gruppo_corrente.append(token.text)
            
            elif token.pos_ == "VERB" or token.pos_ == "AUX":
                # Inizio di un potenziale sintagma verbale
                if gruppo_corrente and tipo_gruppo_corrente != "verbale":
                    gruppi.append({"tipo": tipo_gruppo_corrente, "parole": gruppo_corrente})
                    gruppo_corrente = [token.text]
                    tipo_gruppo_corrente = "verbale"
                elif not gruppo_corrente:
                    gruppo_corrente = [token.text]
                    tipo_gruppo_corrente = "verbale"
                else:
                    gruppo_corrente.append(token.text)
            
            elif token.pos_ == "ADP":
                # Inizio di un potenziale sintagma preposizionale
                if gruppo_corrente and tipo_gruppo_corrente != "preposizionale":
                    gruppi.append({"tipo": tipo_gruppo_corrente, "parole": gruppo_corrente})
                    gruppo_corrente = [token.text]
                    tipo_gruppo_corrente = "preposizionale"
                elif not gruppo_corrente:
                    gruppo_corrente = [token.text]
                    tipo_gruppo_corrente = "preposizionale"
                else:
                    gruppo_corrente.append(token.text)
            
            elif token.pos_ == "ADJ":
                if gruppo_corrente:
                    gruppo_corrente.append(token.text)
                else:
                    gruppo_corrente = [token.text]
                    tipo_gruppo_corrente = "aggettivale"
            
            elif token.pos_ == "PUNCT":
                # La punteggiatura termina un gruppo
                if gruppo_corrente:
                    gruppi.append({"tipo": tipo_gruppo_corrente, "parole": gruppo_corrente})
                    gruppo_corrente = None
                    tipo_gruppo_corrente = None
            
            elif gruppo_corrente:
                # Altre parti del discorso che possono far parte di un gruppo esistente
                gruppo_corrente.append(token.text)
        
        # Aggiungi l'ultimo gruppo se presente
        if gruppo_corrente:
            gruppi.append({"tipo": tipo_gruppo_corrente, "parole": gruppo_corrente})
        
        return gruppi

    def identifica_entita_semantiche(self, frase):
        """Identifica entità semantiche nella frase usando il modello NER"""
        if not self.ner_model:
            return {}
            
        try:
            # Esecuzione del riconoscimento delle entità
            entities = self.ner_model(frase)
            
            # Organizzazione dei risultati per parola
            entity_map = {}
            for entity in entities:
                word = entity['word']
                if word.startswith('##'):  # Gestione dei token WordPiece
                    continue
                    
                entity_type = entity['entity']
                score = entity['score']
                
                if word not in entity_map or score > entity_map[word]['score']:
                    entity_map[word] = {
                        'tipo': entity_type,
                        'descrizione': self.entity_map.get(entity_type, "altro"),
                        'score': score
                    }
                    
            return entity_map
        except Exception as e:
            print(f"Errore nell'identificazione delle entità: {e}")
            return {}

    def analizza_frase(self, frase):
        """
        Analizza una frase e restituisce un array di oggetti con proprietà grammaticali per ogni parola.
        
        Args:
            frase (str): La frase da analizzare
            
        Returns:
            list: Un array di dizionari con le proprietà di ogni parola
        """
        # Processamento della frase con spaCy
        doc = self.nlp(frase)
        
        # Identificazione dei gruppi
        gruppi = self.categorizza_gruppi(doc)
        
        # Identificazione delle entità semantiche
        entita_semantiche = self.identifica_entita_semantiche(frase)
        
        # Creazione dell'array di risultati
        risultato = []
        
        # Analisi token per token
        for token in doc:
            # Determinazione del tempo verbale (se applicabile)
            tempo_verbale = None
            if token.pos_ in ["VERB", "AUX"]:
                for key, value in self.tempi_verbali.items():
                    if all(feat in token.morph.to_dict().items() for feat in [item.split("=") for item in key.split("|")]):
                        tempo_verbale = value
                        break
            
            # Ricerca del gruppo di appartenenza
            gruppo_appartenenza = None
            for i, gruppo in enumerate(gruppi):
                if token.text in gruppo["parole"]:
                    gruppo_appartenenza = {
                        "indice": i, 
                        "tipo": gruppo["tipo"],
                        "parole": gruppo["parole"]
                    }
                    break
                    
            # Ottenimento di sinonimi
            sinonimi = self.ottieni_sinonimi(token.lemma_)
            
            # Categorizzazione semantica
            categorie = self.categorizza_semanticamente(token.lemma_, token.pos_)
            
            # Informazioni sull'entità semantica
            entita = None
            if token.text in entita_semantiche:
                entita = {
                    "tipo": entita_semantiche[token.text]['tipo'],
                    "descrizione": entita_semantiche[token.text]['descrizione'],
                    "confidenza": entita_semantiche[token.text]['score']
                }
            
            # Creazione della struttura dati per la parola
            parola_info = {
                "parola": token.text,
                "lemma": token.lemma_,
                "indice": token.i,
                "posizione_originale": token.idx,
                "categoria_grammaticale": {
                    "codice": token.pos_,
                    "descrizione": self.pos_map.get(token.pos_, "sconosciuto")
                },
                "funzione_sintattica": {
                    "codice": token.dep_,
                    "descrizione": self.dep_map.get(token.dep_, "sconosciuto")
                },
                "morfologia": {
                    "genere": token.morph.get("Gender", [""])[0],
                    "numero": token.morph.get("Number", [""])[0],
                    "persona": token.morph.get("Person", [""])[0],
                    "tempo_verbale": tempo_verbale,
                    "modo": token.morph.get("Mood", [""])[0],
                },
                "gruppo": gruppo_appartenenza,
                "semantica": {
                    "sinonimi": sinonimi,
                    "categorie": categorie,
                    "entita": entita
                }
            }
            
            # Aggiungi la parola al risultato
            risultato.append(parola_info)
        
        return risultato

def analizza(frase):
    """
    Funzione principale per analizzare una frase.
    
    Args:
        frase (str): La frase da analizzare
        
    Returns:
        list: Un array di dizionari con le proprietà di ogni parola
    """
    try:
        analizzatore = AnalizzatoreFrase()
        return analizzatore.analizza_frase(frase)
    except Exception as e:
        return [{"errore": str(e)}]


# Esempio di utilizzo
if __name__ == "__main__":
    frase_esempio = "Il gatto nero corre velocemente nel giardino fiorito della vecchia casa."
    risultato = analizza(frase_esempio)
    
    # Stampa del risultato in formato leggibile
    for i, parola in enumerate(risultato):
        print(f"\n[Parola {i+1}]: {parola['parola']}")
        print(f"  Lemma: {parola['lemma']}")
        print(f"  Categoria: {parola['categoria_grammaticale']['descrizione']}")
        print(f"  Funzione: {parola['funzione_sintattica']['descrizione']}")
        
        # Stampare informazioni morfologiche se disponibili
        morfologia = []
        if parola['morfologia']['genere']:
            morfologia.append(f"genere: {parola['morfologia']['genere']}")
        if parola['morfologia']['numero']:
            morfologia.append(f"numero: {parola['morfologia']['numero']}")
        if parola['morfologia']['tempo_verbale']:
            morfologia.append(f"tempo: {parola['morfologia']['tempo_verbale']}")
            
        if morfologia:
            print(f"  Morfologia: {', '.join(morfologia)}")
            
        # Stampare informazioni sul gruppo
        if parola['gruppo']:
            print(f"  Gruppo: tipo {parola['gruppo']['tipo']}, parole: {' '.join(parola['gruppo']['parole'])}")
            
        # Stampare informazioni semantiche
        print(f"  Sinonimi: {', '.join(parola['semantica']['sinonimi']) if parola['semantica']['sinonimi'] else 'nessuno'}")
        print(f"  Categorie semantiche: {', '.join(parola['semantica']['categorie']) if parola['semantica']['categorie'] else 'nessuna'}")
        
        if parola['semantica']['entita']:
            print(f"  Entità: {parola['semantica']['entita']['descrizione']} ({parola['semantica']['entita']['tipo']}) con confidenza {parola['semantica']['entita']['confidenza']:.2f}")