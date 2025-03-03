import spacy
import re

class AnalizzatoreFrase:
    def __init__(self):
        # Caricamento del modello italiano di spaCy
        try:
            self.nlp = spacy.load("it_core_news_sm")
        except OSError:
            print("Modello italiano non trovato. Installarlo con: python -m spacy download it_core_news_sm")
            raise

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
                "gruppo": gruppo_appartenenza
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
    frase_esempio = "Il gatto nero corre velocemente nel giardino fiorito."
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