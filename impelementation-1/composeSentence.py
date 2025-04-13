#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import nltk
from nltk.tokenize import sent_tokenize

# Assicurati che NLTK abbia le risorse necessarie
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class RuleBasedCombiner:
    """
    Combinatore di concetti basato su regole linguistiche.
    Non richiede modelli di machine learning.
    """

    def __init__(self):
        # Categorie speciali e relativi modificatori
        self.special_categories = {
            "nome di": {"prefix": "il", "connector": ""},
            "colore": {"prefix": "di colore", "connector": ""},
            "razza": {"prefix": "di razza", "connector": ""},
            "età": {"prefix": "di", "connector": "anni"},
            "tipo": {"prefix": "un tipo di", "connector": ""},
            "marca": {"prefix": "di marca", "connector": ""},
            "taglia": {"prefix": "di taglia", "connector": ""}
        }

        # Possessivi e loro forme
        self.possessives = {
            "mio": {"article": "il mio", "plural": "i miei"},
            "tuo": {"article": "il tuo", "plural": "i tuoi"},
            "suo": {"article": "il suo", "plural": "i suoi"},
            "nostro": {"article": "il nostro", "plural": "i nostri"},
            "vostro": {"article": "il vostro", "plural": "i vostri"},
            "loro": {"article": "il loro", "plural": "i loro"}
        }

        # Entità comuni e loro articoli
        self.entities = {
            "gatto": {"article": "il", "plural": "i", "gender": "m"},
            "cane": {"article": "il", "plural": "i", "gender": "m"},
            "libro": {"article": "il", "plural": "i", "gender": "m"},
            "casa": {"article": "la", "plural": "le", "gender": "f"},
            "auto": {"article": "l'", "plural": "le", "gender": "f"},
            "automobile": {"article": "l'", "plural": "le", "gender": "f"},
            "macchina": {"article": "la", "plural": "le", "gender": "f"}
        }

    def _extract_entity(self, attribute, category):
        """Estrae l'entità da un attributo con una categoria specifica."""
        if category in attribute:
            parts = attribute.split(category)
            if len(parts) > 1:
                return parts[1].strip()
        return attribute

    def _categorize_attributes(self, attributes):
        """Classifica gli attributi in categorie."""
        categorized = {
            "name": None,
            "type": None,
            "ownership": None,
            "characteristics": [],
            "other": []
        }

        for attr in attributes:
            # Verifica se è un attributo di nome
            if "nome di" in attr or "si chiama" in attr:
                categorized["name"] = attr
            # Verifica se è un attributo di tipo
            elif any(cat in attr for cat in ["razza", "tipo", "specie", "marca", "modello"]):
                categorized["type"] = attr
            # Verifica se è un attributo di possesso
            elif any(poss in attr for poss in self.possessives.keys()):
                categorized["ownership"] = attr
            # Verifica se è una caratteristica
            elif any(cat in attr for cat in ["colore", "età", "taglia", "dimensione"]):
                categorized["characteristics"].append(attr)
            # Altrimenti è un attributo generico
            else:
                categorized["other"].append(attr)

        return categorized

    def combine(self, concepts):
        """Combina i concetti in frasi naturali usando regole linguistiche."""
        # Raggruppa i concetti per soggetto
        subject_groups = {}
        for concept in concepts:
            if " è " not in concept:
                continue

            subject, attribute = concept.split(" è ", 1)
            subject = subject.strip()
            attribute = attribute.strip()

            if subject not in subject_groups:
                subject_groups[subject] = []

            subject_groups[subject].append(attribute)

        results = []
        for subject, attributes in subject_groups.items():
            # Categorizza gli attributi
            categorized = self._categorize_attributes(attributes)

            # Inizia a costruire la frase
            sentence_parts = []
            entity_type = None
            possessive = None

            # 1. Gestione del nome e del tipo di entità
            if categorized["name"]:
                for category, info in self.special_categories.items():
                    if category in categorized["name"]:
                        entity_type = self._extract_entity(categorized["name"], category)
                        break

            # 2. Gestione del possesso
            if categorized["ownership"]:
                for poss in self.possessives:
                    if poss in categorized["ownership"]:
                        possessive = poss
                        break

            # 3. Gestione del tipo specifico
            if categorized["type"] and not entity_type:
                for category, info in self.special_categories.items():
                    if category in categorized["type"]:
                        entity_type = self._extract_entity(categorized["type"], category)
                        break

            # 4. Costruzione della frase base
            if entity_type and possessive:
                # "Neve è il mio gatto"
                article = self.possessives[possessive]["article"]
                sentence_parts.append(f"{subject} è {article} {entity_type}")
            elif entity_type:
                # "Neve è un gatto"
                article = "un" if entity_type in self.entities and self.entities[entity_type][
                    "gender"] == "m" else "una"
                sentence_parts.append(f"{subject} è {article} {entity_type}")
            elif possessive:
                # "Neve è mio"
                sentence_parts.append(f"{subject} è {possessive}")
            else:
                # Fallback se non abbiamo né entità né possesso
                sentence = f"{subject}"

                # Aggiungi tutti gli attributi
                all_attrs = []
                if categorized["characteristics"]:
                    all_attrs.extend(categorized["characteristics"])
                if categorized["other"]:
                    all_attrs.extend(categorized["other"])

                if all_attrs:
                    sentence += " è"
                    if len(all_attrs) == 1:
                        sentence += f" {all_attrs[0]}"
                    else:
                        sentence += f" {', '.join(all_attrs[:-1])} e {all_attrs[-1]}"

                results.append(sentence)
                continue

            # 5. Aggiungi caratteristiche e altri attributi
            additional_attrs = []
            if categorized["characteristics"]:
                additional_attrs.extend(categorized["characteristics"])
            if categorized["other"]:
                additional_attrs.extend(categorized["other"])

            # Costruisci la frase finale
            if len(additional_attrs) == 0:
                results.append(sentence_parts[0])
            elif len(additional_attrs) == 1:
                if "razza" in additional_attrs[0] or "colore" in additional_attrs[0]:
                    for category, info in self.special_categories.items():
                        if category in additional_attrs[0]:
                            value = self._extract_entity(additional_attrs[0], category)
                            results.append(f"{sentence_parts[0]} {info['prefix']} {value}")
                            break
                    else:
                        results.append(f"{sentence_parts[0]} {additional_attrs[0]}")
                else:
                    results.append(f"{sentence_parts[0]} {additional_attrs[0]}")
            else:
                # Formatta gli attributi in modo leggibile
                formatted_attrs = []
                for attr in additional_attrs:
                    formatted = attr
                    for category, info in self.special_categories.items():
                        if category in attr:
                            value = self._extract_entity(attr, category)
                            formatted = f"{info['prefix']} {value}"
                            break
                    formatted_attrs.append(formatted)

                if len(formatted_attrs) == 2:
                    results.append(f"{sentence_parts[0]}, {formatted_attrs[0]} e {formatted_attrs[1]}")
                else:
                    formatted_str = ", ".join(formatted_attrs[:-1]) + f" e {formatted_attrs[-1]}"
                    results.append(f"{sentence_parts[0]}, {formatted_str}")

        # Raffina ulteriormente i risultati
        refined_results = []
        for result in results:
            # Migliora la struttura della frase
            refined = result

            # Sostituisci "è di razza" con "è un"
            refined = re.sub(r'è (il|un) ([^,]+), di razza', r'è \1 \2 di razza', refined)

            # Migliora l'ordine degli aggettivi
            refined = re.sub(r'è (il mio|un) ([^,]+) siamese', r'è \1 gatto siamese', refined)

            refined_results.append(refined)

        return refined_results


class TemplateCombiner:
    """
    Combinatore di concetti basato su template predefiniti.
    Usa pattern comuni per creare frasi naturali.
    """

    def __init__(self):
        # Template per vari pattern di concetti
        self.templates = {
            "pet_name_type": {
                "pattern": {"nome di": ["gatto", "cane"], "razza": None},
                "template": "{subject} è {article} {pet_type} di razza {breed}"
            },
            "pet_name_ownership": {
                "pattern": {"nome di": ["gatto", "cane"], "mio|tuo|suo": None},
                "template": "{subject} è il {possessive} {pet_type}"
            },
            "pet_complete": {
                "pattern": {"nome di": ["gatto", "cane"], "mio|tuo|suo": None, "razza": None},
                "template": "{subject} è il {possessive} {pet_type} di razza {breed}"
            },
            "book_type_year": {
                "pattern": {"tipo": ["libro"], "pubblicato nel": None},
                "template": "{subject} è un {book_type} pubblicato nel {year}"
            }
        }

        # Articoli per vari tipi di entità
        self.articles = {
            "gatto": "un",
            "cane": "un",
            "libro": "un",
            "casa": "una",
            "auto": "un'",
            "automobile": "un'",
            "macchina": "una"
        }

        # Possessivi
        self.possessives = {
            "mio": "mio",
            "tuo": "tuo",
            "suo": "suo",
            "nostro": "nostro",
            "vostro": "vostro",
            "loro": "loro"
        }

    def _extract_info(self, attributes):
        """Estrae informazioni dagli attributi in base ai pattern."""
        info = {}

        for attr in attributes:
            # Estrai informazioni sul tipo di animale
            match = re.search(r'nome di (gatto|cane)', attr)
            if match:
                info["pet_type"] = match.group(1)
                continue

            # Estrai informazioni sulla razza
            match = re.search(r'razza (\w+)', attr)
            if match:
                info["breed"] = match.group(1)
                continue

            # Estrai informazioni sul possesso
            match = re.search(r'(mio|tuo|suo|nostro|vostro|loro)', attr)
            if match:
                info["possessive"] = match.group(1)
                continue

            # Estrai l'anno di pubblicazione
            match = re.search(r'pubblicato nel (\d{4})', attr)
            if match:
                info["year"] = match.group(1)
                continue

            # Estrai tipo di libro
            match = re.search(r'di (filosofia|storia|scienza|romanzo)', attr)
            if match:
                info["book_type"] = match.group(1)
                continue

            # Estrai altre informazioni rilevanti
            if "interessante" in attr:
                info["interesting"] = True
            if "bello" in attr or "bella" in attr:
                info["beautiful"] = True

        return info

    def _match_template(self, subject, attributes):
        """Trova il template migliore per i concetti dati."""
        info = self._extract_info(attributes)

        # Casi speciali comuni
        if "pet_type" in info and "breed" in info and "possessive" in info:
            return f"{subject} è il {info['possessive']} {info['pet_type']} di razza {info['breed']}"

        elif "pet_type" in info and "possessive" in info:
            return f"{subject} è il {info['possessive']} {info['pet_type']}"

        elif "pet_type" in info and "breed" in info:
            article = self.articles.get(info['pet_type'], "un")
            return f"{subject} è {article} {info['pet_type']} di razza {info['breed']}"

        elif "book_type" in info and "year" in info:
            book_desc = f"libro di {info['book_type']}"
            if "interesting" in info:
                book_desc = f"interessante {book_desc}"
            return f"{subject} è un {book_desc} pubblicato nel {info['year']}"

        # Se non troviamo un template specifico, costruiamo una frase generica
        return self._build_generic_sentence(subject, attributes, info)

    def _build_generic_sentence(self, subject, attributes, info=None):
        """Costruisce una frase generica combinando gli attributi."""
        if not info:
            info = self._extract_info(attributes)

        if len(attributes) == 1:
            return f"{subject} è {attributes[0]}"

        return f"{subject} è {', '.join(attributes[:-1])} e {attributes[-1]}"

    def combine(self, concepts):
        """Combina i concetti usando template predefiniti."""
        # Raggruppa i concetti per soggetto
        subject_groups = {}
        for concept in concepts:
            if " è " not in concept:
                continue

            subject, attribute = concept.split(" è ", 1)
            subject = subject.strip()
            attribute = attribute.strip()

            if subject not in subject_groups:
                subject_groups[subject] = []

            subject_groups[subject].append(attribute)

        results = []
        for subject, attributes in subject_groups.items():
            result = self._match_template(subject, attributes)
            results.append(result)

        return results


def main():
    print("\n=== Test del Combinatore basato su Regole ===")
    rule_combiner = RuleBasedCombiner()

    # Test 1: Esempio del gatto
    concepts_gatto = [
        "Neve è nome di gatto",
        "Neve è mio",
        "Neve è siamese"
    ]

    results1 = rule_combiner.combine(concepts_gatto)
    print("Input:", concepts_gatto)
    for result in results1:
        print(f"Risultato: {result}")

    # Test 2: Esempio del libro
    concepts_libro = [
        "Il libro è interessante",
        "Il libro è di filosofia",
        "Il libro è stato pubblicato nel 2020"
    ]

    results2 = rule_combiner.combine(concepts_libro)
    print("\nInput:", concepts_libro)
    for result in results2:
        print(f"Risultato: {result}")

    # Test 3: Esempio più complesso
    concepts_complessi = [
        "Fido è nome di cane",
        "Fido è razza labrador",
        "Fido è colore nero",
        "Fido è di Marco"
    ]

    results3 = rule_combiner.combine(concepts_complessi)
    print("\nInput:", concepts_complessi)
    for result in results3:
        print(f"Risultato: {result}")

    print("\n=== Test del Combinatore basato su Template ===")
    template_combiner = TemplateCombiner()

    # Riutilizziamo gli stessi esempi
    template_results1 = template_combiner.combine(concepts_gatto)
    print("Input:", concepts_gatto)
    for result in template_results1:
        print(f"Risultato: {result}")

    template_results2 = template_combiner.combine(concepts_libro)
    print("\nInput:", concepts_libro)
    for result in template_results2:
        print(f"Risultato: {result}")

    template_results3 = template_combiner.combine(concepts_complessi)
    print("\nInput:", concepts_complessi)
    for result in template_results3:
        print(f"Risultato: {result}")


if __name__ == "__main__":
    main()