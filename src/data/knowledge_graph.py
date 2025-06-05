#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

"""
Module: knowledge_graph
Functionality: Loads raw knowledge graph data (e.g., YAGO3-10), 
              converts the format if needed so that self.triples, self.entities, and self.relations 
              are all in a unified (YAGO3-10-like) format, and provides statistical and preprocessing 
              functions for downstream usage.

Example usage:
    $ python knowledge_graph.py --kg yago3-10
"""

import os
import logging
import warnings
import argparse
import json
from collections import Counter, defaultdict, deque
import nltk
from nltk.corpus import wordnet as wn
from contextlib import suppress
from datasets import load_dataset
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from scipy.optimize import linprog          
from scipy.sparse.linalg import eigsh      
from transformers import AutoTokenizer, AutoModel
import torch
import re
import time
import seaborn as sns
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gensim.models import KeyedVectors
import requests
import openai
import heapq

# Configure logging output
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


_wd_label_cache = {}  # Cache for Q/P -> English label

def fetch_wikidata_label(qid, lang='en'):
    """
    Fetch the English label for an entity or relation (Qxxx or Pxxx) from the Wikidata API.
    Uses a global _wd_label_cache to avoid repeated calls.
    
    Args:
        qid (str): The Wikidata ID, e.g. 'Q2808' or 'P6379'.
        lang (str): Language code, default 'en'.

    Returns:
        str: The label in the requested language if found, otherwise the original qid.
    """
    # Check if we already have this ID in our cache
    if qid in _wd_label_cache:
        return _wd_label_cache[qid]

    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    try:
        # Optional: time.sleep(0.05)  # to avoid spamming the API too quickly
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        entities_data = data.get('entities', {})
        if qid in entities_data:
            labels = entities_data[qid].get('labels', {})
            if lang in labels:
                label_value = labels[lang].get('value')
                if label_value:
                    _wd_label_cache[qid] = label_value
                    return label_value
    except Exception as e:
        # You can log or ignore failures
        # logging.warning(f"Failed to fetch label for {qid}: {e}")
        pass

    # Fallback: store original ID in cache so we won't keep retrying
    _wd_label_cache[qid] = qid
    return qid


class KnowledgeGraph:
    def __init__(self, name):
        """
        Initialize a Knowledge Graph object.
        
        Args:
            name (str): Name of the knowledge graph, e.g., "wn18rr", "fb15k237", "nation", "umls", "yago3-10".
        """
        self.name = name.lower()
        
        # Automatically detect current script's directory
        self.data_path = os.path.dirname(os.path.abspath(__file__))
        
        # Store the originally loaded triples (could be numeric or textual, depending on the dataset)
        # 'train', 'valid', 'test' are the splits; 'total' is the union of all data
        self.triples = {
            'train': [],
            'valid': [],
            'test': [],
            'total': []
        }

        # Store negative / false triples (if applicable)
        self.negative_triples = {
            'train': [],
            'valid': [],
            'test': [],
            'total': []
        }

        self.threshold_selection_triples = {
            'positive': [],
            'negative': [],
            'relation': [] # select subset of relations for threshold selection
        }
        
        # Store entity and relation mappings; after loading, everything is converted to textual format
        self.entities = {}  # final format: {id: entity_text}
        self.relations = {} # final format: {id: relation_text}
        
        # For fb15k237 only: freebase MID-to-name mapping
        self.fb_mapping = {}

        # An adjacency list built from the unified text triples (for k-hop neighbor lookups)
        self.adj_list = None
        
        # Mapping from relation to its close relation: explicitly store for each relation its top-m semantically similar relations
        self.relation_close_mapping = {}

    # BERT-based Relation Mapping
    def compute_relation_mapping_bert(self, m=10, batch_size=16, bert_model="distilbert-base-uncased"):
        """
        Compute a mapping for each relation to its top m most semantically similar relations 
        using BERT-based sentence embeddings.
        
        Args:
            m (int): Number of top similar relations to select for each relation (default: 10).
            batch_size (int): Number of relation texts to process per batch.
            bert_model (str): Name of the pre-trained BERT model to use.
        
        Returns:
            dict: A dictionary mapping each relation text to a list of its top m similar relations.
        """
        # Ensure that relations are loaded and sorted for consistent ordering.
        relation_ids = sorted(self.relations.keys())
        relation_texts = [self.relations[rid] for rid in relation_ids]

        # Initialize the tokenizer and model.
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        model = AutoModel.from_pretrained(bert_model)
        model.eval()  # Set model to evaluation mode.

        embeddings = []
        # Process relation texts in batches.
        for i in range(0, len(relation_texts), batch_size):
            batch_texts = relation_texts[i:i+batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, 
                                      max_length=128, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
            # Mean pooling: average the token embeddings weighted by the attention mask.
            token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_dim)
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            # Normalize embeddings to unit length.
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())

        # Concatenate embeddings from all batches.
        all_embeddings = np.vstack(embeddings)
        # Compute cosine similarity matrix between all relation embeddings.
        sim_matrix = cosine_similarity(all_embeddings)

        mapping = {}
        # For each relation, select the top m similar relations (excluding itself).
        for idx, rid in enumerate(relation_ids):
            sims = sim_matrix[idx].copy()
            sims[idx] = -1  # Exclude self similarity.
            top_indices = sims.argsort()[-m:][::-1]
            similar_relations = [self.relations[relation_ids[i]] for i in top_indices if sims[i] > 0]
            mapping[self.relations[rid]] = similar_relations

        self.relation_close_mapping = mapping
        return mapping

    # TF-IDF based Relation Mapping
    def compute_relation_mapping_tfidf(self, m=10):
        """
        Compute a mapping for each relation to its top m most similar relations using TF-IDF features.
        This is an alternative to the BERT-based method.
        
        Args:
            m (int): Number of top similar relations to select (default: 10).
        
        Returns:
            dict: A dictionary mapping each relation text to a list of its top m similar relations.
        """
        relation_ids = sorted(self.relations.keys())
        relation_texts = [self.relations[rid] for rid in relation_ids]

        # Use TF-IDF vectorization.
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(relation_texts)
        sim_matrix = cosine_similarity(tfidf_matrix)

        mapping = {}
        for idx, rid in enumerate(relation_ids):
            sims = sim_matrix[idx].copy()
            sims[idx] = -1  # Exclude self similarity.
            top_indices = sims.argsort()[-m:][::-1]
            similar_relations = [self.relations[relation_ids[i]] for i in top_indices if sims[i] > 0]
            mapping[self.relations[rid]] = similar_relations

        self.relation_close_mapping = mapping
        return mapping

    def load_relation_mapping(self, method='bert', m=10, **kwargs):
        """
        Compute and load the relation mapping for the knowledge graph using the specified method.
        The mapping is stored in self.relation_close_mapping.
        
        Args:
            method (str): The method to use ('bert' for BERT-based, otherwise TF-IDF based).
            m (int): Number of top similar relations to store for each relation (default: 10).
            **kwargs: Additional keyword arguments for the chosen mapping computation method.
        
        Returns:
            dict: The computed relation mapping.
        """
        if method == 'bert':
            return self.compute_relation_mapping_bert(m=m, **kwargs)
        else:
            return self.compute_relation_mapping_tfidf(m=m)

    def filter_high_degree_entities(self, degree_threshold=100, save_filtered_info=False):
        """
        Filter out entities with degree higher than the specified threshold.
        Degree is defined as the total number of times an entity appears as either head or tail in triples.
        
        Args:
            degree_threshold (int): Maximum allowed degree for entities.
            save_filtered_info (bool): Whether to save filtered entities and triples information.
        
        Returns:
            tuple: (num_filtered_entities, num_filtered_triples) - counts of filtered entities and triples.
        """
        logging.info(f"Filtering entities with degree > {degree_threshold}...")
        
        # Calculate entity degrees across all splits
        entity_degrees = defaultdict(int)
        for split in ['train', 'valid', 'test']:
            for head, _, tail in self.triples[split]:
                entity_degrees[head] += 1
                entity_degrees[tail] += 1
        
        # Identify high-degree entities
        high_degree_entities = {entity for entity, degree in entity_degrees.items() 
                            if degree > degree_threshold}
        
        if not high_degree_entities:
            logging.info(f"No entities with degree > {degree_threshold} found.")
            return 0, 0
        
        # Log some statistics before filtering
        logging.info(f"Found {len(high_degree_entities)} entities with degree > {degree_threshold}")
        
        # Get details of high-degree entities for logging and saving
        high_degree_entity_details = {
            entity: entity_degrees[entity] for entity in high_degree_entities
        }
        
        # Sort by degree for logging
        top_entities = sorted(
            [(entity, degree) for entity, degree in high_degree_entity_details.items()],
            key=lambda x: x[1], reverse=True
        )[:10]  # Show top 10
        
        for entity, degree in top_entities:
            logging.info(f"High degree entity: {entity} (degree: {degree})")
        
        # Collect filtered triples - we'll need this for saving if requested
        filtered_triples_by_split = {
            'train': [],
            'valid': [],
            'test': []
        }
        
        # Create new triples dict without high-degree entities
        filtered_triples = {
            'train': [],
            'valid': [],
            'test': [],
            'total': []
        }
        
        total_filtered_triples = 0
        for split in ['train', 'valid', 'test']:
            for triple in self.triples[split]:
                head, relation, tail = triple
                if head not in high_degree_entities and tail not in high_degree_entities:
                    filtered_triples[split].append(triple)
                else:
                    # This triple contains at least one high-degree entity
                    filtered_triples_by_split[split].append(triple)
                    total_filtered_triples += 1
        
        # Update total
        filtered_triples['total'] = (
            filtered_triples['train'] +
            filtered_triples['valid'] +
            filtered_triples['test']
        )
        
        # Update the triples in the object
        self.triples = filtered_triples
        
        # Rebuild entity and relation sets from the filtered triples
        entities_set = set()
        relations_set = set()
        for split in ['train', 'valid', 'test']:
            for head, relation, tail in self.triples[split]:
                entities_set.add(head)
                entities_set.add(tail)
                relations_set.add(relation)
        
        # Update entity and relation dictionaries
        self.entities = {i: entity for i, entity in enumerate(sorted(entities_set))}
        self.relations = {i: relation for i, relation in enumerate(sorted(relations_set))}
        
        # Log filtering results
        logging.info(f"Filtered out {len(high_degree_entities)} high-degree entities")
        logging.info(f"Filtered out {total_filtered_triples} triples containing high-degree entities")
        logging.info(f"Remaining entities: {len(self.entities)}")
        logging.info(f"Remaining relations: {len(self.relations)}")
        logging.info(f"Remaining triples: {len(self.triples['total'])}")
        
        # Save filtered entity information if requested
        if save_filtered_info:
            self._save_filtered_entity_info(
                high_degree_entity_details, 
                filtered_triples_by_split,
                degree_threshold
            )
        
        return len(high_degree_entities), total_filtered_triples

    def _save_filtered_entity_info(self, high_degree_entity_details, filtered_triples_by_split, degree_threshold):
        """
        Save information about filtered high-degree entities and their triples to files.
        
        Args:
            high_degree_entity_details (dict): Dictionary mapping entity names to their degrees.
            filtered_triples_by_split (dict): Dictionary of filtered triples by split.
            degree_threshold (int): The degree threshold used for filtering.
        """
        # Create directory if needed
        save_dir = os.path.join(self.data_path, self.name)
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Save the filtered entity information with their degrees
        entities_file_path = os.path.join(save_dir, f"filtered_entities_degree_{degree_threshold}.json")
        
        # Sort entities by degree in descending order
        sorted_entities = sorted(
            [(entity, degree) for entity, degree in high_degree_entity_details.items()],
            key=lambda x: x[1], 
            reverse=True
        )
        
        entities_info = {
            "degree_threshold": degree_threshold,
            "filtered_entities_count": len(sorted_entities),
            "entities": [
                {"entity": entity, "degree": degree}
                for entity, degree in sorted_entities
            ]
        }
        
        with open(entities_file_path, 'w', encoding='utf-8') as f:
            json.dump(entities_info, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Saved filtered entity information to {entities_file_path}")
        
        # 2. Save the filtered triples by split
        triples_file_path = os.path.join(save_dir, f"filtered_triples_degree_{degree_threshold}.json")
        
        # Convert triples to a more JSON-friendly format
        json_friendly_triples = {}
        for split, triples in filtered_triples_by_split.items():
            json_friendly_triples[split] = [
                {"head": head, "relation": relation, "tail": tail}
                for head, relation, tail in triples
            ]
        
        # Count triples associated with each high-degree entity
        entity_triple_counts = defaultdict(int)
        for split in ['train', 'valid', 'test']:
            for head, relation, tail in filtered_triples_by_split[split]:
                if head in high_degree_entity_details:
                    entity_triple_counts[head] += 1
                if tail in high_degree_entity_details:
                    entity_triple_counts[tail] += 1
        
        # Add summary information
        triples_info = {
            "degree_threshold": degree_threshold,
            "total_filtered_triples": sum(len(triples) for triples in filtered_triples_by_split.values()),
            "filtered_triples_by_split": {
                split: len(triples) for split, triples in filtered_triples_by_split.items()
            },
            "entity_triple_counts": [
                {"entity": entity, "degree": high_degree_entity_details[entity], "triple_count": count}
                for entity, count in sorted(entity_triple_counts.items(), key=lambda x: x[1], reverse=True)
            ],
            "triples": json_friendly_triples
        }
        
        with open(triples_file_path, 'w', encoding='utf-8') as f:
            json.dump(triples_info, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Saved filtered triples information to {triples_file_path}")
        
        # 3. Generate a detailed report in a more readable format
        report_file_path = os.path.join(save_dir, f"filtered_entities_report_degree_{degree_threshold}.txt")
        
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write(f"KNOWLEDGE GRAPH FILTERING REPORT - {self.name}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Degree threshold: {degree_threshold}\n")
            f.write(f"Total entities filtered: {len(high_degree_entity_details)}\n")
            f.write(f"Total triples filtered: {sum(len(triples) for triples in filtered_triples_by_split.values())}\n\n")
            
            f.write("Filtered triples by split:\n")
            for split, triples in filtered_triples_by_split.items():
                f.write(f"  {split}: {len(triples)}\n")
            f.write("\n")
            
            f.write("TOP 20 HIGHEST DEGREE ENTITIES:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'ENTITY':<50} {'DEGREE':<10} {'TRIPLES':<10}\n")
            f.write("-"*80 + "\n")
            
            for i, (entity, degree) in enumerate(sorted_entities[:20]):
                triple_count = entity_triple_counts[entity]
                f.write(f"{entity:<50} {degree:<10} {triple_count:<10}\n")
        
        logging.info(f"Generated filtering report at {report_file_path}")

    def load(self, degree_threshold=None, save_filtered_info=False, calculate_close_relation=True, mapping_method='bert', m=10, **kwargs):
        """
        Load the corresponding data files. Depending on the dataset name, decide if format conversion is needed.
        After loading, convert all data to a unified text format, updating self.triples, self.entities, self.relations.
        Optionally, calculate the close relation mapping automatically.
        
        Args:
            degree_threshold (int, optional): If provided, filter out entities with degree higher than this threshold.
            save_filtered_info (bool): Whether to save filtered information.
            calculate_close_relation (bool): If True, automatically compute the relation close mapping after loading.
            mapping_method (str): Which method to use for computing relation mapping ('bert' or others for TF-IDF).
            m (int): Number of top similar relations to store per relation (default: 10).
            **kwargs: Additional keyword arguments passed to the mapping computation method.
        """
        logging.info(f"Loading knowledge graph: {self.name}")

        # Check if this is a subgraph (name ends with "_subgraph")
        is_subgraph = self.name.endswith("-subgraph")
        base_name = self.name.replace("-subgraph", "") if is_subgraph else self.name

        # fb15k237 requires loading a Freebase mapping
        if base_name == "fb15k237" and not is_subgraph:
            self._load_fb_mapping()

        # Depending on the dataset, call the appropriate load methods
        if base_name in ["wn18", "wn18rr", "fb15k237"]:
            self._load_entities_wn_fb("entity2id.txt")
            self._load_relations_wn_fb("relation2id.txt")
            self._load_triples_wn_fb("train2id.txt", 'train')
            self._load_triples_wn_fb("valid2id.txt", 'valid')
            self._load_triples_wn_fb("test2id.txt", 'test')
        elif base_name in ["nation", "umls"]:
            self._load_entities_nation_umls("entities.txt")
            self._load_relations_nation_umls("relations.txt")
            # The nation/umls datasets have only one triple set, put them all into 'train'
            self._load_triples_nation_umls("triples.txt", 'train')
        elif base_name == "yago3-10":
            if is_subgraph:
                self._load_yago3_10_subgraph()
            else:
                self._load_yago3_10()
        elif base_name in ["codex-small", "codex-medium", "codex-large"]:
            # Judge by the name: codex-s, codex-m, codex-l
            if "small" in base_name:
                size = "s"
            elif "medium" in base_name:
                size = "m"
            else:  # "large"
                size = "l"
            
            if is_subgraph:
                self._load_codex_subgraph(size)
            else:
                self._load_codex(size)
                if base_name in ["codex-small", "codex-medium"]:
                    self._load_codex_negatives(size)
        elif base_name == "wd-singer":
            if is_subgraph:
                self._load_wd_singer_subgraph()
            else:
                self._load_wd_singer()
        else:
            raise ValueError(f"Unsupported dataset name: {self.name}")
        
        if self.name in ["wd-singer", "codex-medium", "yago3-10"]:
            self._load_threshold_selection_triples()

        # Combine all triples for statistical use
        self.triples['total'] = (
            self.triples['train'] +
            self.triples['valid'] +
            self.triples['test']
        )

        # Convert to a unified text format: no matter the original style,
        # unify them as (head_text, relation_text, tail_text)
        self._convert_to_unified_format()
        if base_name in ["codex-small", "codex-medium"] and not is_subgraph:
            self._convert_negatives_to_unified_format()
        
        # Apply degree filtering if threshold is provided
        if degree_threshold is not None:
            self.filter_high_degree_entities(degree_threshold, save_filtered_info)

        # If calculate_close_relation flag is True, compute relation close mapping automatically.
        if calculate_close_relation:
            logging.info("Calculating close relation mapping using method: " + mapping_method)
            self.load_relation_mapping(method=mapping_method, m=m, **kwargs)

    def _sanitize(self, text):
        """
        Simple cleanup of text, replacing commas with spaces and stripping whitespace,
        to reduce risk when generating textual triples with special symbols.
        """
        return text.replace(",", " ").strip()

    def _convert_to_unified_format(self):
        """
        Depending on self.name, decide if the data needs conversion.
        Transform the raw self.triples into a unified text format, and rebuild self.entities and self.relations.
        """
        new_triples = {'train': [], 'valid': [], 'test': []}
        
        # Check if this is a subgraph and extract the base name
        is_subgraph = self.name.endswith("-subgraph")
        base_name = self.name.replace("-subgraph", "") if is_subgraph else self.name

        for split in ['train', 'valid', 'test']:
            for triple in self.triples[split]:
                if base_name in ["wn18", "wn18rr"]:
                    # Original format: (head_id, rel_id, tail_id)
                    head_id, rel_id, tail_id = triple
                    head_raw = self.entities.get(head_id, str(head_id))
                    tail_raw = self.entities.get(tail_id, str(tail_id))
                    # Parse using the WordNet offset to convert to text
                    head_text = self._parse_wn_entity(head_raw, return_all=False, include_definition=False)
                    tail_text = self._parse_wn_entity(tail_raw, return_all=False, include_definition=False)
                    relation_text = self.relations.get(rel_id, str(rel_id))
                elif base_name == "fb15k237":
                    # Original format: (head_id, rel_id, tail_id), convert entities via fb_mapping
                    head_id, rel_id, tail_id = triple
                    head_raw = self.entities.get(head_id, str(head_id))
                    tail_raw = self.entities.get(tail_id, str(tail_id))
                    # For subgraphs, we might not have the fb_mapping loaded
                    if hasattr(self, 'fb_mapping') and self.fb_mapping:
                        head_text = self.fb_mapping.get(head_raw, head_raw)
                        tail_text = self.fb_mapping.get(tail_raw, tail_raw)
                    else:
                        head_text = head_raw
                        tail_text = tail_raw
                    relation_text = self.relations.get(rel_id, str(rel_id))
                elif base_name in ["nation", "umls", "yago3-10"]:
                    # The dataset itself is already textual
                    head_text, relation_text, tail_text = triple
                elif base_name in ["codex-small", "codex-medium", "codex-large", "wd-singer"]:
                    # Here triple is (head_id, rel_id, tail_id) and all are Wikidata ID strings
                    head_id, rel_id, tail_id = triple
                    head_text = self.entities.get(head_id, head_id)
                    relation_text = self.relations.get(rel_id, rel_id)
                    tail_text = self.entities.get(tail_id, tail_id)
                else:
                    raise ValueError(f"Unsupported dataset: {self.name}")

                # Clean text by replacing commas, trimming whitespace
                head_text = self._sanitize(str(head_text))
                relation_text = self._sanitize(str(relation_text))
                tail_text = self._sanitize(str(tail_text))

                # If head and tail are the same, skip
                if head_text == tail_text:
                    continue

                new_triples[split].append((head_text, relation_text, tail_text))

        # Update to the unified-format triples
        self.triples = new_triples
        self.triples['total'] = new_triples['train'] + new_triples['valid'] + new_triples['test']

        # Rebuild entity and relation mappings from the converted triples
        entities_set = set()
        relations_set = set()
        for split in ['train', 'valid', 'test']:
            for head, relation, tail in self.triples[split]:
                entities_set.add(head)
                entities_set.add(tail)
                relations_set.add(relation)
        self.entities = {i: entity for i, entity in enumerate(sorted(entities_set))}
        self.relations = {i: relation for i, relation in enumerate(sorted(relations_set))}

    # ---------------------- Convert negative_triples to a unified format ----------------------
    def _convert_negatives_to_unified_format(self):
        """
        Convert (head_id, rel_id, tail_id) in self.negative_triples to text format (head_text, rel_text, tail_text).
        Since this applies only to CoDEx-S/M/L, the logic is consistent with the CoDEx section in _convert_to_unified_format().
        """
        new_neg_triples = {'train': [], 'valid': [], 'test': []}

        for split in ['train', 'valid', 'test']:
            for triple in self.negative_triples[split]:
                # For CoDEx, the triple is also in the form of (head_id, rel_id, tail_id)
                head_id, rel_id, tail_id = triple
                head_text = self.orig_entities.get(head_id, head_id)
                relation_text = self.orig_relations.get(rel_id, rel_id)
                tail_text = self.orig_entities.get(tail_id, tail_id)

                head_text = self._sanitize(str(head_text))
                relation_text = self._sanitize(str(relation_text))
                tail_text = self._sanitize(str(tail_text))

                # If head and tail are the same, it can be ignored; modify based on whether you want to keep this check
                if head_text == tail_text:
                    continue

                new_neg_triples[split].append((head_text, relation_text, tail_text))

        # Update negative_triples
        self.negative_triples = {
            'train': new_neg_triples['train'],
            'valid': new_neg_triples['valid'],
            'test': new_neg_triples['test'],
            'total': new_neg_triples['train'] + new_neg_triples['valid'] + new_neg_triples['test']
        }

    # ------------------ Load threshold selection triples ------------------
    def _load_threshold_selection_triples(self):
        """
        Loads the triple data required for threshold selection, including a selected subset of relations,
        positive samples, and negative samples. The data is sourced from the `selected_relations.json`
        file in the corresponding dataset.

        Applicable to wd-singer, codex-medium, and yago3-10 datasets.
        """
        logging.info(f"Loading threshold selection triples for {self.name}...")

        # Construct the JSON file path
        json_path = os.path.join(self.data_path, self.name, f"{self.name}_selected_relations.json")

        if not os.path.exists(json_path):
            logging.warning(f"Threshold selection data file not found: {json_path}")
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load selected relations
            self.threshold_selection_triples['relation'] = data.get('relations', [])

            # Load positive triples and convert lists to tuples
            positive_lists = data.get('positive', [])
            self.threshold_selection_triples['positive'] = [tuple(triple) for triple in positive_lists]

            # Load negative triples (Note: In the yago3-10 dataset, the key might be misspelled as "negtive")
            if 'negative' in data:
                negative_lists = data.get('negative', [])
                self.threshold_selection_triples['negative'] = [tuple(triple) for triple in negative_lists]
            elif 'negtive' in data:  # Handle the misspelling in yago3-10
                negative_lists = data.get('negtive', [])
                self.threshold_selection_triples['negative'] = [tuple(triple) for triple in negative_lists]

            logging.info(f"Loaded {len(self.threshold_selection_triples['relation'])} relations, "
                        f"{len(self.threshold_selection_triples['positive'])} positive triples, "
                        f"{len(self.threshold_selection_triples['negative'])} negative triples "
                        f"for threshold selection.")

        except Exception as e:
            logging.error(f"Error occurred while loading threshold selection data: {e}")


    # ------------------ Load CoDEx negatives ------------------
    def _load_codex_negatives(self, size):
        """
        Load CoDEx (codex-s or codex-m) valid_negatives.txt and test_negatives.txt.
        Since train_negatives.txt is currently unavailable, self.negative_triples['train'] remains empty.

        Args:
            size (str): "s" or "m".
        """
        logging.info(f"Loading negative triples for CoDEx-{size} (if available) ...")
        codex_base = os.path.join(self.data_path, "codex")
        triple_dir = os.path.join(codex_base, "triples", f"codex-{size}")

        # Load valid_negatives.txt
        valid_neg_file = os.path.join(triple_dir, "valid_negatives.txt")
        if os.path.exists(valid_neg_file):
            with open(valid_neg_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 3:
                        continue
                    head_id, rel_id, tail_id = parts
                    self.negative_triples['valid'].append((head_id, rel_id, tail_id))

        # Load test_negatives.txt
        test_neg_file = os.path.join(triple_dir, "test_negatives.txt")
        if os.path.exists(test_neg_file):
            with open(test_neg_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 3:
                        continue
                    head_id, rel_id, tail_id = parts
                    self.negative_triples['test'].append((head_id, rel_id, tail_id))

        # For other cases (train or missing files), keep them empty.
        # Note: self.negative_triples['total'] is not updated here; it will be updated after the unified conversion.

    # ------------------ Load WD-singer ------------------
    def _load_wd_singer(self):
        """
        Load the WD-singer dataset from local files: 'train.triples', 'dev.triples', 'test.triples'.
        Each line is 'Qxxx Qyyy Pzzz', meaning (head_entity, tail_entity, relation).
        If a wikidata_labels.json file exists, use it directly, otherwise fetch labels from Wikidata API
        and save them to this file for future use.
        """
        logging.info("Loading WD-singer dataset...")

        wd_singer_dir = os.path.join(self.data_path, "wd-singer")
        file_map = {
            'train': "train.triples",
            'valid': "dev.triples",  # 'dev' => 'valid'
            'test':  "test.triples"
        }
        
        # Path to the mapping cache file
        labels_cache_file = os.path.join(wd_singer_dir, "wikidata_labels.json")
        
        # Check if mapping file already exists
        if os.path.exists(labels_cache_file):
            logging.info(f"Found existing Wikidata labels cache at {labels_cache_file}")
            with open(labels_cache_file, 'r', encoding='utf-8') as f:
                # Load mapping directly from file
                global _wd_label_cache
                _wd_label_cache = json.load(f)
        else:
            # 1) Gather all QIDs/PIDs in a set
            all_ids = set()
            for split, filename in file_map.items():
                filepath = os.path.join(wd_singer_dir, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"File not found for wd-singer {split}: {filepath}")

                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            head_id, tail_id, rel_id = parts
                            all_ids.update([head_id, tail_id, rel_id])

            # 2) Use tqdm to fetch or skip if in cache
            logging.info(f"Fetching Wikidata labels for {len(all_ids)} unique IDs...")
            for qid in tqdm(all_ids, desc="Wikidata label fetch"):
                _ = fetch_wikidata_label(qid, lang='en')  # This will populate the cache
                
            # Save the cache to file for future use
            logging.info(f"Saving Wikidata labels to {labels_cache_file}")
            with open(labels_cache_file, 'w', encoding='utf-8') as f:
                json.dump(_wd_label_cache, f, ensure_ascii=False, indent=2)

        # 3) Now parse each file line and build raw triples
        lines_map = {}
        for split, filename in file_map.items():
            filepath = os.path.join(wd_singer_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                lines_map[split] = f.readlines()
        
        # Process all files using the cached labels
        for split, lines in lines_map.items():
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                head_id, tail_id, rel_id = parts

                # Now that we have them in cache, store them in self.entities / self.relations
                head_label = _wd_label_cache.get(head_id, head_id)
                tail_label = _wd_label_cache.get(tail_id, tail_id)
                rel_label = _wd_label_cache.get(rel_id, rel_id)

                self.entities[head_id] = head_label
                self.entities[tail_id] = tail_label
                self.relations[rel_id] = rel_label

                # Add the raw triple (head_id, rel_id, tail_id)
                self.triples[split].append((head_id, rel_id, tail_id))

    # ------------------ Load CoDEx ------------------
    def _load_codex(self, size):
        """
        Load the CoDEx dataset (codex-s, codex-m, codex-l). The dataset includes:
        - entities/en/entities.json: { <Wikidata entity ID>: {"label":..., "description":..., "wiki":...}, ... }
        - relations/en/relations.json: { <Wikidata relation ID>: {"label":..., "description":...}, ... }
        - triples/codex-s|codex-m|codex-l contain train.txt, valid.txt, test.txt, where each line follows: head_id \t rel_id \t tail_id

        Args:
            size (str): "s", "m", or "l", corresponding to codex-s, codex-m, and codex-l
        """
        logging.info(f"Loading CoDEx-{size} dataset...")

        codex_base = os.path.join(self.data_path, "codex")

        # 1) Load entities
        entities_file = os.path.join(codex_base, "entities", "en", "entities.json")
        if not os.path.exists(entities_file):
            raise FileNotFoundError(f"CoDEx entities file not found: {entities_file}")
        with open(entities_file, "r", encoding="utf-8") as f:
            entities_data = json.load(f)
            # entities_data: { <entity_id>: {"label": ..., "description": ..., "wiki": ...}, ... }
            for ent_id, ent_info in entities_data.items():
                label = ent_info.get("label", "").strip()
                # If no label is available, fall back to using the entity ID
                if not label:
                    label = ent_id
                self.entities[ent_id] = label

        # 2) Load relations
        relations_file = os.path.join(codex_base, "relations", "en", "relations.json")
        if not os.path.exists(relations_file):
            raise FileNotFoundError(f"CoDEx relations file not found: {relations_file}")
        with open(relations_file, "r", encoding="utf-8") as f:
            relations_data = json.load(f)
            # relations_data: { <rel_id>: {"label": ..., "description": ...}, ... }
            for rel_id, rel_info in relations_data.items():
                label = rel_info.get("label", "").strip()
                if not label:
                    label = rel_id
                self.relations[rel_id] = label

        # 3) Load triples
        triple_dir = os.path.join(codex_base, "triples", f"codex-{size}")
        for split in ["train", "valid", "test"]:
            triple_file = os.path.join(triple_dir, f"{split}.txt")
            if not os.path.exists(triple_file):
                raise FileNotFoundError(f"CoDEx {split}.txt not found: {triple_file}")
            with open(triple_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 3:
                        continue
                    head_id, rel_id, tail_id = parts
                    # Store triples as (head_id, rel_id, tail_id) format for unified conversion later
                    self.triples[split].append((head_id, rel_id, tail_id))
        
        # Store original entities and relations for later use
        self.orig_entities = dict(self.entities)
        self.orig_relations = dict(self.relations)


    # ------------------ Load YAGO3-10 -------------------
    def _load_yago3_10(self):
        """
        Load the YAGO3-10 dataset from Hugging Face. The raw triples are already in text format.
        """
        logging.info("Loading YAGO3-10 dataset from Hugging Face...")
        ds = load_dataset("VLyb/YAGO3-10")

        mapping = {
            "train": "train",
            "validation": "valid",  # rename "validation" to "valid"
            "test": "test",
        }

        all_entities = set()
        all_relations = set()

        for split, mapped_split in mapping.items():
            if split in ds:
                for row in ds[split]:
                    head, relation, tail = row["head"], row["relation"], row["tail"]
                    self.triples[mapped_split].append((head, relation, tail))
                    all_entities.add(head)
                    all_entities.add(tail)
                    all_relations.add(relation)

        # Build initial entity/relation dicts (will be rebuilt in the unified conversion anyway)
        self.entities = {i: entity for i, entity in enumerate(sorted(all_entities))}
        self.relations = {i: relation for i, relation in enumerate(sorted(all_relations))}

    def _load_yago3_10_subgraph(self):
        """
        Load the YAGO3-10 subgraph from JSON files saved by _save_subgraph method.
        """
        
        logging.info(f"Loading YAGO3-10 subgraph from {self.data_path}/{self.name}")
        
        all_entities = set()
        all_relations = set()
        
        # Map from original dataset name to our expected internal splits
        mapping = {
            "train.json": "train",
            "valid.json": "valid",
            "test.json": "test"
        }
        
        for file_name, split in mapping.items():
            file_path = os.path.join(self.data_path, self.name, file_name)
            
            if not os.path.exists(file_path):
                logging.warning(f"File not found: {file_path}")
                continue
                
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                for item in data:
                    head = item["head"]
                    relation = item["relation"]
                    tail = item["tail"]
                    
                    self.triples[split].append((head, relation, tail))
                    all_entities.add(head)
                    all_entities.add(tail)
                    all_relations.add(relation)
                    
            except Exception as e:
                logging.error(f"Error loading {file_path}: {str(e)}")
        
        # Build entity and relation dictionaries
        self.entities = {i: entity for i, entity in enumerate(sorted(all_entities))}
        self.relations = {i: relation for i, relation in enumerate(sorted(all_relations))}

    def _load_codex_subgraph(self, size):
        """
        Load the CoDEx subgraph saved by _save_subgraph method.
        
        Args:
            size (str): "s", "m", or "l" for codex-small, codex-medium, or codex-large.
        """
        
        logging.info(f"Loading CoDEx-{size} subgraph from {self.data_path}/{self.name}")
        
        # 1) Load entities
        entities_path = os.path.join(self.data_path, self.name, "entities", "en", "entities.json")
        if not os.path.exists(entities_path):
            raise FileNotFoundError(f"CoDEx subgraph entities file not found: {entities_path}")
            
        with open(entities_path, "r", encoding="utf-8") as f:
            entities_data = json.load(f)
            for ent_id, ent_info in entities_data.items():
                label = ent_info.get("label", "").strip()
                if not label:
                    label = ent_id
                self.entities[ent_id] = label
        
        # 2) Load relations
        relations_path = os.path.join(self.data_path, self.name, "relations", "en", "relations.json")
        if not os.path.exists(relations_path):
            raise FileNotFoundError(f"CoDEx subgraph relations file not found: {relations_path}")
            
        with open(relations_path, "r", encoding="utf-8") as f:
            relations_data = json.load(f)
            for rel_id, rel_info in relations_data.items():
                label = rel_info.get("label", "").strip()
                if not label:
                    label = rel_id
                self.relations[rel_id] = label
        
        # 3) Load triples
        original_name = self.name.replace("_subgraph", "")
        triples_dir = os.path.join(self.data_path, self.name, "triples", f"{original_name}_subgraph")
        
        if not os.path.exists(triples_dir):
            raise FileNotFoundError(f"CoDEx subgraph triples directory not found: {triples_dir}")
        
        for split in ["train", "valid", "test"]:
            triple_file = os.path.join(triples_dir, f"{split}.txt")
            if not os.path.exists(triple_file):
                logging.warning(f"CoDEx subgraph {split}.txt not found: {triple_file}")
                continue
                
            with open(triple_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 3:
                        continue
                    head_id, rel_id, tail_id = parts
                    self.triples[split].append((head_id, rel_id, tail_id))
        
        # Store original entities and relations for reference
        self.orig_entities = dict(self.entities)
        self.orig_relations = dict(self.relations)

    def _load_wd_singer_subgraph(self):
        """
        Load the WD-Singer subgraph saved by _save_subgraph method.
        """
        
        logging.info(f"Loading WD-Singer subgraph from {self.data_path}/{self.name}")
        
        # WD-Singer uses different file names for train/valid/test
        file_map = {
            'train': "train.triples",
            'valid': "dev.triples",  # 'dev' => 'valid'
            'test':  "test.triples"
        }
        
        entities_set = set()
        relations_set = set()
        
        # Process each file
        for split, filename in file_map.items():
            file_path = os.path.join(self.data_path, self.name, filename)
            if not os.path.exists(file_path):
                logging.warning(f"File not found for WD-Singer subgraph {split}: {file_path}")
                continue
                
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 3:
                        continue
                        
                    head_id, tail_id, rel_id = parts
                    
                    # Add the raw triple (head_id, rel_id, tail_id)
                    self.triples[split].append((head_id, rel_id, tail_id))
                    
                    # Collect entity and relation IDs
                    entities_set.add(head_id)
                    entities_set.add(tail_id)
                    relations_set.add(rel_id)
        
        # For subgraphs, we use the entity/relation IDs as labels
        # This is simplified from the original _load_wd_singer method that fetches labels from Wikidata
        for entity_id in entities_set:
            self.entities[entity_id] = entity_id
            
        for relation_id in relations_set:
            self.relations[relation_id] = relation_id

    # ------------------ Load wn18, wn18rr, fb15k237 -------------------
    def _load_entities_wn_fb(self, filename):
        """
        Load entity2id.txt for wn18, wn18rr, fb15k237 (skip the first line).
        """
        file_path = os.path.join(self.data_path, self.name, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            next(f)  # skip the first line
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    logging.warning(f"Skipping malformed line: {line.strip()}")
                    continue
                entity, eid = parts
                self.entities[int(eid)] = entity

    def _load_relations_wn_fb(self, filename):
        """
        Load relation2id.txt for wn18, wn18rr, fb15k237 (skip the first line).
        """
        file_path = os.path.join(self.data_path, self.name, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    logging.warning(f"Skipping malformed line: {line.strip()}")
                    continue
                relation, rid = parts
                self.relations[int(rid)] = relation

    def _load_triples_wn_fb(self, filename, split):
        """
        Load triples for wn18, wn18rr, fb15k237 in the format "head_id tail_id relation_id" (skip first line).
        """
        file_path = os.path.join(self.data_path, self.name, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                head_id, tail_id, rel_id = map(int, parts)
                self.triples[split].append((head_id, rel_id, tail_id))

    # ------------------ Load NATION, UMLS -------------------
    def _load_entities_nation_umls(self, filename):
        """
        Load entities.txt for nation or umls, format "index entity_text".
        """
        file_path = os.path.join(self.data_path, self.name, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    logging.warning(f"Skipping malformed line: {line.strip()}")
                    continue
                idx_str, entity_text = parts
                try:
                    idx_int = int(idx_str)
                except ValueError:
                    logging.warning(f"Skipping line with invalid index: {line.strip()}")
                    continue
                self.entities[idx_int] = entity_text

    def _load_relations_nation_umls(self, filename):
        """
        Load relations.txt for nation or umls, format "index relation_text".
        """
        file_path = os.path.join(self.data_path, self.name, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    logging.warning(f"Skipping malformed line: {line.strip()}")
                    continue
                idx_str, relation_text = parts
                try:
                    idx_int = int(idx_str)
                except ValueError:
                    logging.warning(f"Skipping line with invalid index: {line.strip()}")
                    continue
                self.relations[idx_int] = relation_text

    def _load_triples_nation_umls(self, filename, split):
        """
        Load triples for nation or umls in the format "relation_id entity_id1 entity_id2",
        storing them as (entity_id1, relation_id, entity_id2).
        """
        file_path = os.path.join(self.data_path, self.name, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    logging.warning(f"Skipping malformed triple line: {line.strip()}")
                    continue
                try:
                    rel_id, ent1_id, ent2_id = map(int, parts)
                except ValueError:
                    logging.warning(f"Skipping invalid triple line: {line.strip()}")
                    continue
                # self.triples[split].append((ent1_id, rel_id, ent2_id))
                
                head_text = self.entities.get(ent1_id, str(ent1_id))
                relation_text = self.relations.get(rel_id, str(rel_id))
                tail_text = self.entities.get(ent2_id, str(ent2_id))

                # Store textual triples so we don't see numeric IDs later:
                self.triples[split].append((head_text, relation_text, tail_text))

    # ------------------ Freebase Mapping for fb15k237 -------------------
    def _load_fb_mapping(self):
        """
        Load the Freebase MID-to-name mapping (file: fb_wiki_mapping.tsv) for fb15k237.
        """
        fb_mapping_file = os.path.join(self.data_path, self.name, "fb_wiki_mapping.tsv")
        if not os.path.exists(fb_mapping_file):
            raise FileNotFoundError(f"Mapping file not found: {fb_mapping_file}")

        with open(fb_mapping_file, "r", encoding="utf-8") as f:
            next(f)  # skip the header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    freebase_id, _, label = parts
                    self.fb_mapping[freebase_id] = label

    # ------------------ WordNet helper functions -------------------
    def offset_to_word(self, offset, return_all=False, include_definition=False):
        """
        Convert a WordNet offset to the corresponding lemma(s), optionally returning all candidates or including definitions.
        (Used only for WN-based datasets)
        """
        fallback_pos = ['n', 'v', 'a', 'r']
        lemmas = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p in fallback_pos:
                try:
                    synset = wn.synset_from_pos_and_offset(p, offset)
                    lemma_names = [lemma.name() for lemma in synset.lemmas()]
                    if include_definition:
                        definition = synset.definition()
                        lemma_names = [f"{lemma} (def: {definition})" for lemma in lemma_names]
                    lemmas.extend(lemma_names)
                except:
                    pass
        if not lemmas:
            return None
        return lemmas if return_all else lemmas[0]

    def _parse_wn_entity(self, entity_str, return_all=False, include_definition=False):
        """
        For WN-based datasets, parse the entity offset into the corresponding lemma text.
        """
        if '-' in entity_str:
            offset_str, _ = entity_str.rsplit('-', 1)
        else:
            offset_str = entity_str
        try:
            offset_int = int(offset_str)
        except ValueError:
            return entity_str
        word = self.offset_to_word(offset_int, return_all, include_definition)
        return word if word else entity_str

    # ------------------ Statistics & text triple generation -------------------
    def compute_statistics(self):
        """
        Compute knowledge graph statistics, including the number of entities, relations,
        and triple counts for each split.
        """
        stats = {
            'total_entities': len(self.entities),
            'total_relations': len(self.relations),
            'total_triples': {split: len(triples) for split, triples in self.triples.items()},
            'entity_frequency': Counter([head for split in self.triples.values() for head, _, _ in split]),
            'relation_frequency': Counter([relation for split in self.triples.values() for _, relation, _ in split])
        }
        return stats

    def generate_text_triples(self, template="{head},{relation},{tail}"):
        """
        Given the unified text-format triples, generate formatted text triples.

        Args:
            template (str): A format template like "{head} {relation} {tail}."

        Returns:
            List[str]: A list of formatted text triples.
        """
        text_triples = []
        for split in ['train', 'valid', 'test']:
            for head, relation, tail in self.triples[split]:
                text_triples.append(template.format(head=head, relation=relation, tail=tail))
        return text_triples

    # generate_full_text_triples and generate_text_triples are identical under the unified format
    def generate_full_text_triples(self, template="{head},{relation},{tail}"):
        """
        Generate full text triples (entities and relations in unified text format).

        Args:
            template (str): The format template.

        Returns:
            List[str]: A list of formatted text triples.
        """
        return self.generate_text_triples(template=template)

    def build_adjacency_list(self):
        """
        Build an undirected adjacency list from the unified text triples
        for quick lookup of entity neighbors.
        """
        self.adj_list = defaultdict(set)
        full_text_triples = self.generate_full_text_triples()
        for triple in full_text_triples:
            parts = triple.strip().split(",")
            if len(parts) != 3:
                continue
            head_text, relation_text, tail_text = parts
            self.adj_list[head_text].add((relation_text, tail_text))
            self.adj_list[tail_text].add((relation_text, head_text))

    def get_k_hop_neighbors(self, entity, k=2, sample=False, max_sample=200):
        """
        Retrieve the k-hop neighbors of the specified entity from the unified adjacency list.

        Args:
            entity (str): The target entity.
            k (int): The number of hops (default 2).
            sample (bool): Whether to sample neighbors to limit quantity.
            max_sample (int): If sampling, the max neighbors to take.

        Returns:
            set: The set of k-hop neighbors.
        """
        if self.adj_list is None:
            self.build_adjacency_list()
        neighbors = set()
        queue = deque([(entity, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= k:
                continue
            neighbor_list = list(self.adj_list.get(node, []))
            if sample and len(neighbor_list) > max_sample:
                neighbor_list = random.sample(neighbor_list, max_sample)
            for relation, neigh in neighbor_list:
                if neigh not in neighbors:
                    neighbors.add(neigh)
                    queue.append((neigh, depth + 1))
        return neighbors

    def save_statistics(self, stats, file_path):
        """
        Save computed statistics to a JSON file.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)

    def print_statistics(self):
        """
        Print basic knowledge graph statistics.
        """
        logging.info(f"Knowledge Graph: {self.name}")
        logging.info(f"Total triples: {len(self.triples['total'])}")
        logging.info(f"Total unique entities: {len(self.entities)}")
        logging.info(f"Total unique relations: {len(self.relations)}")

    def relation_graph(self, methods=["co-occurrence", "transitivity", "embedding", "semantic"], 
                    weights=None, normalize=True, max_samples=10000, bert_model="distilbert-base-uncased", semantic_method="bert"):
        
            """
            Construct a Relation-of-Relations Graph, where:
            - Each node represents a type of relation.
            - Each edge represents the similarity or inference strength between two relations.

            This function creates a relation graph matrix W, where W[i,j] indicates the correlation between relation r_i and r_j.

            Parameters:
                methods (list): Methods for calculating relation similarity, including:
                    - "co-occurrence": If (h, r_i, t) and (h, r_j, t) frequently occur together.
                    - "transitivity": If (h, r_i, m) and (m, r_j, t) frequently form a path.
                    - "embedding": Compute embedding similarity based on relation connection patterns.
                    - "semantic": Use NLP methods to compare the textual meanings of the relations.
                weights (dict): The weight for each method, e.g., {"co-occurrence": 0.6, "transitivity": 0.6}.
                    If None, equal weights will be assigned to all methods.
                normalize (bool): Whether to normalize the weights to the [0,1] range.
                max_samples (int): Maximum number of samples to consider for efficiency.

            Returns:
                numpy.ndarray: The relation graph matrix W, where W[i,j] is the correlation between relation i and relation j.
                dict: A mapping from relation indices to relation names.
            """

        
        
            # Get all relations
            relations = list(set(self.relations.values()))
            n_relations = len(relations)
            
            logging.info(f"Building relation graph with {n_relations} relations")
            
            # Initialize adjacency matrix
            W = np.zeros((n_relations, n_relations))
            
            # Create mapping from relations to indices
            relation_to_idx = {relation: i for i, relation in enumerate(relations)}
            idx_to_relation = {i: relation for i, relation in enumerate(relations)}
            
            # If weights are not specified, set equal weights
            if weights is None:
                weights = {method: 1.0 / len(methods) for method in methods}
            
            # Method 1: Co-occurrence analysis
            if "co-occurrence" in methods:
                logging.info("Calculating co-occurrence similarity")
                co_occurrence = np.zeros((n_relations, n_relations))
                
                # Create mapping from (head, tail) pairs to list of relations
                entity_pair_to_relations = defaultdict(list)
                for split in ['train', 'valid', 'test']:
                    for head, rel, tail in self.triples[split]:
                        entity_pair_to_relations[(head, tail)].append(rel)
                
                # Limit the number of entity pairs for efficiency
                entity_pairs = list(entity_pair_to_relations.keys())
                if max_samples and len(entity_pairs) > max_samples:
                    entity_pairs = random.sample(entity_pairs, max_samples)
                
                # Calculate co-occurrence counts
                for pair in entity_pairs:
                    rels = entity_pair_to_relations[pair]
                    for rel_i in rels:
                        for rel_j in rels:
                            if rel_i != rel_j:
                                i = relation_to_idx[rel_i]
                                j = relation_to_idx[rel_j]
                                co_occurrence[i, j] += 1
                
                # Normalize co-occurrence matrix
                if normalize and np.max(co_occurrence) > 0:
                    co_occurrence = co_occurrence / np.max(co_occurrence)
                
                W += weights.get("co-occurrence", 0.25) * co_occurrence
            
            # Method 2: Transitivity analysis
            if "transitivity" in methods:
                logging.info("Calculating transitivity similarity")
                transitivity = np.zeros((n_relations, n_relations))
                
                # Build entity-relation graph
                entity_out_graph = defaultdict(list)  # head -> [(relation, tail), ...]
                entity_in_graph = defaultdict(list)   # tail -> [(relation, head), ...]
                head_tail_relations = defaultdict(list)  # (head, tail) -> [relation]
                
                for split in ['train', 'valid', 'test']:
                    for head, rel, tail in self.triples[split]:
                        entity_out_graph[head].append((rel, tail))
                        entity_in_graph[tail].append((rel, head))
                        head_tail_relations[(head, tail)].append(rel)
                
                # Sample entities for efficiency
                entities = list(set(entity_out_graph.keys()) | set(entity_in_graph.keys()))
                if max_samples and len(entities) > max_samples:
                    entities = random.sample(entities, max_samples)
                
                # Find two-step paths and potential third relations
                for entity in entities:
                    if entity in entity_out_graph:
                        for rel1, mid in entity_out_graph[entity]:
                            if mid in entity_out_graph:
                                for rel2, tail in entity_out_graph[mid]:
                                    # Find two-step path: entity --(rel1)--> mid --(rel2)--> tail
                                    if rel1 != rel2:
                                        i = relation_to_idx[rel1]
                                        j = relation_to_idx[rel2]
                                        
                                        # If direct relation exists, increase transitivity weight
                                        if (entity, tail) in head_tail_relations:
                                            weight_mult = len(head_tail_relations[(entity, tail)])
                                            transitivity[i, j] += weight_mult
                                        else:
                                            transitivity[i, j] += 1
                
                # Normalize transitivity matrix
                if normalize and np.max(transitivity) > 0:
                    transitivity = transitivity / np.max(transitivity)
                
                W += weights.get("transitivity", 0.25) * transitivity
            
            # Method 3: Embedding similarity
            if "embedding" in methods:
                logging.info("Calculating embedding similarity")
                try:

                    
                    # Collect all entities
                    entities = set()
                    for split in ['train', 'valid', 'test']:
                        for h, _, t in self.triples[split]:
                            entities.add(h)
                            entities.add(t)
                    
                    entities = list(entities)
                    
                    # Sample entities for efficiency
                    if max_samples and len(entities) > max_samples * 2:
                        entities = random.sample(entities, max_samples * 2)
                    
                    entity_to_idx = {e: i for i, e in enumerate(entities)}
                    n_entities = len(entities)
                    
                    # Create embeddings for each relation based on usage patterns
                    # Each embedding encodes how the relation connects entities
                    rel_embeddings = np.zeros((n_relations, n_entities * 2))
                    
                    for split in ['train', 'valid', 'test']:
                        for head, rel, tail in self.triples[split]:
                            if rel in relation_to_idx and head in entity_to_idx and tail in entity_to_idx:
                                rel_idx = relation_to_idx[rel]
                                head_idx = entity_to_idx[head]
                                tail_idx = entity_to_idx[tail]
                                # First half of embedding: as head entity, second half: as tail entity
                                rel_embeddings[rel_idx, head_idx] += 1
                                rel_embeddings[rel_idx, n_entities + tail_idx] += 1
                    
                    # Calculate cosine similarity
                    embedding_similarity = cosine_similarity(rel_embeddings)
                    
                    # Scale from [-1, 1] to [0, 1]
                    if normalize:
                        embedding_similarity = (embedding_similarity + 1) / 2
                    
                    W += weights.get("embedding", 0.25) * embedding_similarity
                    
                except ImportError:
                    logging.warning("scikit-learn not available, skipping embedding similarity calculation")
                except Exception as e:
                    logging.error(f"Error calculating embedding similarity: {e}")
            

            if "semantic" in methods:            
                # Process relation names for text processing
                relation_texts = []
                for relation in relations:
                    # Convert camel case or snake case to spaces
                    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', relation)  # Camel case -> spaces
                    text = text.replace('_', ' ').replace('/', ' ')       # Snake case -> spaces
                    relation_texts.append(text.lower())
                
                # Calculate semantic similarity based on specified method
                semantic_similarity = None
                
                if semantic_method == "bert":
                    # Use BERT to calculate semantic similarity
                    logging.info("Calculating semantic similarity using BERT")
                    try:
                        # Function: Mean pooling for sentence representations
                        def mean_pooling(model_output, attention_mask):
                            token_embeddings = model_output[0]
                            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        
                        # Load pre-trained model and tokenizer
                        tokenizer = AutoTokenizer.from_pretrained(bert_model)
                        model = AutoModel.from_pretrained(bert_model)
                        
                        # Convert relation names to embedding vectors
                        embeddings = []
                        batch_size = 32
                        
                        # Process in batches for efficiency
                        for i in range(0, len(relation_texts), batch_size):
                            batch_texts = relation_texts[i:i+batch_size]
                            
                            # Tokenize
                            encoded_input = tokenizer(
                                batch_texts, 
                                padding=True, 
                                truncation=True, 
                                max_length=128, 
                                return_tensors='pt'
                            )
                            
                            # Calculate token embeddings
                            with torch.no_grad():
                                model_output = model(**encoded_input)
                            
                            # Mean pooling of token embeddings
                            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                            
                            # Normalize embeddings
                            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                            
                            # Add embeddings to list
                            embeddings.append(sentence_embeddings)
                        
                        # Concatenate all embeddings into a single tensor
                        all_embeddings = torch.cat(embeddings, dim=0).numpy()
                        
                        # Calculate cosine similarity
                        semantic_similarity = cosine_similarity(all_embeddings)
                        
                        if normalize:
                            semantic_similarity = (semantic_similarity + 1) / 2
                        
                    except ImportError as e:
                        logging.warning(f"Required libraries for BERT not available: {e}")
                        logging.warning("Falling back to TF-IDF for semantic similarity calculation")
                        semantic_method = "tfidf"  # Fallback to TF-IDF
                    except Exception as e:
                        logging.error(f"Error calculating semantic similarity with BERT: {e}")
                        logging.warning("Falling back to TF-IDF for semantic similarity calculation")
                        semantic_method = "tfidf"  # Fallback to TF-IDF
                
                if semantic_method == "word2vec":
                    # Use Word2Vec to calculate semantic similarity
                    logging.info("Calculating semantic similarity using Word2Vec")
                    try:
                        def get_word_vectors(text, word_vectors, vector_size=300):
                            words = text.split()
                            vectors = []
                            for word in words:
                                if word in word_vectors:
                                    vectors.append(word_vectors[word])
                            
                            if vectors:
                                # Return the average of all word vectors
                                return np.mean(vectors, axis=0)
                            else:
                                # If no word vectors are found, return a zero vector
                                return np.zeros(vector_size)
                        
                        # Try to load pre-trained Word2Vec model
                        # First check if the model is already downloaded
                        word2vec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "word2vec_model.bin")
                        
                        if not os.path.exists(word2vec_path):
                            # If the model file does not exist, try to download it, but it may be large
                            logging.warning("Pre-trained Word2Vec model not found. This may require downloading a large file.")
                            logging.warning("Using simple word vector representation")

                            # Extract all unique words from relation texts
                            all_words = set()
                            for text in relation_texts:
                                all_words.update(text.split())
                            
                            # Assign a random vector to each word
                            np.random.seed(42)  # Set random seed for reproducibility
                            simple_word_vectors = {}
                            vector_size = 100
                            
                            for word in all_words:
                                simple_word_vectors[word] = np.random.randn(vector_size)
                            
                            # Calculate vector representation for each relation text
                            relation_vectors = []
                            for text in relation_texts:
                                relation_vectors.append(get_word_vectors(text, simple_word_vectors, vector_size))
                            
                            # Calculate cosine similarity
                            relation_vectors = np.array(relation_vectors)
                            semantic_similarity = cosine_similarity(relation_vectors)
                            
                        else:
                            # Load pre-trained model
                            word_vectors = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
                            vector_size = word_vectors.vector_size
                            
                            # Calculate vector representation for each relation text
                            relation_vectors = []
                            for text in relation_texts:
                                relation_vectors.append(get_word_vectors(text, word_vectors, vector_size))
                            
                            # Calculate cosine similarity
                            relation_vectors = np.array(relation_vectors)
                            semantic_similarity = cosine_similarity(relation_vectors)
                        
                        if normalize:
                            semantic_similarity = (semantic_similarity + 1) / 2
                        
                    except ImportError as e:
                        logging.warning(f"Required libraries for Word2Vec not available: {e}")
                        logging.warning("Falling back to TF-IDF for semantic similarity calculation")
                        semantic_method = "tfidf"  # Fallback to TF-IDF
                    except Exception as e:
                        logging.error(f"Error calculating semantic similarity with Word2Vec: {e}")
                        logging.warning("Falling back to TF-IDF for semantic similarity calculation")
                        semantic_method = "tfidf"  # Fallback to TF-IDF
                
                if semantic_method == "tfidf" or semantic_similarity is None:
                    # Use TF-IDF to calculate semantic similarity
                    logging.info("Calculating semantic similarity using TF-IDF")
                    try:
                        vectorizer = TfidfVectorizer()
                        tfidf_matrix = vectorizer.fit_transform(relation_texts)
                        semantic_similarity = cosine_similarity(tfidf_matrix)
                        
                    except ImportError:
                        logging.warning("scikit-learn not available, cannot calculate semantic similarity")
                        semantic_similarity = None
                    except Exception as e:
                        logging.error(f"Error calculating semantic similarity with TF-IDF: {e}")
                        semantic_similarity = None

                if semantic_similarity is not None:
                    W += weights.get("semantic", 0.25) * semantic_similarity
                else:
                    logging.warning("Semantic similarity cannot be calculated, skip this method")

            if normalize and np.max(W) > 0:
                W = W / np.max(W)
            
            return W, idx_to_relation

    def relation_graph_visualization(self, relation_matrix=None, relation_mapping=None, 
                               output_path="relation_graph.png", 
                               weight_threshold=0.1, 
                               max_relations=50,
                               figsize=(14, 10),
                               node_size_factor=500,
                               edge_scale_factor=3.0,
                               layout="spring",
                               colormap="viridis",
                               show_labels=True,
                               title="Knowledge Graph Relation Network",
                               customize_fn=None):
        """
            Visualize the Relation Graph and save it as an image.

            Parameters:
                relation_matrix (numpy.ndarray): The relation graph matrix. If None, self.relation_graph will be called to obtain it.
                relation_mapping (dict): Mapping from indices to relation names. If None, self.relation_graph will be called to obtain it.
                output_path (str): Path to save the image.
                weight_threshold (float): Only display edges with weights above this threshold.
                max_relations (int): Maximum number of relations to display, to prevent the graph from becoming overly complex.
                figsize (tuple): The size of the matplotlib figure.
                node_size_factor (float): A factor to control the size of the nodes.
                edge_scale_factor (float): A factor to control the width of the edges.
                layout (str): Graph layout algorithm, options include: "spring", "circular", "random", "spectral", "shell".
                colormap (str): The name of the matplotlib colormap to use.
                show_labels (bool): Whether to display relation name labels.
                title (str): The title of the graph.
                customize_fn (callable): A custom function that takes (G, pos, ax) as parameters for additional customization.

            Returns:
                str: The path where the image is saved.
        """
       
        
        try:
            if relation_matrix is None or relation_mapping is None:
                logging.info("The relation matrix is not provided, the relation_graph method is automatically invoked")
                relation_matrix, relation_mapping = self.relation_graph()

            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
  
            G = nx.Graph()
            
            n_relations = relation_matrix.shape[0]
            if n_relations > max_relations:
                logging.info(f"The number of relations({n_relations}) exceed the maximum number of displays ({max_relations}), choose the relations that matter most")

                relation_importance = np.sum(relation_matrix, axis=1)
                top_indices = np.argsort(relation_importance)[-max_relations:]

                filtered_matrix = relation_matrix[np.ix_(top_indices, top_indices)]
                filtered_mapping = {i: relation_mapping[idx] for i, idx in enumerate(top_indices)}
                
                relation_matrix = filtered_matrix
                relation_mapping = filtered_mapping
                n_relations = len(filtered_mapping)
            
            node_sizes = np.sum(relation_matrix, axis=1)
            if np.max(node_sizes) > 0:
                node_sizes = node_sizes / np.max(node_sizes) * node_size_factor
            else:
                node_sizes = np.ones(n_relations) * node_size_factor
            
            for i in range(n_relations):
                rel_name = relation_mapping[i]
                if len(rel_name) > 20:
                    rel_name = rel_name[:17] + "..."
                G.add_node(i, label=rel_name, size=node_sizes[i])

            for i in range(n_relations):
                for j in range(i+1, n_relations):  
                    weight = relation_matrix[i, j]
                    if weight > weight_threshold:
                        G.add_edge(i, j, weight=weight)
   
            if len(G.edges()) == 0:
                logging.warning(f"There are no edges in the graph. It could be a threshold ({weight_threshold}) set too high")
                plt.figure(figsize=figsize)
                plt.text(0.6, 0.6, "No edges found with current threshold", 
                        ha='center', va='center', fontsize=16)
                plt.title(title)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                return output_path
            plt.figure(figsize=figsize)
            if layout == "spring":
                pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
            elif layout == "circular":
                pos = nx.circular_layout(G)
            elif layout == "random":
                pos = nx.random_layout(G, seed=42)
            elif layout == "spectral":
                pos = nx.spectral_layout(G)
            elif layout == "shell":
                pos = nx.shell_layout(G)
            else:
                logging.warning(f"Unknown layout: {layout}, use spring layout")
                pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
            ax = plt.gca()
            nodes = nx.draw_networkx_nodes(
                G, pos, 
                node_size=[G.nodes[n]['size'] for n in G.nodes()],
                node_color=list(range(len(G.nodes()))),
                cmap=plt.get_cmap(colormap),
                alpha=0.8
            )
            edges = nx.draw_networkx_edges(
                G, pos,
                width=[G[u][v]['weight'] * edge_scale_factor for u, v in G.edges()],
                alpha=0.6,
                edge_cmap=plt.get_cmap('Blues')
            )
            if show_labels:
                labels = {n: G.nodes[n]['label'] for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family="sans-serif")
            if customize_fn is not None and callable(customize_fn):
                customize_fn(G, pos, ax)
            
            plt.title(title)
            plt.axis('off')

            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"The relation graph was saved to: {output_path}")
            return output_path
            
        except ImportError as e:
            logging.error(f"Lack of necessary libraries: {e}")
            raise ImportError(f"To use the relation graph visualization feature, install matplotlib and networkx: {e}")
        except Exception as e:
            logging.error(f"An error occurred during diagram visualizatio: {e}")
            raise
    

    def relation_selection(self, relation_matrix=None, relation_mapping=None, 
                       method="spectral_clustering", num_relations=10, 
                       **kwargs):
        """
        Select a subset of relations from the relation graph based on a specified selection method.

        Parameters:
            relation_matrix (numpy.ndarray): The relation graph matrix. If None, self.relation_graph() will be called to obtain it.
            relation_mapping (dict): Mapping from indices to relation names. If None, self.relation_graph() will be called to obtain it.
            method (str): The selection method to use. Options:
                - "independent_set": Select a maximum independent set to minimize redundancy.
                - "min_cut": Use a minimum cut algorithm to separate groups of relations.
                - "spectral_clustering": Apply spectral clustering to extract clusters of relations.
            num_relations (int or None): The number of relations to select. If None and method="spectral_clustering",
                returns a dictionary mapping each cluster to its contained relations instead of selecting relations.
            **kwargs: Additional parameters specific to the method.
                For "independent_set":
                    - use_lp (bool): Whether to use LP relaxation (True) or a greedy algorithm (False). Default: False.
                For "min_cut":
                    - num_partitions (int): The number of partitions to create. Default: 2.
                For "spectral_clustering":
                    - n_clusters (int): The number of clusters to create. Default: 3.
                    - assign_strategy (str): Strategy to select relations from clusters ("max_internal", "diverse"). Default: "max_internal".

        Returns:
            If num_relations is not None or method is not "spectral_clustering":
                list: The list of selected relation indices.
                list: The list of selected relation names.
            If num_relations is None and method is "spectral_clustering":
                dict: A dictionary mapping cluster IDs to lists of relation names contained in each cluster.
        """
    
        if relation_matrix is None or relation_mapping is None:
            logging.info("Relation matrix or mapping not provided, being called relation_graph()...")
            relation_matrix, relation_mapping = self.relation_graph()
        
        n_relations = relation_matrix.shape[0]
        logging.info(f"Use {method} select {num_relations} relation from {n_relations} relations")

        if num_relations is None and method == "spectral_clustering":
            logging.info(f"Performing spectral clustering to group relations without selection")
            
            # Get cluster assignments and other cluster information
            cluster_info = self._get_spectral_clusters(relation_matrix, relation_mapping, **kwargs)
            
            return cluster_info
        
        num_relations = min(num_relations, n_relations)
        if method == "independent_set":
            selected_indices = self._select_independent_set(relation_matrix, num_relations, **kwargs)
        elif method == "min_cut":
            selected_indices = self._select_min_cut(relation_matrix, num_relations, **kwargs)
        elif method == "spectral_clustering":
            selected_indices = self._select_spectral_clustering(relation_matrix, num_relations, **kwargs)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        if len(selected_indices) > num_relations:
            selected_indices = selected_indices[:num_relations]
        elif len(selected_indices) < num_relations and len(selected_indices) < n_relations:
            importances = np.sum(relation_matrix, axis=1)
            sorted_by_importance = np.argsort(-importances)
            
            additional_indices = [idx for idx in sorted_by_importance if idx not in selected_indices]
            needed = num_relations - len(selected_indices)
            selected_indices.extend(additional_indices[:needed])
        
        selected_relations = [relation_mapping[idx] for idx in selected_indices]
        
        logging.info(f"Already selected {len(selected_relations)} relations: {selected_relations}")
        
        return selected_indices, selected_relations

    def _select_independent_set(self, relation_matrix, num_relations, use_lp=False):
        """
        Select a maximum independent set of relations to minimize redundancy.

        Parameters:
            relation_matrix (numpy.ndarray): The relation graph matrix.
            num_relations (int): The number of relations to select.
            use_lp (bool): Whether to use LP relaxation (True) or a greedy algorithm (False).

        Returns:
            list: The list of selected relation indices.
        """
        n_relations = relation_matrix.shape[0]
        
        if use_lp:
            try:
                c = -np.ones(n_relations)  
                A_ub = []
                b_ub = []
                
                for i in range(n_relations):
                    for j in range(i+1, n_relations):
                        if relation_matrix[i, j] > 0: 
                            constraint = np.zeros(n_relations)
                            constraint[i] = 1
                            constraint[j] = 1
                            A_ub.append(constraint)
                            b_ub.append(1)
                
                A_ub = np.array(A_ub) if A_ub else np.zeros((0, n_relations))
                b_ub = np.array(b_ub) if b_ub else np.zeros(0)
                bounds = [(0, 1) for _ in range(n_relations)]
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point')
                
                if result.success:
                    solution = result.x
                    indices = np.argsort(-solution)

                    selected_indices = []
                    for idx in indices:
                        if len(selected_indices) >= num_relations:
                            break
                        is_independent = True
                        for sel_idx in selected_indices:
                            if relation_matrix[idx, sel_idx] > 0:
                                is_independent = False
                                break
                        
                        if is_independent:
                            selected_indices.append(idx)
                    
                    return selected_indices
                else:
                    logging.warning("LP optimization failed and fell back to greedy algorithm")
                    return self._select_independent_set(relation_matrix, num_relations, use_lp=False)
                    
            except ImportError:
                logging.warning("scipy.optimize can not be used and fell back to greedy algorithm")
                return self._select_independent_set(relation_matrix, num_relations, use_lp=False)
        
        else: 
            adj_matrix = relation_matrix.copy()
            degrees = np.sum(adj_matrix, axis=1)
            selected_indices = []
            available_indices = list(range(n_relations))
            
            while len(selected_indices) < num_relations and available_indices:
                max_degree_idx = max(available_indices, key=lambda idx: degrees[idx])
                selected_indices.append(max_degree_idx)
            
                neighbors = [i for i in available_indices if adj_matrix[max_degree_idx, i] > 0]
                for neighbor in neighbors:
                    if neighbor in available_indices:
                        available_indices.remove(neighbor)
                
                if max_degree_idx in available_indices:
                    available_indices.remove(max_degree_idx)
            
            return selected_indices

    def _select_min_cut(self, relation_matrix, num_relations, num_partitions=2):
        """
        Use a minimum cut algorithm to separate strongly connected groups of relations.

        Parameters:
            relation_matrix (numpy.ndarray): The relation graph matrix.
            num_relations (int): The number of relations to select.
            num_partitions (int): The number of partitions to create.

        Returns:
            list: The list of selected relation indices.
        """
        try:

            similarity_matrix = relation_matrix.copy()
            
            clustering = SpectralClustering(
                n_clusters=num_partitions,
                affinity='precomputed',
                random_state=42
            ).fit(similarity_matrix)

            labels = clustering.labels_
            
            cluster_sizes = np.bincount(labels, minlength=num_partitions)
            
            relations_per_cluster = {}
            remaining = num_relations
            
            for cluster_id in range(num_partitions):
                if cluster_sizes[cluster_id] > 0:
                    cluster_allocation = max(1, int(num_relations * cluster_sizes[cluster_id] / len(labels)))
                    cluster_allocation = min(cluster_allocation, remaining, cluster_sizes[cluster_id])
                    relations_per_cluster[cluster_id] = cluster_allocation
                    remaining -= cluster_allocation
            while remaining > 0:
                for cluster_id in sorted(range(num_partitions), key=lambda c: cluster_sizes[c], reverse=True):
                    if cluster_sizes[cluster_id] > relations_per_cluster.get(cluster_id, 0):
                        relations_per_cluster[cluster_id] = relations_per_cluster.get(cluster_id, 0) + 1
                        remaining -= 1
                        if remaining == 0:
                            break
            
            selected_indices = []
            
            for cluster_id, count in relations_per_cluster.items():
                cluster_indices = np.where(labels == cluster_id)[0]
                if len(cluster_indices) > 0:
                    cluster_importances = np.sum(relation_matrix[np.ix_(cluster_indices, cluster_indices)], axis=1)
                    sorted_cluster_indices = cluster_indices[np.argsort(-cluster_importances)]
                    selected_indices.extend(sorted_cluster_indices[:count])
            
            return selected_indices
            
        except ImportError:
            logging.warning("sklearn can not be used")
            importances = np.sum(relation_matrix, axis=1)
            sorted_indices = np.argsort(-importances)
            
            return sorted_indices[:num_relations].tolist()

    def _select_spectral_clustering(self, relation_matrix, num_relations, n_clusters=3, assign_strategy="max_internal"):
        """
        Apply spectral clustering on the relation graph to extract meaningful clusters.

        Parameters:
            relation_matrix (numpy.ndarray): The relation graph matrix.
            num_relations (int): The number of relations to select.
            n_clusters (int): The number of clusters to create.
            assign_strategy (str): The strategy to select relations from clusters.
                - "max_internal": Select from the cluster with the highest internal consistency.
                - "diverse": Select relations from different clusters to maximize diversity.

        Returns:
            list: The list of selected relation indices.
        """
        
        try:

            degrees = np.sum(relation_matrix, axis=1)
            D = np.diag(degrees)
            
            L = D - relation_matrix
            
            D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees + 1e-10))
            L_norm = D_sqrt_inv @ L @ D_sqrt_inv
            
            n_eigenvectors = min(n_clusters + 1, L_norm.shape[0] - 1)
            eigenvalues, eigenvectors = eigsh(L_norm, k=n_eigenvectors, which='SM')  # SM = Minimum eigenvalue
            
            # Sort the eigenvectors by their eigenvalues
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Skip the smallest eigenvalue/vector (zero eigenvalue)
            eigenvectors_to_use = eigenvectors[:, 1:n_clusters+1] if eigenvectors.shape[1] > 1 else eigenvectors
            
            # Clustering is performed using K-means on the top feature vector
            kmeans = KMeans(n_clusters=min(n_clusters, eigenvectors_to_use.shape[1]), random_state=42)
            cluster_labels = kmeans.fit_predict(eigenvectors_to_use)
            
            if assign_strategy == "max_internal":
                # Calculate the internal consistency of each cluster
                cluster_consistency = []
                
                for cluster_id in range(n_clusters):
                    cluster_indices = np.where(cluster_labels == cluster_id)[0]
                    
                    if len(cluster_indices) <= 1:
                        cluster_consistency.append(0)  
                        continue
                
                    internal_weights = np.sum(relation_matrix[np.ix_(cluster_indices, cluster_indices)])
                    max_possible_edges = len(cluster_indices) * (len(cluster_indices) - 1) / 2
                    consistency = internal_weights / max_possible_edges if max_possible_edges > 0 else 0
                    cluster_consistency.append(consistency)
                
                sorted_clusters = np.argsort(-np.array(cluster_consistency))
                
                selected_indices = []
                for cluster_id in sorted_clusters:
                    cluster_indices = np.where(cluster_labels == cluster_id)[0]
                    
                    if len(cluster_indices) == 0:
                        continue
                    
                    importances = np.sum(relation_matrix[cluster_indices, :], axis=1)
                    sorted_indices = cluster_indices[np.argsort(-importances)]
                    
                    available = min(len(sorted_indices), num_relations - len(selected_indices))
                    selected_indices.extend(sorted_indices[:available])
                    
                    if len(selected_indices) >= num_relations:
                        break
            
            elif assign_strategy == "diverse":
                # Select relationships from different clusters in a balanced manner
                
                # Calculate how many relationships to select from each cluster
                cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
                selection_counts = np.zeros(n_clusters, dtype=int)
                
                # Initial allocation, ensuring that at least one is selected from each non-empty cluster
                for cluster_id in range(n_clusters):
                    if cluster_sizes[cluster_id] > 0:
                        selection_counts[cluster_id] = 1
                
                # If we allocate too much, reduce from the smallest cluster
                while np.sum(selection_counts) > num_relations and np.any(selection_counts > 0):
                    smallest_cluster = np.argmin([
                        sizes if count > 0 else float('inf') 
                        for sizes, count in zip(cluster_sizes, selection_counts)
                    ])
                    selection_counts[smallest_cluster] -= 1
                
                # Allocate the remaining positions proportionally
                remaining = num_relations - np.sum(selection_counts)
                if remaining > 0:
                    for _ in range(remaining):
                        # Find the cluster with the lowest selection/total ratio
                        ratios = [
                            sel / size if size > 0 else float('inf')
                            for sel, size in zip(selection_counts, cluster_sizes)
                        ]
                        lowest_ratio_cluster = np.argmin(ratios)
                        selection_counts[lowest_ratio_cluster] += 1
                
                # Select a specified number of relations from each cluster
                selected_indices = []
                for cluster_id in range(n_clusters):
                    if selection_counts[cluster_id] > 0:
                        cluster_indices = np.where(cluster_labels == cluster_id)[0]
                        
                        if len(cluster_indices) == 0:
                            continue
                        
                        # In order of importance
                        importances = np.sum(relation_matrix[cluster_indices, :], axis=1)
                        sorted_indices = cluster_indices[np.argsort(-importances)]
                        
                        cnt = min(selection_counts[cluster_id], len(sorted_indices))
                        selected_indices.extend(sorted_indices[:cnt])
            
            else:
                raise ValueError(f"Unknown assign policy: {assign_strategy}")
                
            return selected_indices
            
        except ImportError as e:
            logging.warning(f"The package required for spectral clustering is not available: {e}")
            importances = np.sum(relation_matrix, axis=1)
            return np.argsort(-importances)[:num_relations].tolist()
    

    def _get_spectral_clusters(self, relation_matrix, relation_mapping, n_clusters=3, **kwargs):
        """
        Apply spectral clustering to the relation graph and return a dictionary mapping each cluster
        to the relations it contains.

        Parameters:
            relation_matrix (numpy.ndarray): The relation graph matrix.
            relation_mapping (dict): Mapping from indices to relation names.
            n_clusters (int): The number of clusters to create.
            **kwargs: Additional parameters for spectral clustering.

        Returns:
            dict: A dictionary with:
                - "cluster_mapping": Dictionary mapping cluster IDs to lists of relation names
                - "cluster_stats": Dictionary with statistics for each cluster
                - "eigenvalues": List of eigenvalues used in clustering
                - "cluster_labels": Array of cluster assignments for each relation
        """
        from scipy.sparse.linalg import eigsh
        try:
            # Calculate Laplacian matrix
            degrees = np.sum(relation_matrix, axis=1)
            D = np.diag(degrees)
            L = D - relation_matrix
            
            # Calculate normalized Laplacian matrix
            D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees + 1e-10))
            L_norm = D_sqrt_inv @ L @ D_sqrt_inv
            
            # Perform eigendecomposition
            n_eigenvectors = min(n_clusters + 1, L_norm.shape[0] - 1)
            eigenvalues, eigenvectors = eigsh(L_norm, k=n_eigenvectors, which='SM')  # SM = Smallest eigenvalues
            
            # Sort eigenvectors by eigenvalues
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Skip the smallest eigenvalue/vector (zero eigenvalue)
            eigenvectors_to_use = eigenvectors[:, 1:n_clusters+1] if eigenvectors.shape[1] > 1 else eigenvectors
            
            # Perform K-means clustering on the eigenvectors
            kmeans = KMeans(n_clusters=min(n_clusters, eigenvectors_to_use.shape[1]), random_state=42)
            cluster_labels = kmeans.fit_predict(eigenvectors_to_use)
            
            # Create a dictionary mapping cluster IDs to relation names
            cluster_mapping = {}
            for cluster_id in range(n_clusters):
                # Get indices of relations in this cluster
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                
                # Map indices to relation names
                relations_in_cluster = [relation_mapping[idx] for idx in cluster_indices]
                
                # Store in dictionary
                cluster_mapping[cluster_id] = relations_in_cluster
            
            # Compute statistics for each cluster
            cluster_stats = {}
            for cluster_id, relations in cluster_mapping.items():
                if not relations:
                    cluster_stats[cluster_id] = {
                        "size": 0,
                        "avg_similarity": 0,
                        "internal_density": 0,
                        "importance": 0
                    }
                    continue
                    
                # Get indices of relations in this cluster
                cluster_indices = [idx for idx, rel in relation_mapping.items() if rel in relations]
                
                # Calculate average similarity between relations in this cluster
                if len(cluster_indices) > 1:
                    cluster_submatrix = relation_matrix[np.ix_(cluster_indices, cluster_indices)]
                    # Remove diagonal (self-similarity)
                    np.fill_diagonal(cluster_submatrix, 0)
                    # Calculate average similarity
                    avg_similarity = np.sum(cluster_submatrix) / (len(cluster_indices) * (len(cluster_indices) - 1))
                    # Calculate density (proportion of edges that exist)
                    edge_count = np.sum(cluster_submatrix > 0)
                    max_edges = len(cluster_indices) * (len(cluster_indices) - 1)
                    internal_density = edge_count / max_edges if max_edges > 0 else 0
                else:
                    avg_similarity = 0
                    internal_density = 0
                
                # Calculate total importance of relations in this cluster
                importance = np.sum(np.sum(relation_matrix[cluster_indices, :], axis=1))
                
                # Store statistics
                cluster_stats[cluster_id] = {
                    "size": len(relations),
                    "avg_similarity": float(avg_similarity),
                    "internal_density": float(internal_density),
                    "importance": float(importance)
                }
            
            # Return comprehensive information
            return {
                "cluster_mapping": cluster_mapping,
                "cluster_stats": cluster_stats,
                "eigenvalues": eigenvalues.tolist(),
                "cluster_labels": cluster_labels.tolist()
            }
            
        except Exception as e:
            logging.error(f"Error during spectral clustering: {e}")
            # Return a simple mapping based on relation importance as fallback
            importances = np.sum(relation_matrix, axis=1)
            sorted_indices = np.argsort(-importances)
            
            # Create a dictionary with a single cluster containing all relations
            return {
                "cluster_mapping": {0: [relation_mapping[idx] for idx in sorted_indices]},
                "cluster_stats": {0: {"size": len(relation_mapping), "importance": float(np.sum(importances))}},
                "error": str(e)
            }
    
    def visualize_selected_relations(self, relation_matrix, relation_mapping, weight_threshold, selected_indices, 
                               output_path, highlight_selected=True):
        """
        Visualize the relation graph and highlight the selected relations.
        """
        G = nx.Graph()
        
        for i, rel_name in relation_mapping.items():
            display_name = rel_name[:17] + "..." if len(rel_name) > 20 else rel_name
            is_selected = i in selected_indices
            G.add_node(i, label=display_name, selected=is_selected)
        
        
        for i in range(len(relation_mapping)):
            for j in range(i+1, len(relation_mapping)):
                weight = relation_matrix[i, j]
                if weight > weight_threshold:
                    G.add_edge(i, j, weight=weight)
        
        plt.figure(figsize=(12, 10))
        
        pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
        
        nx.draw_networkx_edges(
            G, pos,
            width=[G[u][v]['weight'] * 3 for u, v in G.edges()],
            alpha=0.6,
            edge_color='gray'
        )
        
        if highlight_selected:
            selected_nodes = [n for n in G.nodes() if G.nodes[n]['selected']]
            non_selected_nodes = [n for n in G.nodes() if not G.nodes[n]['selected']]
            
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=non_selected_nodes,
                node_size=300,
                node_color='lightgray',
                alpha=0.7
            )
            
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=selected_nodes,
                node_size=500,
                node_color='red',
                alpha=0.9
            )
            
            labels = {n: G.nodes[n]['label'] for n in selected_nodes}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        else:
            nx.draw_networkx_nodes(
                G, pos, 
                node_size=300,
                node_color=['red' if G.nodes[n]['selected'] else 'lightblue' for n in G.nodes()],
                alpha=0.8
            )
            
            labels = {n: G.nodes[n]['label'] for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.axis('off')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"The selected relationship visualization is saved to: {output_path}")

    def optimize_spectral_clustering_params(self, relation_matrix=None, relation_mapping=None,
                                   min_clusters=2, max_clusters=20, step=1,
                                   num_relations=10, evaluation_metrics=None,
                                   trials=1, assign_strategy="max_internal",
                                   visualize=True, output_dir="outputs",
                                   interactive=False, eigenvalue_gap_analysis=True):
        """
        Find the optimal hyperparameter n_clusters for spectral clustering relation selection through experimentation
        
        Parameters:
            relation_matrix (numpy.ndarray): Relation graph matrix. If None, self.relation_graph() will be called
            relation_mapping (dict): Index to relation name mapping. If None, self.relation_graph() will be called
            min_clusters (int): Minimum number of clusters to try
            max_clusters (int): Maximum number of clusters to try
            step (int): Step size for increasing cluster numbers
            num_relations (int): Number of relations to select
            evaluation_metrics (list): List of evaluation metrics to use. If None, default metrics will be used
            trials (int): Number of repetitions for each n_clusters value to assess stability
            assign_strategy (str): Strategy for selecting relations from clusters ("diverse" or "max_internal")
            visualize (bool): Whether to generate visualization results
            output_dir (str): Directory to save results and visualization charts
            interactive (bool): Whether to return interactive charts (requires additional libraries)
            eigenvalue_gap_analysis (bool): Whether to perform and visualize eigenvalue gap analysis
            
        Returns:
            dict: Dictionary containing optimal parameters and evaluation results
        """
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # If relation matrix or mapping not provided, call relation_graph
        if relation_matrix is None or relation_mapping is None:
            logging.info("Relation matrix or mapping not provided, calling relation_graph()...")
            relation_matrix, relation_mapping = self.relation_graph()
        
        n_relations = relation_matrix.shape[0]
        logging.info(f"Starting spectral clustering optimization, total relations: {n_relations}")
        
        # Ensure num_relations is valid
        num_relations = min(num_relations, n_relations)
        
        # If evaluation metrics not specified, use default metrics
        if evaluation_metrics is None:
            evaluation_metrics = [
                "silhouette_score",      # Silhouette coefficient - measures cluster cohesion and separation
                "calinski_harabasz",     # Calinski-Harabasz index - higher is better
                "davies_bouldin",        # Davies-Bouldin index - lower is better
                "eigenvalue_gap",        # Eigenvalue gap - identifies natural cluster separations
                "relation_coverage",     # Relation coverage - measures proportion of triples covered by selected relations
                "relation_diversity",    # Relation diversity - measures semantic diversity of selected relations
                "graph_connectivity"     # Graph connectivity - measures connectivity of subgraph formed by selected relations
            ]
        
        # Initialize results storage
        results = {
            "n_clusters": [],
            "metrics": defaultdict(list),
            "std_metrics": defaultdict(list),
            "selected_relations": defaultdict(list),
            "best_params": {},
            "execution_time": defaultdict(list)
        }
        
        # Try different n_clusters values
        cluster_range = range(min_clusters, min(max_clusters + 1, n_relations), step)
        for n_clusters in cluster_range:
            logging.info(f"Evaluating n_clusters = {n_clusters}")
            
            # Initialize trial results for each metric
            trial_metrics = defaultdict(list)
            trial_relations = []
            
            # Multiple trials to evaluate stability
            for trial in range(trials):
                start_time = time.time()
                
                # Perform relation selection with current n_clusters
                selected_indices = self._select_spectral_clustering(
                    relation_matrix, num_relations, n_clusters, assign_strategy
                )
                
                # Convert indices to relation names
                selected_relations = [relation_mapping[idx] for idx in selected_indices]
                
                # Record selected relations
                trial_relations.append(selected_relations)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                results["execution_time"][n_clusters].append(execution_time)
                
                # Evaluate results
                metrics = self._evaluate_clustering(
                    relation_matrix, relation_mapping, selected_indices, 
                    n_clusters, evaluation_metrics
                )
                
                # Store metrics for this trial
                for metric_name, value in metrics.items():
                    trial_metrics[metric_name].append(value)
            
            # Calculate mean and standard deviation for each metric
            results["n_clusters"].append(n_clusters)
            for metric_name, values in trial_metrics.items():
                results["metrics"][metric_name].append(np.mean(values))
                results["std_metrics"][metric_name].append(np.std(values))
            
            # Store most commonly selected relations
            # Simple approach: relations from different trials may vary slightly, we store the most frequent ones
            all_selected = []
            for relations in trial_relations:
                all_selected.extend(relations)
            
            most_common = Counter(all_selected).most_common(num_relations)
            results["selected_relations"][n_clusters] = [rel for rel, _ in most_common]
        
        # Determine the best n_clusters
        best_params = self._determine_best_params(results, evaluation_metrics)
        results["best_params"] = best_params
        
        # Visualize results
        if visualize:
            self._visualize_parameter_optimization(results, output_dir, evaluation_metrics, interactive)
        
        # Perform eigenvalue gap analysis if requested
        if eigenvalue_gap_analysis and visualize:
            try:
                logging.info("Performing eigenvalue gap analysis...")
                gap_suggested_k, eigenvalues, gaps = self.visualize_eigenvalue_gap(
                    relation_matrix, 
                    output_path=os.path.join(output_dir, "eigenvalue_gap_analysis.png")
                )
                
                if gap_suggested_k is not None:
                    logging.info(f"Eigenvalue gap analysis suggests optimal n_clusters = {gap_suggested_k}")
                    results["eigenvalue_gap_analysis"] = {
                        "suggested_k": gap_suggested_k,
                        "eigenvalues": eigenvalues.tolist() if eigenvalues is not None else None,
                        "gaps": gaps.tolist() if gaps is not None else None
                    }
            except Exception as e:
                logging.warning(f"Error during eigenvalue gap analysis: {e}")
        
        # Perform final selection with optimal parameters
        best_n_clusters = best_params["n_clusters"]
        logging.info(f"Best n_clusters value: {best_n_clusters}")
        logging.info(f"Best evaluation scores: {best_params['scores']}")
        
        final_indices = self._select_spectral_clustering(
            relation_matrix, num_relations, best_n_clusters, assign_strategy
        )
        
        # Convert indices to relation names
        final_relations = [relation_mapping[idx] for idx in final_indices]
        
        # Visualize final selected relations
        if visualize:
            output_path = os.path.join(output_dir, "optimal_relation_selection.png")
            self._visualize_selected_relations(
                relation_matrix, relation_mapping, final_indices,
                output_path, f"Relations selected with optimal n_clusters={best_n_clusters}"
            )
        
        logging.info(f"Relations selected with optimal parameters: {final_relations}")
        
        return {
            "best_n_clusters": best_n_clusters,
            "evaluation_results": results,
            "selected_indices": final_indices,
            "selected_relations": final_relations,
            "eigenvalue_gap_suggested_k": results.get("eigenvalue_gap_analysis", {}).get("suggested_k", None)
        }

    def _evaluate_clustering(self, relation_matrix, relation_mapping, selected_indices, 
                            n_clusters, evaluation_metrics):
        """
        Evaluate the quality of clustering results
        
        Parameters:
            relation_matrix (numpy.ndarray): Relation graph matrix
            relation_mapping (dict): Index to relation name mapping
            selected_indices (list): Indices of selected relations
            n_clusters (int): Number of clusters used
            evaluation_metrics (list): Evaluation metrics to calculate
            
        Returns:
            dict: Scores for different evaluation metrics
        """
        metrics = {}
        try:
            # If using silhouette score
            if "silhouette_score" in evaluation_metrics:
                try:
                    # Prepare feature vectors for spectral clustering
                    # Calculate Laplacian matrix
                    n_relations = relation_matrix.shape[0]
                    degrees = np.sum(relation_matrix, axis=1)
                    D = np.diag(degrees)
                    L = D - relation_matrix
                    # Calculate normalized Laplacian matrix
                    D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees + 1e-10))
                    L_norm = D_sqrt_inv @ L @ D_sqrt_inv
                    
                    # Perform eigendecomposition
                    n_eigenvectors = min(n_clusters + 1, L_norm.shape[0] - 1)
                    eigenvalues, eigenvectors = eigsh(L_norm, k=n_eigenvectors, which='SM')
                    
                    # Sort eigenvectors by eigenvalues
                    idx = np.argsort(eigenvalues)
                    eigenvectors = eigenvectors[:, idx]
                    
                    # Skip the smallest eigenvalue/vector (corresponding to zero eigenvalue)
                    eigenvectors_to_use = eigenvectors[:, 1:n_clusters+1] if eigenvectors.shape[1] > 1 else eigenvectors
                    
                    # Use K-means clustering
                    kmeans = KMeans(n_clusters=min(n_clusters, eigenvectors_to_use.shape[1]), random_state=42)
                    cluster_labels = kmeans.fit_predict(eigenvectors_to_use)
                    
                    # Calculate silhouette score
                    if len(set(cluster_labels)) > 1:  # Ensure multiple clusters exist
                        score = silhouette_score(eigenvectors_to_use, cluster_labels)
                        metrics["silhouette_score"] = score
                    else:
                        metrics["silhouette_score"] = -1  # Set to lowest score if only one cluster
                except Exception as e:
                    import logging
                    logging.warning(f"Error calculating silhouette score: {e}")
                    metrics["silhouette_score"] = -1
            
            # If using Calinski-Harabasz index
            if "calinski_harabasz" in evaluation_metrics:
                try:
                    from sklearn.metrics import calinski_harabasz_score
                    # Use the same feature vectors and labels as silhouette score
                    if "eigenvectors_to_use" in locals() and "cluster_labels" in locals():
                        if len(set(cluster_labels)) > 1:  # Ensure multiple clusters exist
                            score = calinski_harabasz_score(eigenvectors_to_use, cluster_labels)
                            metrics["calinski_harabasz"] = score
                        else:
                            metrics["calinski_harabasz"] = 0  # Set to 0 if only one cluster
                except Exception as e:
                    import logging
                    logging.warning(f"Error calculating Calinski-Harabasz index: {e}")
                    metrics["calinski_harabasz"] = 0
            
            # If using Davies-Bouldin index
            if "davies_bouldin" in evaluation_metrics:
                try:
                    from sklearn.metrics import davies_bouldin_score
                    # Use the same feature vectors and labels as silhouette score
                    if "eigenvectors_to_use" in locals() and "cluster_labels" in locals():
                        if len(set(cluster_labels)) > 1:  # Ensure multiple clusters exist
                            score = davies_bouldin_score(eigenvectors_to_use, cluster_labels)
                            metrics["davies_bouldin"] = score
                        else:
                            metrics["davies_bouldin"] = float('inf')  # Set to infinity (worst) if only one cluster
                except Exception as e:
                    import logging
                    logging.warning(f"Error calculating Davies-Bouldin index: {e}")
                    metrics["davies_bouldin"] = float('inf')
            

            # If using Eigenvalue Gap method
            if "eigenvalue_gap" in evaluation_metrics:
                try:
                    # We need to compute eigenvalues if they aren't already computed
                    # Check if eigenvalues were computed for silhouette score
                    if "eigenvalues" not in locals():
                        # Calculate Laplacian matrix if not already done
                        if "L_norm" not in locals():
                            n_relations = relation_matrix.shape[0]
                            degrees = np.sum(relation_matrix, axis=1)
                            D = np.diag(degrees)
                            L = D - relation_matrix
                            
                            # Calculate normalized Laplacian matrix
                            D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees + 1e-10))
                            L_norm = D_sqrt_inv @ L @ D_sqrt_inv
                        
                        # Compute eigenvalues (use more eigenvalues than just n_clusters)
                        max_k = min(n_relations - 1, 20)  # Use at most 20 eigenvalues for efficiency
                        eigenvalues, _ = eigsh(L_norm, k=max_k, which='SM')
                        eigenvalues = np.sort(eigenvalues)  # Sort in ascending order
                    
                    # Calculate gaps between consecutive eigenvalues
                    if len(eigenvalues) > 1:
                        gaps = np.diff(eigenvalues)
                        
                        # If n_clusters is within the range of our eigenvalues
                        if n_clusters - 1 < len(gaps):
                            # Check if the gap at n_clusters position is significant
                            
                            # Normalize the gaps for better comparison
                            normalized_gaps = gaps / np.mean(gaps) if np.mean(gaps) > 0 else gaps
                            
                            # Get the gap at the position corresponding to n_clusters
                            gap_at_k = normalized_gaps[n_clusters - 1]
                            
                            # Find the maximum gap
                            max_gap = np.max(normalized_gaps)
                            
                            # Calculate ratio of current gap to max gap
                            gap_ratio = gap_at_k / max_gap if max_gap > 0 else 0
                            
                            # Score is higher if gap at n_clusters is close to the maximum gap
                            metrics["eigenvalue_gap"] = gap_ratio
                            
                            # If n_clusters exactly corresponds to max gap position, give maximum score
                            if n_clusters - 1 == np.argmax(normalized_gaps):
                                metrics["eigenvalue_gap"] = 1.0
                        else:
                            # n_clusters is too large for our eigenvalue calculations
                            metrics["eigenvalue_gap"] = 0.0
                    else:
                        metrics["eigenvalue_gap"] = 0.0
                except Exception as e:
                    import logging
                    logging.warning(f"Error calculating Eigenvalue Gap metric: {e}")
                    metrics["eigenvalue_gap"] = 0.0

            
            # Relation coverage - evaluates how many triples are covered by selected relations
            if "relation_coverage" in evaluation_metrics:
                try:
                    # Calculate the proportion of triples covered by selected relations
                    selected_relation_names = [relation_mapping[idx] for idx in selected_indices]
                    
                    # Count occurrences of each relation in triples
                    relation_counts = {}
                    total_triples = 0
                    
                    for split in ['train', 'valid', 'test']:
                        for _, rel, _ in self.triples[split]:
                            relation_counts[rel] = relation_counts.get(rel, 0) + 1
                            total_triples += 1
                    
                    # Calculate number of triples covered by selected relations
                    covered_triples = sum(relation_counts.get(rel, 0) for rel in selected_relation_names)
                    
                    # Calculate coverage
                    coverage = covered_triples / total_triples if total_triples > 0 else 0
                    metrics["relation_coverage"] = coverage
                except Exception as e:
                    import logging
                    logging.warning(f"Error calculating relation coverage: {e}")
                    metrics["relation_coverage"] = 0
            
            # Relation diversity - evaluates semantic diversity of selected relations
            if "relation_diversity" in evaluation_metrics:
                try:
                    # Calculate average similarity between selected relations, lower similarity means higher diversity
                    if len(selected_indices) > 1:
                        similarities = []
                        for i, idx1 in enumerate(selected_indices):
                            for j, idx2 in enumerate(selected_indices):
                                if i < j:  # Avoid duplicates
                                    similarities.append(relation_matrix[idx1, idx2])
                        
                        # Average similarity lower means diversity higher, convert to diversity score (1 - avg_similarity)
                        avg_similarity = np.mean(similarities) if similarities else 0
                        diversity = 1 - avg_similarity
                        metrics["relation_diversity"] = diversity
                    else:
                        metrics["relation_diversity"] = 0  # Only one relation means diversity is 0
                except Exception as e:
                    import logging
                    logging.warning(f"Error calculating relation diversity: {e}")
                    metrics["relation_diversity"] = 0
            
            # Graph connectivity - evaluates connectivity of subgraph formed by selected relations
            if "graph_connectivity" in evaluation_metrics:
                try:
                    import networkx as nx
                    
                    # Create subgraph of selected relations
                    G = nx.Graph()
                    
                    # Add nodes
                    for idx in selected_indices:
                        G.add_node(idx)
                    
                    # Add edges (if two relations have strong connections)
                    threshold = 0.3  # This threshold can be adjusted
                    for i, idx1 in enumerate(selected_indices):
                        for j, idx2 in enumerate(selected_indices):
                            if i < j and relation_matrix[idx1, idx2] > threshold:
                                G.add_edge(idx1, idx2, weight=relation_matrix[idx1, idx2])
                    
                    # Calculate number of connected components
                    num_components = nx.number_connected_components(G)
                    
                    # Calculate average degree
                    if len(G.nodes()) > 0:
                        avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                    else:
                        avg_degree = 0
                    
                    # Combined metric of connected components and average degree
                    # Fewer components and higher degree is better
                    connectivity = (avg_degree + 1) / (num_components + 1)
                    metrics["graph_connectivity"] = connectivity
                except Exception as e:
                    import logging
                    logging.warning(f"Error calculating graph connectivity: {e}")
                    metrics["graph_connectivity"] = 0
        
        except Exception as e:
            import logging
            logging.error(f"Error evaluating clustering quality: {e}")
        
        return metrics

    def _determine_best_params(self, results, evaluation_metrics):
        """
        Determine the best parameters based on evaluation metrics
        
        Parameters:
            results (dict): Evaluation results for different n_clusters
            evaluation_metrics (list): List of evaluation metrics used
            
        Returns:
            dict: Best parameters and corresponding scores
        """
        # Goal for each metric (maximize or minimize)
        metric_goals = {
            "silhouette_score": "max",       # Higher is better
            "calinski_harabasz": "max",      # Higher is better
            "davies_bouldin": "min",         # Lower is better
            "eigenvalue_gap": "max",         # Higher is better
            "relation_coverage": "max",      # Higher is better
            "relation_diversity": "max",      # Higher is better
            "graph_connectivity": "max"      # Higher is better
        }
        
        # Calculate composite score for each n_clusters
        n_clusters_values = results["n_clusters"]
        scores = np.zeros(len(n_clusters_values))
        
        # Calculate normalized score for each metric
        for metric in evaluation_metrics:
            if metric not in results["metrics"]:
                continue
                
            values = np.array(results["metrics"][metric])
            
            # If all values are the same, this metric provides no information
            if np.all(values == values[0]):
                continue
                
            # Normalize to [0,1] range
            min_val = np.min(values)
            max_val = np.max(values)
            
            if max_val - min_val > 0:
                if metric_goals[metric] == "max":
                    # For maximization goals, higher values get higher scores
                    normalized = (values - min_val) / (max_val - min_val)
                else:
                    # For minimization goals, lower values get higher scores
                    normalized = 1 - (values - min_val) / (max_val - min_val)
                
                # Add normalized scores to total
                scores += normalized
        
        # Find n_clusters with highest score
        best_idx = np.argmax(scores)
        best_n_clusters = n_clusters_values[best_idx]
        
        # Collect all metric scores for this n_clusters
        best_scores = {}
        for metric in evaluation_metrics:
            if metric in results["metrics"]:
                best_scores[metric] = results["metrics"][metric][best_idx]
        
        return {
            "n_clusters": best_n_clusters,
            "scores": best_scores,
            "overall_score": scores[best_idx]
        }

    def _visualize_parameter_optimization(self, results, output_dir, evaluation_metrics, interactive=False):
        """
        Visualize parameter optimization results
        
        Parameters:
            results (dict): Results of optimization experiments
            output_dir (str): Directory to save visualization charts
            evaluation_metrics (list): Evaluation metrics used
            interactive (bool): Whether to create interactive charts
        """
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        n_clusters_values = results["n_clusters"]
        
        if interactive:
            try:
                # Create multiple subplots
                rows = (len(evaluation_metrics) + 1) // 2
                fig = make_subplots(rows=rows, cols=2, 
                                subplot_titles=[metric for metric in evaluation_metrics if metric in results["metrics"]])
                
                # Plot line graph for each metric
                plot_idx = 1
                for metric in evaluation_metrics:
                    if metric not in results["metrics"]:
                        continue
                    
                    row = (plot_idx - 1) // 2 + 1
                    col = (plot_idx - 1) % 2 + 1
                    
                    values = results["metrics"][metric]
                    errors = results["std_metrics"][metric]
                    
                    # Add line chart and error band
                    fig.add_trace(
                        go.Scatter(
                            x=n_clusters_values,
                            y=values,
                            mode="lines+markers",
                            name=metric,
                            line=dict(width=2),
                            marker=dict(size=8)
                        ),
                        row=row, col=col
                    )
                    
                    # Add error band
                    fig.add_trace(
                        go.Scatter(
                            x=n_clusters_values + n_clusters_values[::-1],
                            y=[v+e for v, e in zip(values, errors)] + [v-e for v, e in zip(values, errors)][::-1],
                            fill="toself",
                            fillcolor="rgba(0,176,246,0.2)",
                            line=dict(color="rgba(255,255,255,0)"),
                            name=f"{metric} std",
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                    
                    plot_idx += 1
                
                # Update layout
                fig.update_layout(
                    title="Spectral Clustering Parameter Optimization Results",
                    xaxis_title="Number of Clusters (n_clusters)",
                    height=300 * rows,
                    width=1000,
                    showlegend=True
                )
                
                # Save as HTML file
                fig.write_html(os.path.join(output_dir, "parameter_optimization_interactive.html"))
                
            except ImportError:
                # Fall back to non-interactive charts if plotly is not available
                interactive = False
        
        if not interactive:
            # Non-interactive charts
            n_metrics = len([m for m in evaluation_metrics if m in results["metrics"]])
            rows = (n_metrics + 1) // 2
            
            plt.figure(figsize=(12, 4 * rows))
            
            # Plot subplot for each metric
            plot_idx = 1
            for metric in evaluation_metrics:
                if metric not in results["metrics"]:
                    continue
                    
                plt.subplot(rows, 2, plot_idx)
                
                values = results["metrics"][metric]
                errors = results["std_metrics"][metric]
                
                # Plot line chart and error bands
                plt.errorbar(n_clusters_values, values, yerr=errors, 
                            fmt='-o', capsize=5, elinewidth=1, markersize=5)
                
                plt.title(metric)
                plt.xlabel("Number of Clusters (n_clusters)")
                plt.ylabel("Score")
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Mark best value
                best_idx = np.argmax(values) if metric != "davies_bouldin" else np.argmin(values)
                best_value = values[best_idx]
                best_cluster = n_clusters_values[best_idx]
                
                plt.scatter([best_cluster], [best_value], c='red', s=100, zorder=5)
                plt.annotate(f"Best: {best_cluster}", 
                            (best_cluster, best_value),
                            xytext=(0, 10), textcoords="offset points",
                            ha='center', va='bottom',
                            bbox=dict(boxstyle="round,pad=0.3", alpha=0.1))
                
                plot_idx += 1
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "parameter_optimization.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot execution time
        plt.figure(figsize=(10, 5))
        
        exec_times = [np.mean(results["execution_time"][n]) for n in n_clusters_values]
        exec_errors = [np.std(results["execution_time"][n]) for n in n_clusters_values]
        
        plt.errorbar(n_clusters_values, exec_times, yerr=exec_errors, 
                    fmt='-o', capsize=5, elinewidth=1, markersize=5)
        
        plt.title("Execution Time for Different Cluster Numbers")
        plt.xlabel("Number of Clusters (n_clusters)")
        plt.ylabel("Execution Time (seconds)")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "execution_time.png"), dpi=300, bbox_inches='tight')
        plt.close()


    def visualize_eigenvalue_gap(self, relation_matrix, output_path="eigenvalue_gap.png", max_k=20):
        """
        Visualize the eigenvalue spectrum and eigenvalue gaps of the relation graph Laplacian.
        This can help identify the optimal number of clusters.
        
        Parameters:
            relation_matrix (numpy.ndarray): Relation graph matrix
            output_path (str): Path to save the visualization
            max_k (int): Maximum number of eigenvalues to compute
            
        Returns:
            tuple: Suggested optimal number of clusters, eigenvalues, and gaps
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Calculate Laplacian matrix
            n_relations = relation_matrix.shape[0]
            degrees = np.sum(relation_matrix, axis=1)
            D = np.diag(degrees)
            L = D - relation_matrix
            
            # Calculate normalized Laplacian matrix
            D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees + 1e-10))
            L_norm = D_sqrt_inv @ L @ D_sqrt_inv
            
            # Compute eigenvalues
            max_k = min(n_relations - 1, max_k)  # Use at most max_k eigenvalues
            eigenvalues, _ = eigsh(L_norm, k=max_k, which='SM')
            eigenvalues = np.sort(eigenvalues)  # Sort in ascending order
            
            # Calculate gaps between consecutive eigenvalues
            gaps = np.diff(eigenvalues)
            
            # Find the position of the maximum gap
            max_gap_idx = np.argmax(gaps[1:]) + 1
            optimal_k = max_gap_idx + 1  # Number of clusters is position + 1
            
            # Create visualization with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Plot eigenvalues
            ax1.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', markersize=8)
            ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
            ax1.set_title("Eigenvalue Spectrum of Normalized Laplacian")
            ax1.set_xlabel("Index")
            ax1.set_ylabel("Eigenvalue")
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()
            
            # Plot eigenvalue gaps
            ax2.plot(range(1, len(gaps) + 1), gaps, 'o-', markersize=8)
            ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
            ax2.set_title("Eigenvalue Gaps")
            ax2.set_xlabel("k")
            ax2.set_ylabel("Gap (λₖ₊₁ - λₖ)")
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
            
            # Highlight the maximum gap
            ax2.plot(optimal_k, gaps[max_gap_idx], 'ro', markersize=10)
            ax2.annotate(f'Max gap: {gaps[max_gap_idx]:.4f}',
                        xy=(optimal_k, gaps[max_gap_idx]),
                        xytext=(optimal_k + 1, gaps[max_gap_idx] * 1.1),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Eigenvalue gap visualization saved to: {output_path}")
            logging.info(f"Suggested optimal number of clusters (k): {optimal_k}")
            
            return optimal_k, eigenvalues, gaps
            
        except ImportError as e:
            logging.error(f"Required libraries not available: {e}")
            return None, None, None
        except Exception as e:
            logging.error(f"Error during eigenvalue gap visualization: {e}")
            return None, None, None
    

    def _visualize_selected_relations(self, relation_matrix, relation_mapping, selected_indices, 
                                    output_path, title):
        """
        Visualize selected relations
        
        Parameters:
            relation_matrix (numpy.ndarray): Relation graph matrix
            relation_mapping (dict): Index to relation name mapping
            selected_indices (list): Indices of selected relations
            output_path (str): Path to save visualization
            title (str): Chart title
        """
        try:
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Create complete graph
            G = nx.Graph()
            
            # Add all nodes
            for i, rel_name in relation_mapping.items():
                # Truncate long relation names
                display_name = rel_name[:17] + "..." if len(rel_name) > 20 else rel_name
                is_selected = i in selected_indices
                G.add_node(i, label=display_name, selected=is_selected)
            
            # Add edges
            weight_threshold = 0.1
            for i in range(len(relation_mapping)):
                for j in range(i+1, len(relation_mapping)):
                    weight = relation_matrix[i, j]
                    if weight > weight_threshold:
                        G.add_edge(i, j, weight=weight)
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Use spring layout
            pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                width=[G[u][v]['weight'] * 3 for u, v in G.edges()],
                alpha=0.5,
                edge_color='gray'
            )
            
            # Draw selected and non-selected nodes separately
            selected_nodes = [n for n in G.nodes() if G.nodes[n]['selected']]
            non_selected_nodes = [n for n in G.nodes() if not G.nodes[n]['selected']]
            
            # Draw non-selected nodes (smaller)
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=non_selected_nodes,
                node_size=300,
                node_color='lightgray',
                alpha=0.7
            )
            
            # Draw selected nodes (larger and different color)
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=selected_nodes,
                node_size=500,
                node_color='red',
                alpha=0.9
            )
            
            # Draw labels only for selected nodes
            labels = {n: G.nodes[n]['label'] for n in selected_nodes}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
            
            plt.title(title)
            plt.axis('off')
            
            # Save image
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError as e:
            import logging
            logging.warning(f"Missing libraries required for visualization: {e}")
        except Exception as e:
            import logging
            logging.error(f"Error visualizing relation selection: {e}")


    def visualize_relation_clusters(self, relation_matrix, relation_mapping, weight_threshold, cluster_info, 
                              output_path="relation_clusters.png", figsize=(14, 12)):
        """
        Visualize the relation clusters obtained from spectral clustering.
        
        Parameters:
            relation_matrix (numpy.ndarray): The relation graph matrix.
            relation_mapping (dict): Mapping from indices to relation names.
            cluster_info (dict): The cluster information returned by relation_selection or _get_spectral_clusters.
            output_path (str): Path to save the visualization.
            figsize (tuple): Size of the figure (width, height).
            
        Returns:
            str: Path to the saved visualization.
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Create graph
            G = nx.Graph()
            
            # Extract cluster labels and mapping
            cluster_mapping = cluster_info.get("cluster_mapping", {})
            cluster_stats = cluster_info.get("cluster_stats", {})
            
            # Create reverse mapping from relation name to index
            relation_to_idx = {name: idx for idx, name in relation_mapping.items()}
            
            # Add nodes with cluster information
            for cluster_id, relations in cluster_mapping.items():
                for relation in relations:
                    if relation in relation_to_idx:
                        idx = relation_to_idx[relation]
                        # Shorten relation name if too long
                        display_name = relation[:20] + "..." if len(relation) > 20 else relation
                        G.add_node(idx, name=display_name, cluster=cluster_id)
            
            # Add edges (connections between relations)
            for i in range(len(relation_mapping)):
                if i not in G.nodes():
                    continue
                    
                for j in range(i+1, len(relation_mapping)):
                    if j not in G.nodes():
                        continue
                        
                    weight = relation_matrix[i, j]
                    if weight > weight_threshold:
                        G.add_edge(i, j, weight=weight)
            
            # Create figure
            plt.figure(figsize=figsize)
            
            # Position nodes using ForceAtlas2-like layout
            pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
            
            # Get number of clusters
            n_clusters = len(cluster_mapping)
            
            # Create a colormap
            import matplotlib.cm as cm
            colors = cm.tab10(np.linspace(0, 1, n_clusters))
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                width=[G[u][v]['weight'] * 3 for u, v in G.edges()],
                alpha=0.4,
                edge_color='gray'
            )
            
            # Draw nodes for each cluster
            for cluster_id in range(n_clusters):
                # Get nodes in this cluster
                node_list = [n for n in G.nodes() if G.nodes[n].get('cluster') == cluster_id]
                
                if not node_list:
                    continue
                    
                # Draw nodes
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=node_list,
                    node_size=300,
                    node_color=[colors[cluster_id]],
                    label=f"Cluster {cluster_id} (size: {len(node_list)})"
                )
            
            # Draw node labels
            nx.draw_networkx_labels(
                G, pos,
                labels={n: G.nodes[n].get('name') for n in G.nodes()},
                font_size=8,
                font_weight='normal'
            )
            
            # Add cluster statistics as a text box
            cluster_text = "Cluster Statistics:\n" + "-" * 40 + "\n"
            for cluster_id, stats in cluster_stats.items():
                if cluster_id not in cluster_mapping:
                    continue
                    
                size = stats.get("size", 0)
                avg_sim = stats.get("avg_similarity", 0)
                density = stats.get("internal_density", 0)
                
                cluster_text += f"Cluster {cluster_id}: {size} relations\n"
                cluster_text += f"   Internal similarity: {avg_sim:.3f}\n"
                cluster_text += f"   Density: {density:.3f}\n"
                cluster_text += "-" * 40 + "\n"
            
            plt.figtext(0.01, 0.01, cluster_text, fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
            # Add title and legend
            plt.title("Relation Clusters from Spectral Clustering", fontsize=16)
            plt.legend(loc='upper right')
            
            # Remove axis
            plt.axis('off')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Cluster visualization saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error during cluster visualization: {e}")
            return None

    def analyze_relation_clusters(self, relation_matrix, relation_mapping, cluster_info):
        """
        Analyze the relation clusters and print comprehensive statistics.
        
        Parameters:
            relation_matrix (numpy.ndarray): The relation graph matrix.
            relation_mapping (dict): Mapping from indices to relation names.
            cluster_info (dict): The cluster information returned by relation_selection or _get_spectral_clusters.
            
        Returns:
            dict: Extended analysis information about the clusters.
        """
        
        try:
            # Extract cluster mapping
            cluster_mapping = cluster_info.get("cluster_mapping", {})
            n_clusters = len(cluster_mapping)
            
            logging.info(f"Analyzing {n_clusters} relation clusters...")
            
            # Create reverse mapping from relation name to index
            relation_to_idx = {name: idx for idx, name in relation_mapping.items()}
            
            # Initialize analysis dictionary
            analysis = {
                "cluster_details": {},
                "cross_cluster_relations": [],
                "overall_stats": {}
            }
            
            # Analyze each cluster
            for cluster_id, relations in cluster_mapping.items():
                # Skip empty clusters
                if not relations:
                    analysis["cluster_details"][cluster_id] = {
                        "size": 0,
                        "relations": [],
                        "key_relations": [],
                        "avg_weight": 0,
                        "density": 0,
                        "isolation": 0
                    }
                    continue
                    
                # Get indices of relations in this cluster
                indices = [relation_to_idx[rel] for rel in relations if rel in relation_to_idx]
                
                # Calculate internal weights (connections within cluster)
                internal_weights = []
                for i, idx1 in enumerate(indices):
                    for j, idx2 in enumerate(indices):
                        if i < j:  # Avoid duplicates and self-loops
                            internal_weights.append(relation_matrix[idx1, idx2])
                
                avg_internal_weight = np.mean(internal_weights) if internal_weights else 0
                
                # Calculate density (proportion of potential connections that exist)
                potential_connections = len(indices) * (len(indices) - 1) / 2
                actual_connections = sum(1 for w in internal_weights if w > 0.1)  # Count edges above threshold
                density = actual_connections / potential_connections if potential_connections > 0 else 0
                
                # Calculate isolation (how separated this cluster is from others)
                # Higher value means more isolated
                external_weights = []
                for idx1 in indices:
                    for idx2 in range(len(relation_mapping)):
                        if idx2 not in indices:  # Only connections to other clusters
                            external_weights.append(relation_matrix[idx1, idx2])
                
                avg_external_weight = np.mean(external_weights) if external_weights else 0
                isolation = 1 - (avg_external_weight / (avg_internal_weight + 1e-10))  # Avoid division by zero
                
                # Find key relations (most central to the cluster)
                relation_importance = {}
                for rel in relations:
                    if rel not in relation_to_idx:
                        continue
                        
                    idx = relation_to_idx[rel]
                    # Sum of weights to other relations in the cluster
                    importance = sum(relation_matrix[idx, relation_to_idx[other]] 
                                    for other in relations if other in relation_to_idx)
                    relation_importance[rel] = importance
                
                # Get top relations by importance
                key_relations = sorted(relation_importance.items(), key=lambda x: x[1], reverse=True)
                key_relations = [rel for rel, _ in key_relations[:min(3, len(key_relations))]]
                
                # Store cluster details
                analysis["cluster_details"][cluster_id] = {
                    "size": len(relations),
                    "relations": relations,
                    "key_relations": key_relations,
                    "avg_weight": float(avg_internal_weight),
                    "density": float(density),
                    "isolation": float(isolation)
                }
            
            # Find significant cross-cluster relations
            for cluster1 in range(n_clusters):
                for cluster2 in range(cluster1 + 1, n_clusters):
                    relations1 = cluster_mapping.get(cluster1, [])
                    relations2 = cluster_mapping.get(cluster2, [])
                    
                    # Skip if either cluster is empty
                    if not relations1 or not relations2:
                        continue
                    
                    # Get indices
                    indices1 = [relation_to_idx[rel] for rel in relations1 if rel in relation_to_idx]
                    indices2 = [relation_to_idx[rel] for rel in relations2 if rel in relation_to_idx]
                    
                    # Calculate cross-cluster weights
                    cross_weights = []
                    for idx1 in indices1:
                        for idx2 in indices2:
                            cross_weights.append((idx1, idx2, relation_matrix[idx1, idx2]))
                    
                    # Sort by weight
                    cross_weights.sort(key=lambda x: x[2], reverse=True)
                    
                    # Get top connections
                    top_connections = []
                    for idx1, idx2, weight in cross_weights[:3]:  # Get top 3
                        if weight > 0.2:  # Only strong connections
                            top_connections.append({
                                "relation1": relation_mapping[idx1],
                                "relation2": relation_mapping[idx2],
                                "weight": float(weight)
                            })
                    
                    # Store if there are any strong connections
                    if top_connections:
                        analysis["cross_cluster_relations"].append({
                            "cluster1": cluster1,
                            "cluster2": cluster2,
                            "connections": top_connections,
                            "avg_connection_strength": float(np.mean([c["weight"] for c in top_connections]))
                        })
            
            # Calculate overall statistics
            avg_cluster_size = np.mean([len(rels) for rels in cluster_mapping.values()])
            avg_density = np.mean([details["density"] for details in analysis["cluster_details"].values()])
            avg_isolation = np.mean([details["isolation"] for details in analysis["cluster_details"].values()])
            
            analysis["overall_stats"] = {
                "num_clusters": n_clusters,
                "total_relations": len(relation_mapping),
                "avg_cluster_size": float(avg_cluster_size),
                "avg_density": float(avg_density),
                "avg_isolation": float(avg_isolation),
                "cross_cluster_connections": len(analysis["cross_cluster_relations"])
            }
            
            # Print summary
            logging.info(f"Cluster Analysis Summary:")
            logging.info(f"Number of clusters: {n_clusters}")
            logging.info(f"Average cluster size: {avg_cluster_size:.2f} relations")
            logging.info(f"Average intra-cluster density: {avg_density:.3f}")
            logging.info(f"Average cluster isolation: {avg_isolation:.3f}")
            logging.info(f"Number of significant cross-cluster connections: {len(analysis['cross_cluster_relations'])}")
            
            # Print details for each cluster
            logging.info("\nCluster Details:")
            for cluster_id, details in analysis["cluster_details"].items():
                logging.info(f"\nCluster {cluster_id}:")
                logging.info(f"  Size: {details['size']} relations")
                logging.info(f"  Density: {details['density']:.3f}")
                logging.info(f"  Isolation: {details['isolation']:.3f}")
                if details["key_relations"]:
                    logging.info(f"  Key relations: {', '.join(details['key_relations'])}")
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error during cluster analysis: {e}")
            return {"error": str(e)}
    
    
    def generate_relation_samples(self, relations, k, output_path, openai_api_key=None):
        """
        Generate positive and negative triple samples for specified relations.
        Verify each negative sample using GPT-4o to ensure it's actually incorrect.
        
        Parameters:
            relations (list): List of relation names to sample from
            k (int): Number of samples per relation (will use all available samples if less than k)
            output_path (str): Path to save the resulting JSON file
            openai_api_key (str, optional): OpenAI API key for GPT-4o verification
        
        Returns:
            dict: Dictionary containing positive and negative samples
        """
        if relations is None:
            relations = list(set(rel for _, rel, _ in self.triples['total']))

        # Initialize result structure
        self.threshold_selection_tripes = {
            'relations': relations, 
            'positive': [],  # samples for each relation
            'negative': {},  # Will be a dictionary with three types of negative samples
            'relation': []   # subset of relations for threshold selection
        }
        
        # Initialize negative samples dictionary
        self.threshold_selection_negative_triples = {
            'relation_replaced': [],
            'flipped': [],    # tail-head flipped from original kg
            'covered': [],    # fix relation, random choose from covered head & tail
            'random': []      # fix relation, random choose from any head & tail
        }
        
        # Function to verify if a negative triple is truly incorrect
        def verify_negative_triple(triple):
            """
            Use GPT-4o to verify if a negative sample is truly incorrect.
            
            Parameters:
                triple (tuple): A triple (head, relation, tail)
                
            Returns:
                bool: True if the triple is confirmed incorrect, False if possibly correct
            """
            if not openai_api_key:
                # Skip verification if no API key provided
                return True
                
            head, relation, tail = triple
            
            # Format prompt for GPT
            prompt = (
                "You are a knowledge assessment expert. "
                "Your task is to analyze whether the following triple is factually incorrect. "
                "The triple consists of [head entity, relation, tail entity].\n\n"
                f"Triple: [\"{head}\", \"{relation}\", \"{tail}\"]\n\n"
                "Please analyze this triple and determine if it is indeed factually incorrect. "
                "Consider the entities, their relationship, and whether this statement is reasonable in reality.\n\n"
                "If the triple is factually incorrect, begin your answer with 'INCORRECT'; "
                "if it could be correct, begin with 'POSSIBLY CORRECT'. "
                "Include a brief explanation of your reasoning."
            )
            
            try:
                time.sleep(1)
                # Initialize OpenAI client
                client = openai.OpenAI(api_key=openai_api_key)
                
                # Query GPT-4o
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a knowledge assessment expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=250
                )
                
                # Extract response text
                response_text = response.choices[0].message.content.strip()
                
                # Check if the response indicates the triple is incorrect
                is_incorrect = response_text.upper().startswith("INCORRECT")
                
                if not is_incorrect:
                    logging.info(f"GPT suggests triple may be correct: {triple} - {response_text[:100]}...")
                
                return is_incorrect
                
            except Exception as e:
                logging.warning(f"Error during GPT verification: {e}. Assuming triple is incorrect.")
                return True  # Default to assuming it's incorrect if verification fails
        
        # Create a mapping from relation names to their triples
        relation_to_triples = {}
        for relation in relations:
            relation_to_triples[relation] = []
            
            # Collect all triples for this relation from all splits
            for split in ['train', 'valid', 'test']:
                for h, r, t in self.triples[split]:
                    if r == relation:
                        relation_to_triples[relation].append((h, r, t))
        
        for relation in relations:
            print(relation, len(relation_to_triples[relation]))

        logging.info(f"Processing {len(relations)} relations")
        
        # Create a set of all entities
        all_entities = set()
        for split in ['train', 'valid', 'test']:
            for h, r, t in self.triples[split]:
                all_entities.add(h)
                all_entities.add(t)
        all_entities = list(all_entities)
        
        # Create a set of all existing triples for negative sampling
        all_triples = set()
        for split in ['train', 'valid', 'test']:
            for h, r, t in self.triples[split]:
                all_triples.add((h, r, t))
        
        # For each relation, generate samples
        for relation in relations:
            # Get all triples for this relation
            relation_triples = relation_to_triples.get(relation, [])
            if not relation_triples:
                logging.warning(f"Relation '{relation}' has no triples. Skipping.")
                continue
            
            # Get unique heads and tails for this relation
            relation_heads = set([h for h, r, t in relation_triples])
            relation_tails = set([t for h, r, t in relation_triples])
            
            # Convert to lists for random sampling
            relation_heads = list(relation_heads)
            relation_tails = list(relation_tails)
            
            # Determine sample size for this relation
            # If we have fewer samples than k, use all of them
            # Otherwise, sample k triples
            relation_sample_size = len(relation_triples)
            if relation_sample_size >= k:
                relation_sample_size = k
                positive_samples = random.sample(relation_triples, relation_sample_size)
            else:
                # Use all available samples
                positive_samples = relation_triples
            
            logging.info(f"Using {len(positive_samples)} positive samples for relation '{relation}'")
            
            # Add to results
            self.threshold_selection_tripes['positive'].extend(positive_samples)
            
            # Add to relation subset (format: (relation, head, tail))
            for h, r, t in positive_samples:
                self.threshold_selection_tripes['relation'].append((r, h, t))
            
            # Generate negative samples
            
            # 1. Replace relation samples (keep head and tail, replace relation)
            relation_replaced_samples = []
            # Get all relations except the current one
            other_relations = [r for r in relations if r != relation]
            # If not enough relations in selected list, add more from all available relations
            if len(other_relations) < 3:
                all_available_relations = set()
                for split in ['train', 'valid', 'test']:
                    for _, r, _ in self.triples[split]:
                        if r != relation:  # Exclude current relation
                            all_available_relations.add(r)
                other_relations.extend([r for r in all_available_relations if r not in other_relations])
            
            attempts = 0
            while len(relation_replaced_samples) < relation_sample_size and attempts < relation_sample_size * 40:  # Increased attempts
                # Take a positive sample and replace its relation
                if not positive_samples:  # Safety check
                    break
                    
                # Select a positive sample
                h, _, t = random.choice(positive_samples)
                
                # Replace relation with a random different relation
                new_relation = random.choice(other_relations)
                triple = (h, new_relation, t)
                
                # Check if this triple exists in the dataset
                if triple not in all_triples and triple not in relation_replaced_samples:
                    # Verify with GPT-4o
                    if verify_negative_triple(triple):
                        relation_replaced_samples.append(triple)
                
                attempts += 1
                
                # If after many attempts we still can't find enough samples, break
                if attempts >= relation_sample_size * 40 and relation_replaced_samples:
                    logging.warning(f"Could only generate {len(relation_replaced_samples)} 'relation replaced' negative samples for relation '{relation}' after {attempts} attempts.")
                    break
            
            self.threshold_selection_negative_triples['relation_replaced'].extend(relation_replaced_samples[:relation_sample_size])

            """ # 2. Flipped samples (randomly select k different triples from all relation triples and flip head-tail)
            flipped_candidates = relation_to_triples[relation]
            all_relation_triples = relation_to_triples[relation]
                
            # Try to sample k triples for flipping (might sample more for higher success rate)
            sample_size = min(len(flipped_candidates), relation_sample_size * 5)  # Increased sample size
            flipped_candidates = random.sample(flipped_candidates, sample_size)
            
            flipped_samples = []
            for h, r, t in flipped_candidates:
                flipped = (t, relation, h)  # Flip head and tail
                if flipped not in all_triples and flipped not in flipped_samples:
                    # Verify with GPT-4o
                    if verify_negative_triple(flipped):
                        flipped_samples.append(flipped)
                        if len(flipped_samples) >= relation_sample_size:
                            break

            # If still not enough, try some more random flips
            if len(flipped_samples) < relation_sample_size:
                logging.warning(f"Could only generate {len(flipped_samples)} 'flipped' negative samples for relation '{relation}'. Attempting more...")
                attempts = 0
                while len(flipped_samples) < relation_sample_size and attempts < relation_sample_size * 40:  # Increased attempts
                    h, r, t = random.choice(all_relation_triples)
                    flipped = (t, relation, h)
                    if flipped not in all_triples and flipped not in flipped_samples and flipped not in relation_replaced_samples:
                        # Verify with GPT-4o
                        if verify_negative_triple(flipped):
                            flipped_samples.append(flipped)
                    attempts += 1
            
            self.threshold_selection_negative_triples['flipped'].extend(flipped_samples[:relation_sample_size])
            
            # 3. Random samples from covered heads and tails
            covered_samples = []
            
            attempts = 0
            while len(covered_samples) < relation_sample_size and attempts < relation_sample_size * 40:  # Increased attempts
                # Random head and tail from covered entities
                h = random.choice(relation_heads)
                t = random.choice(relation_tails)
                triple = (h, relation, t)
                
                # Check if this triple exists in the dataset
                if triple not in all_triples and triple not in covered_samples and triple not in flipped_samples:
                    # Verify with GPT-4o
                    if verify_negative_triple(triple):
                        covered_samples.append(triple)
                
                attempts += 1
                
                # If after many attempts we still can't find enough samples, break
                if attempts >= relation_sample_size * 40 and covered_samples:
                    logging.warning(f"Could only generate {len(covered_samples)} 'covered' negative samples for relation '{relation}' after {attempts} attempts.")
                    break
            
            self.threshold_selection_negative_triples['covered'].extend(covered_samples[:relation_sample_size])
            
            # 4. Random samples from all entities
            random_samples = []
            
            attempts = 0
            while len(random_samples) < relation_sample_size and attempts < relation_sample_size * 40:  # Increased attempts
                # Random head and tail from all entities
                h = random.choice(all_entities)
                t = random.choice(all_entities)
                triple = (h, relation, t)
                
                # Check if this triple exists in the dataset
                if triple not in all_triples and triple not in random_samples and triple not in covered_samples and triple not in flipped_samples and triple not in relation_replaced_samples:
                    # Verify with GPT-4o
                    if verify_negative_triple(triple):
                        random_samples.append(triple)
                
                attempts += 1
                
                # If after many attempts we still can't find enough samples, break
                if attempts >= relation_sample_size * 40 and random_samples:
                    logging.warning(f"Could only generate {len(random_samples)} 'random' negative samples for relation '{relation}' after {attempts} attempts.")
                    break
            
            self.threshold_selection_negative_triples['random'].extend(random_samples[:relation_sample_size]) """
        
        # Add negative triples to main dictionary
        self.threshold_selection_tripes['negative'] = self.threshold_selection_negative_triples
        
        # Save to JSON
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.threshold_selection_tripes, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved {len(self.threshold_selection_tripes['positive'])} positive samples")
        logging.info(f"Saved {len(self.threshold_selection_negative_triples['relation_replaced'])} relation_replaced negative samples")
        logging.info(f"Saved {len(self.threshold_selection_negative_triples['flipped'])} flipped negative samples")
        logging.info(f"Saved {len(self.threshold_selection_negative_triples['covered'])} covered negative samples")
        logging.info(f"Saved {len(self.threshold_selection_negative_triples['random'])} random negative samples")
        logging.info(f"Output saved to {output_path}")
        
        # Count verified samples
        total_verified = (len(self.threshold_selection_negative_triples['relation_replaced']) + 
                        len(self.threshold_selection_negative_triples['flipped']) +
                        len(self.threshold_selection_negative_triples['covered']) +
                        len(self.threshold_selection_negative_triples['random']))
        
        logging.info(f"Total negative samples verified with GPT-4o: {total_verified}")
        
        return self.threshold_selection_tripes


    def create_diverse_subgraphs(self, num_subgraphs=1, allowed_relations=None, min_nodes=10, 
                             min_edges_per_node=2, path_length=3, output_dir=None, seed=None):
        """
        Create diverse subgraphs that show correlations between entities.
        
        Args:
            num_subgraphs (int): Number of subgraphs to create
            allowed_relations (list): List of relation names to use for building the subgraph
                                    (if None, all relations are allowed)
            min_nodes (int): Minimum number of nodes in each subgraph
            min_edges_per_node (int): Minimum number of edges for each node
            path_length (int): Desired path length (2-3 recommended)
            output_dir (str): Directory to save visualizations and JSON files 
                            (if None, don't save)
            seed (int): Random seed for reproducibility (if None, no seed is set)
        
        Returns:
            list: List of subgraphs, where each subgraph is a dictionary with 
                'nodes', 'edges', and 'triples' keys
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            import json
            import os
            import random
            from collections import defaultdict, Counter
        except ImportError as e:
            raise ImportError(f"Required package missing: {e}. Please install networkx and matplotlib.")
        
        # Set seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Ensure adjacency list is built
        if self.adj_list is None:
            self.build_adjacency_list()
        
        # If no output directory specified but saving is requested, create one
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Filter allowed relations if specified
        filtered_triples = []
        for head, relation, tail in self.triples['total']:
            if allowed_relations is None or relation in allowed_relations:
                filtered_triples.append((head, relation, tail))
        
        if not filtered_triples:
            raise ValueError("No triples found with the specified relations")
        
        # Count occurrences of each entity to find important entities
        entity_counts = Counter()
        for head, _, tail in filtered_triples:
            entity_counts[head] += 1
            entity_counts[tail] += 1
        
        # Build a directed graph from the triples
        G = nx.DiGraph()
        for head, relation, tail in filtered_triples:
            G.add_edge(head, tail, relation=relation)
        
        # List to store all generated subgraphs
        all_subgraphs = []
        
        for sg_idx in range(num_subgraphs):
            attempts = 0
            max_attempts = 100  # Limit the number of attempts
            
            while attempts < max_attempts:
                attempts += 1
                
                # Find important entities to connect
                important_entities = [e for e, c in entity_counts.most_common(100) if e in G]
                if len(important_entities) < 2:
                    important_entities.extend(list(G.nodes())[:100-len(important_entities)])
                
                if len(important_entities) < 2:
                    print(f"Warning: Not enough entities in the graph for subgraph {sg_idx}")
                    break
                
                # Randomly select two distinct important entities
                source = random.choice(important_entities)
                important_entities.remove(source)
                target = random.choice(important_entities)
                
                # Find all simple paths up to path_length between source and target
                # Limit to 100 paths to avoid explosion in dense graphs
                all_paths = []
                try:
                    for path in nx.all_simple_paths(G, source, target, cutoff=path_length):
                        if len(all_paths) >= 100:
                            break
                        all_paths.append(path)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                
                if not all_paths:
                    continue  # No paths found
                
                # Select diverse paths (paths with different relations)
                selected_paths = []
                used_relations = set()
                
                # Shuffle paths to increase diversity
                random.shuffle(all_paths)
                
                for path in all_paths:
                    path_relations = set()
                    for i in range(len(path) - 1):
                        relation = G[path[i]][path[i+1]]['relation']
                        path_relations.add(relation)
                    
                    # If this path adds new relations, select it
                    if path_relations - used_relations:
                        selected_paths.append(path)
                        used_relations.update(path_relations)
                    
                    # If we have enough diverse paths, stop
                    if len(selected_paths) >= 5:  # Aim for 5 diverse paths
                        break
                
                if not selected_paths:
                    continue  # No diverse paths found
                
                # Convert selected paths to a subgraph
                subgraph_nodes = set()
                subgraph_edges = []
                
                for path in selected_paths:
                    for i in range(len(path)):
                        subgraph_nodes.add(path[i])
                        if i < len(path) - 1:
                            relation = G[path[i]][path[i+1]]['relation']
                            edge = (path[i], path[i+1], relation)
                            if edge not in subgraph_edges:
                                subgraph_edges.append(edge)
                
                # Check if subgraph meets the requirements
                if len(subgraph_nodes) < min_nodes:
                    # Add more nodes to reach minimum
                    candidate_nodes = set(G.nodes()) - subgraph_nodes
                    if candidate_nodes:
                        # Prioritize nodes that connect to existing subgraph nodes
                        connected_candidates = []
                        for node in candidate_nodes:
                            for sg_node in subgraph_nodes:
                                if G.has_edge(node, sg_node) or G.has_edge(sg_node, node):
                                    connected_candidates.append(node)
                                    break
                        
                        # Add connected nodes first, then random nodes if needed
                        random.shuffle(connected_candidates)
                        while len(subgraph_nodes) < min_nodes and connected_candidates:
                            new_node = connected_candidates.pop()
                            subgraph_nodes.add(new_node)
                            
                            # Add edges to this new node
                            for sg_node in list(subgraph_nodes - {new_node}):
                                if G.has_edge(new_node, sg_node):
                                    relation = G[new_node][sg_node]['relation']
                                    edge = (new_node, sg_node, relation)
                                    if edge not in subgraph_edges:
                                        subgraph_edges.append(edge)
                                if G.has_edge(sg_node, new_node):
                                    relation = G[sg_node][new_node]['relation']
                                    edge = (sg_node, new_node, relation)
                                    if edge not in subgraph_edges:
                                        subgraph_edges.append(edge)
                    
                        # If still need more nodes, add random ones
                        remaining_candidates = list(candidate_nodes - subgraph_nodes)
                        random.shuffle(remaining_candidates)
                        while len(subgraph_nodes) < min_nodes and remaining_candidates:
                            new_node = remaining_candidates.pop()
                            subgraph_nodes.add(new_node)
                            
                            # Find connections to existing subgraph
                            for sg_node in list(subgraph_nodes - {new_node}):
                                if G.has_edge(new_node, sg_node):
                                    relation = G[new_node][sg_node]['relation']
                                    edge = (new_node, sg_node, relation)
                                    if edge not in subgraph_edges:
                                        subgraph_edges.append(edge)
                                if G.has_edge(sg_node, new_node):
                                    relation = G[sg_node][new_node]['relation']
                                    edge = (sg_node, new_node, relation)
                                    if edge not in subgraph_edges:
                                        subgraph_edges.append(edge)
                
                # Check if each node has at least min_edges_per_node edges
                node_edge_count = defaultdict(int)
                for s, t, _ in subgraph_edges:
                    node_edge_count[s] += 1
                    node_edge_count[t] += 1
                
                # If some nodes don't have enough edges, add more
                nodes_needing_edges = [n for n in subgraph_nodes if node_edge_count[n] < min_edges_per_node]
                if nodes_needing_edges:
                    for node in nodes_needing_edges:
                        # Try to add edges until this node has enough
                        missing_edges = min_edges_per_node - node_edge_count[node]
                        potential_connections = list(G.nodes())
                        random.shuffle(potential_connections)
                        
                        for pot_node in potential_connections:
                            if missing_edges <= 0:
                                break
                                
                            if G.has_edge(node, pot_node):
                                relation = G[node][pot_node]['relation']
                                edge = (node, pot_node, relation)
                                if edge not in subgraph_edges:
                                    subgraph_edges.append(edge)
                                    node_edge_count[node] += 1
                                    node_edge_count[pot_node] += 1
                                    subgraph_nodes.add(pot_node)
                                    missing_edges -= 1
                                
                            elif G.has_edge(pot_node, node):
                                relation = G[pot_node][node]['relation']
                                edge = (pot_node, node, relation)
                                if edge not in subgraph_edges:
                                    subgraph_edges.append(edge)
                                    node_edge_count[node] += 1
                                    node_edge_count[pot_node] += 1
                                    subgraph_nodes.add(pot_node)
                                    missing_edges -= 1
                
                # Check final criteria
                node_edge_count = defaultdict(int)
                for s, t, _ in subgraph_edges:
                    node_edge_count[s] += 1
                    node_edge_count[t] += 1
                    
                all_nodes_have_enough_edges = all(node_edge_count[n] >= min_edges_per_node for n in subgraph_nodes)
                
                if len(subgraph_nodes) >= min_nodes and all_nodes_have_enough_edges:
                    # Success: Create the subgraph
                    subgraph_triples = [(s, r, t) for s, t, r in subgraph_edges]
                    
                    subgraph = {
                        'id': sg_idx,
                        'nodes': list(subgraph_nodes),
                        'edges': subgraph_edges,
                        'triples': subgraph_triples,
                        'source': source,
                        'target': target,
                        'paths': selected_paths
                    }
                    
                    # Visualization
                    if output_dir:
                        self._visualize_subgraph(subgraph, output_dir, sg_idx)
                        self._save_subgraph_json(subgraph, output_dir, sg_idx)
                    
                    all_subgraphs.append(subgraph)
                    break  # Successfully created this subgraph
                    
            if attempts >= max_attempts:
                print(f"Warning: Could not create subgraph {sg_idx} after {max_attempts} attempts")
        
        if not all_subgraphs:
            print("Warning: No subgraphs were created. Try adjusting parameters or using a different dataset.")
        
        return all_subgraphs

    def _visualize_subgraph(self, subgraph, output_dir, sg_idx):
        """
        Visualize a subgraph using networkx and matplotlib.
        
        Args:
            subgraph (dict): The subgraph to visualize
            output_dir (str): Directory to save the visualization
            sg_idx (int): Index of the subgraph
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            import os
        except ImportError as e:
            print(f"Warning: Visualization skipped due to missing package: {e}")
            return
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in subgraph['nodes']:
            G.add_node(node)
        
        # Add edges with relation as edge attribute
        for src, dst, rel in subgraph['edges']:
            G.add_edge(src, dst, relation=rel)
        
        # Create the plot
        plt.figure(figsize=(15, 12))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes with different colors for source and target
        source_node = subgraph['source']
        target_node = subgraph['target']
        other_nodes = [n for n in subgraph['nodes'] if n != source_node and n != target_node]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=[source_node], node_size=800, node_color='lightgreen')
        nx.draw_networkx_nodes(G, pos, nodelist=[target_node], node_size=800, node_color='lightcoral')
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_size=600, node_color='lightblue')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1, arrows=True, arrowsize=15)
        
        # Prepare node labels (truncate long entity names)
        max_label_len = 20
        node_labels = {}
        for node in G.nodes():
            label = str(node)
            if len(label) > max_label_len:
                label = label[:max_label_len-3] + "..."
            node_labels[node] = label
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_family='sans-serif')
        
        # Prepare edge labels (truncate long relation names)
        edge_labels = {}
        for src, dst, data in G.edges(data=True):
            label = str(data['relation'])
            if len(label) > max_label_len:
                label = label[:max_label_len-3] + "..."
            edge_labels[(src, dst)] = label
        
        # Draw edge labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        
        plt.title(f"Subgraph {sg_idx} - Path from '{source_node}' to '{target_node}'")
        plt.axis('off')
        
        # Save to file
        output_file = os.path.join(output_dir, f"subgraph_{sg_idx}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_file}")

    def _save_subgraph_json(self, subgraph, output_dir, sg_idx):
        """
        Save a subgraph as a JSON file.
        
        Args:
            subgraph (dict): The subgraph to save
            output_dir (str): Directory to save the JSON file
            sg_idx (int): Index of the subgraph
        """
        try:
            import json
            import os
        except ImportError as e:
            print(f"Warning: JSON saving skipped due to missing package: {e}")
            return
        
        # Create a JSON-friendly copy of the subgraph
        json_subgraph = {
            'id': subgraph['id'],
            'source': subgraph['source'],
            'target': subgraph['target'],
            'nodes': list(subgraph['nodes']),
            'triples': subgraph['triples'],
            'paths': [list(path) for path in subgraph['paths']]
        }
        
        # Save to file
        output_file = os.path.join(output_dir, f"subgraph_{sg_idx}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_subgraph, f, ensure_ascii=False, indent=4)
        
        print(f"JSON saved to {output_file}")
    

    def save_statistics(self, stats, file_path):
        """
        Save computed statistics to a JSON file.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)

    def print_statistics(self):
        """
        Print basic knowledge graph statistics.
        """
        logging.info(f"Knowledge Graph: {self.name}")
        logging.info(f"Total triples: {len(self.triples['total'])}")
        logging.info(f"Total unique entities: {len(self.entities)}")
        logging.info(f"Total unique relations: {len(self.relations)}")
        
    def diagnose_graph_issues(self, allowed_relations=None):
        """
        Diagnose issues that might prevent finding valid subgraphs.
        
        Args:
            allowed_relations: List of relations to use (None for all relations)
            
        Returns:
            dict: Diagnostic information about the graph
        """
        # Build graph with filtered relations
        G = nx.DiGraph()
        if allowed_relations is None:
            allowed_relations = list(set(rel for _, rel, _ in self.triples['total']))
        
        filtered_triples = [(h, r, t) for h, r, t in self.triples['total'] if r in allowed_relations]
        for head, rel, tail in filtered_triples:
            G.add_edge(head, tail, relation=rel)
        
        # Calculate basic graph statistics
        stats = {
            "num_nodes": len(G.nodes()),
            "num_edges": len(G.edges()),
            "num_relations": len(allowed_relations),
            "density": nx.density(G),
        }
        
        # Check connectivity
        components = list(nx.weakly_connected_components(G))
        stats["num_components"] = len(components)
        stats["largest_component_size"] = len(max(components, key=len))
        stats["largest_component_pct"] = stats["largest_component_size"] / stats["num_nodes"] * 100
        
        # Analyze degree distribution
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        total_degrees = [d_in + d_out for (_, d_in), (_, d_out) in zip(G.in_degree(), G.out_degree())]
        
        stats["avg_in_degree"] = sum(in_degrees) / len(in_degrees) if in_degrees else 0
        stats["avg_out_degree"] = sum(out_degrees) / len(out_degrees) if out_degrees else 0
        stats["avg_total_degree"] = sum(total_degrees) / len(total_degrees) if total_degrees else 0
        
        stats["nodes_with_in_degree_gt_0"] = sum(1 for d in in_degrees if d > 0)
        stats["nodes_with_out_degree_gt_0"] = sum(1 for d in out_degrees if d > 0)
        stats["nodes_with_total_degree_gte_2"] = sum(1 for d in total_degrees if d >= 2)
        
        stats["pct_nodes_with_total_degree_gte_2"] = stats["nodes_with_total_degree_gte_2"] / stats["num_nodes"] * 100
        
        # Check existence of bridge nodes (nodes with both in and out edges)
        bridge_nodes = [node for node in G.nodes() 
                    if G.in_degree(node) > 0 and G.out_degree(node) > 0]
        
        stats["num_bridge_nodes"] = len(bridge_nodes)
        stats["pct_bridge_nodes"] = (len(bridge_nodes) / stats["num_nodes"]) * 100 if stats["num_nodes"] > 0 else 0
        
        # Check path existence between high-degree nodes
        high_out_degree_nodes = [n for n, d in G.out_degree() if d >= 1]
        high_in_degree_nodes = [n for n, d in G.in_degree() if d >= 1]
        
        # Sample paths between nodes with sufficient degree
        paths_found = 0
        path_lengths = []
        
        if high_out_degree_nodes and high_in_degree_nodes:
            samples = min(10, len(high_out_degree_nodes) * len(high_in_degree_nodes))
            pairs_tested = 0
            
            for _ in range(samples):
                start = random.choice(high_out_degree_nodes)
                end = random.choice(high_in_degree_nodes)
                
                if start != end:
                    pairs_tested += 1
                    try:
                        # Check if a path exists
                        path = nx.shortest_path(G, start, end)
                        paths_found += 1
                        path_lengths.append(len(path) - 1)
                    except nx.NetworkXNoPath:
                        pass
            
            stats["path_test_pairs"] = pairs_tested
            stats["path_test_success_rate"] = paths_found / pairs_tested if pairs_tested else 0
            stats["avg_path_length"] = sum(path_lengths) / len(path_lengths) if path_lengths else 0
        
        return stats

    def create_diverse_random_walk_subgraph(self, n_triples, allowed_relations=None, output_dir=None, openai_api_key=None):
        """
        Creates a subgraph by random walk from a diverse entity and uses GPT-4o to identify correlated triples.
        
        Args:
            n_triples (int): Number of triples to include in the subgraph
            allowed_relations (list): List of relation names to use for building the subgraph
                                    (if None, all relations are allowed)
            output_dir (str): Directory to save visualizations and JSON files 
                            (if None, don't save)
            openai_api_key (str): OpenAI API key for using GPT-4o to analyze triples
            
        Returns:
            tuple: (original_subgraph, correlated_subgraph) - dictionaries containing the original
                and correlated triples respectively
        """
        
        # Ensure output directory exists
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Filter allowed relations if specified
        if allowed_relations is None:
            allowed_relations = list(set(relation for _, relation, _ in self.triples['total']))
        
        # Count relation diversity for each entity (focusing on outgoing relations)
        entity_relation_diversity = defaultdict(set)
        for head, relation, tail in self.triples['total']:
            if relation in allowed_relations:
                entity_relation_diversity[head].add(relation)
        
        # Find the entity with most diverse relations
        if not entity_relation_diversity:
            raise ValueError("No triples found with the specified relations")
        
        most_diverse_entity = max(entity_relation_diversity.items(), key=lambda x: len(x[1]))[0]
        print(f"Starting random walk from entity: {most_diverse_entity} with {len(entity_relation_diversity[most_diverse_entity])} different relation types")
        
        # Build a directed graph
        G = nx.DiGraph()
        for head, relation, tail in self.triples['total']:
            if relation in allowed_relations:
                G.add_edge(head, tail, relation=relation)
        
        # Perform random walk
        subgraph_triples = []
        current_entity = most_diverse_entity
        visited_entities = {current_entity}
        
        while len(subgraph_triples) < n_triples:
            # Get all outgoing edges from current entity
            outgoing_edges = list(G.out_edges(current_entity, data=True))
            if not outgoing_edges:
                # If no outgoing edges, try starting from another entity we've visited
                unprocessed_entities = [e for e in visited_entities if list(G.out_edges(e, data=True))]
                if not unprocessed_entities:
                    break
                current_entity = random.choice(unprocessed_entities)
                continue
            
            # Choose a random outgoing edge
            head, tail, data = random.choice(outgoing_edges)
            relation = data['relation']
            
            # Add this triple to our subgraph
            triple = (head, relation, tail)
            if triple not in subgraph_triples:
                subgraph_triples.append(triple)
            
            # Move to the next entity
            current_entity = tail
            visited_entities.add(current_entity)
            
            # If we've exhausted all possible triples, break
            if len(subgraph_triples) >= n_triples or len(subgraph_triples) >= len(G.edges()):
                break
        
        # If we couldn't get enough triples from random walk, supplement with additional edges
        if len(subgraph_triples) < n_triples:
            # Get all edges that have at least one endpoint in our visited entities
            candidate_edges = []
            for head, relation, tail in self.triples['total']:
                if relation in allowed_relations and (head in visited_entities or tail in visited_entities):
                    triple = (head, relation, tail)
                    if triple not in subgraph_triples:
                        candidate_edges.append(triple)
            
            # Add random edges until we reach n_triples or run out of candidates
            random.shuffle(candidate_edges)
            subgraph_triples.extend(candidate_edges[:n_triples - len(subgraph_triples)])
        
        print(f"Collected {len(subgraph_triples)} triples for the subgraph")
        
        # Define visualization function
        def _visualize_diverse_walk_subgraph(triples, filename, title="Subgraph from Random Walk"):
            """Helper function to visualize the subgraph with colors per relation"""
            G = nx.DiGraph()
            
            # Add edges
            relation_types = set()
            for head, relation, tail in triples:
                G.add_edge(head, tail, relation=relation)
                relation_types.add(relation)
            
            # Remove isolated nodes
            G.remove_nodes_from(list(nx.isolates(G)))
            
            if len(G.nodes()) == 0:
                print("No connected nodes to visualize")
                return
            
            # Create color map for relations
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            
            # Use a colormap that gives distinct colors
            cmap = cm.get_cmap('tab20', max(len(relation_types), 1))
            relation_colors = {rel: mcolors.rgb2hex(cmap(i)) for i, rel in enumerate(sorted(relation_types))}
            
            # Create plot
            plt.figure(figsize=(20, 16))
            
            # Use a layout that spreads nodes evenly and minimizes overlap
            try:
                # First try Kamada-Kawai layout for better distribution
                pos = nx.kamada_kawai_layout(G)
            except:
                try:
                    # Fall back to spring layout with strong repulsion
                    pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G.nodes())), iterations=50, seed=42)
                except:
                    # Last resort: basic spring layout
                    pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.8)
            
            # Draw edges with colors based on relation
            for relation in relation_types:
                # Extract edges of this relation type
                edges = [(u, v) for u, v, d in G.edges(data=True) if d['relation'] == relation]
                if edges:
                    nx.draw_networkx_edges(
                        G, pos, 
                        edgelist=edges,
                        width=2,
                        alpha=0.7,
                        edge_color=relation_colors[relation],
                        arrows=True,
                        arrowsize=20
                    )
            
            # Draw labels with truncated text if too long
            labels = {}
            for node in G.nodes():
                if len(str(node)) > 30:  # Truncate long entity names
                    labels[node] = str(node)[:27] + "..."
                else:
                    labels[node] = str(node)
            
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_family='sans-serif')
            
            # Create a legend for relation types
            patches = [mpatches.Patch(color=relation_colors[rel], label=rel) for rel in sorted(relation_types)]
            plt.legend(handles=patches, loc='upper right', fontsize=10)
            
            plt.title(title, fontsize=16)
            plt.axis('off')
            
            # Adjust the plot margins
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Visualization saved to {filename}")
        
        # Define GPT-4o analysis function
        def _analyze_with_gpt4o(triples, api_key):
            """Use GPT-4o to identify correlated triples within the given set."""
            try:
                import openai
                
                if not api_key:
                    raise ValueError("OpenAI API key is required to use GPT-4o analysis")
                
                client = openai.OpenAI(api_key=api_key)
                
                # Format triples as text for GPT-4o
                triples_text = "\n".join([f"({head}, {relation}, {tail})" for head, relation, tail in triples])
                
                # Construct the prompt
                prompt = f"""Given the following triples from a knowledge graph, identify sets of correlated triples.
                            Correlated triples are those that have logical or inferential relationships between them.
                            For example, if we have (A, relation1, B), (B, relation2, C), and (A, relation3, C), these three triples might be correlated
                            because they form a pattern where the third relationship might be inferred from the first two.

                            Here are the triples:
                            {triples_text}

                            Please:
                            1. Identify groups of triples that are correlated with each other
                            2. Explain your reasoning for each identified correlation
                            3. Return the correlated triples in the same format they were provided
                            4. Focus on finding meaningful patterns rather than surface-level connections

                            Format your response as:
                            ---
                            # Correlated Triples

                            ## Group 1
                            [list of correlated triples in (head, relation, tail) format]

                            ### Reasoning
                            [Your explanation for why these triples are correlated]

                            ## Group 2
                            [list of correlated triples in (head, relation, tail) format]

                            ### Reasoning
                            [Your explanation for why these triples are correlated]

                            And so on for additional groups...
                            ---

                            Think step by step and be thorough in your analysis."""

                # Make the API call to GPT-4o
                response = client.chat.completions.create(
                    model="gpt-4o",  # Use gpt-4o
                    messages=[
                        {"role": "system", "content": "You are a knowledge graph expert specializing in finding correlations and inference patterns in triples."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4096
                )
                
                # Extract the response
                gpt_response = response.choices[0].message.content
                
                # Parse the response to extract the correlated triples
                correlated_triples = []
                reasoning = gpt_response
                
                # Basic parsing logic for the GPT-4o response
                import re
                
                # Find all triple patterns in the response
                triple_pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
                found_triples = re.findall(triple_pattern, gpt_response)
                
                # Convert to the right format
                for triple_match in found_triples:
                    head, relation, tail = [part.strip() for part in triple_match]
                    correlated_triples.append((head, relation, tail))
                
                return correlated_triples, reasoning
                
            except Exception as e:
                print(f"Error using GPT-4o: {str(e)}")
                return [], f"Error: {str(e)}"
        
        # Create the original subgraph visualization and save as JSON
        original_subgraph = {
            'triples': subgraph_triples,
            'starting_entity': most_diverse_entity
        }
        
        if output_dir:
            # Save original subgraph as JSON
            with open(os.path.join(output_dir, 'original_subgraph.json'), 'w', encoding='utf-8') as f:
                json.dump(original_subgraph, f, indent=2, ensure_ascii=False)
            
            # Visualize original subgraph
            _visualize_diverse_walk_subgraph(
                subgraph_triples, 
                os.path.join(output_dir, 'original_subgraph.png'),
                title=f"Random Walk Subgraph (Starting from {most_diverse_entity})"
            )
        
        # Use GPT-4o to identify correlated triples if API key is provided
        correlated_subgraph = {'triples': [], 'reasoning': ''}
        
        if openai_api_key:
            print("Analyzing triples with GPT-4o to find correlations...")
            correlated_triples, reasoning = _analyze_with_gpt4o(subgraph_triples, openai_api_key)
            
            correlated_subgraph = {
                'triples': correlated_triples,
                'reasoning': reasoning
            }
            
            if output_dir:
                # Save correlated subgraph as JSON
                with open(os.path.join(output_dir, 'correlated_subgraph.json'), 'w', encoding='utf-8') as f:
                    json.dump(correlated_subgraph, f, indent=2, ensure_ascii=False)
                
                # Visualize correlated subgraph
                if correlated_triples:
                    _visualize_diverse_walk_subgraph(
                        correlated_triples, 
                        os.path.join(output_dir, 'correlated_subgraph.png'),
                        title="Correlated Triples Identified by GPT-4o"
                    )
                    print(f"Found {len(correlated_triples)} correlated triples")
                else:
                    print("No correlated triples found")
        
        return original_subgraph, correlated_subgraph


    def extract_relation_and_khop_triples(self, relation_list, k, output_file=None):
        """
        Extracts triples containing specified relations and their k-hop neighborhood triples.
        
        Args:
            relation_list (list): List of relation names to filter by
            k (int): Number of hops to expand from entities in the filtered triples
            output_file (str): Path to save the JSON output (if None, doesn't save)
            
        Returns:
            tuple: (filtered_triples, khop_triples, combined_triples) - Lists of triples
                filtered by relations, their k-hop neighbors, and the combined unique set
        """

        # 1. Extract triples with the specified relations (triples list 1)
        filtered_triples = []
        count_for_relations = {}
        for relation in relation_list:
            count_for_relations[relation] = 0

        for head, relation, tail in self.triples['total']:
            if relation in relation_list:
                filtered_triples.append((head, relation, tail))
                count_for_relations[relation] += 1
        
        for relation in relation_list:
            print(relation, count_for_relations[relation])

        print(f"Found {len(filtered_triples)} triples with specified relations")
        
        # 2. Identify all entities involved in these triples
        entities_of_interest = set()
        for head, _, tail in filtered_triples:
            entities_of_interest.add(head)
            entities_of_interest.add(tail)
        
        print(f"Found {len(entities_of_interest)} entities involved in those triples")
        
        # 3. For each entity, find all k-hop neighbors
        all_khop_entities = set(entities_of_interest)  # Start with the original entities
        for entity in entities_of_interest:
            khop_neighbors = self.get_k_hop_neighbors(entity, k=k)
            all_khop_entities.update(khop_neighbors)
        
        print(f"Found {len(all_khop_entities)} entities within {k}-hop neighborhood")
        
        # 4. Collect all triples involving any of these entities
        khop_triples = []
        for head, relation, tail in self.triples['total']:
            if (head in all_khop_entities or tail in all_khop_entities) and (head, relation, tail) not in filtered_triples:
                khop_triples.append((head, relation, tail))
        
        print(f"Found {len(khop_triples)} additional triples in {k}-hop neighborhood")
        
        # 5. Combine both lists (no need to deduplicate as we already ensured no duplicates)
        combined_triples = filtered_triples + khop_triples
        
        # 6. Save as JSON if output_file is provided
        if output_file:
            # Convert triples to a serializable format (list of lists)
            output_data = {
                'filtered_triples': [list(t) for t in filtered_triples],
                'khop_triples': [list(t) for t in khop_triples],
                'combined_triples': [list(t) for t in combined_triples]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved {len(combined_triples)} triples to {output_file}")
        
        return filtered_triples, khop_triples, combined_triples



    def extract_subgraph(self, output_dir, size_param=0.1, seed=42, method="mincut"):
        """
        Extract a subgraph from the knowledge graph using the specified method and save it
        in the same format as the original graph.
        
        Args:
            output_dir (str): Directory to save the extracted subgraph.
            size_param (float): Hyperparameter to control the size of the subgraph (0-1).
                            This represents the approximate fraction of the original graph to keep.
            seed (int): Random seed for reproducibility.
            method (str): Method to extract subgraph: 'mincut' or 'k_hop'.
        
        Returns:
            KnowledgeGraph: A new KnowledgeGraph object containing the subgraph.
        """
        
        # Make sure adjacency list is built
        if self.adj_list is None:
            self.build_adjacency_list()
        
        # Create a NetworkX graph from the adjacency list
        G = nx.Graph()
        for entity in self.adj_list:
            G.add_node(entity)
            for _, neighbor in self.adj_list[entity]:
                G.add_edge(entity, neighbor)
        
        # Get target size
        entities_list = list(G.nodes)
        target_entities = max(10, int(len(entities_list) * size_param))
        
        # Extract subgraph entities based on the chosen method
        subgraph_entities = set()
        
        if method == "mincut":
            try:
                np.random.seed(seed)
                
                # Function to safely handle disconnected components
                def extract_with_mincut():
                    # Find the largest connected component
                    largest_cc = max(nx.connected_components(G), key=len)
                    
                    # If the largest connected component is too small, use k_hop instead
                    if len(largest_cc) < target_entities:
                        logging.warning("Largest connected component too small. Using k_hop method.")
                        return set()
                    
                    # Create a subgraph for the largest connected component
                    largest_subgraph = G.subgraph(largest_cc)
                    
                    # Convert to adjacency matrix
                    nodes_list = list(largest_subgraph.nodes())
                    adj_matrix = nx.adjacency_matrix(largest_subgraph, nodelist=nodes_list)
                    
                    # Apply spectral clustering for approximating mincut
                    clustering = SpectralClustering(
                        n_clusters=2,  # Binary partition
                        assign_labels="discretize",
                        random_state=seed,
                        affinity="precomputed"
                    ).fit(adj_matrix)
                    
                    labels = clustering.labels_
                    
                    # Get the partitions
                    partition0 = [nodes_list[i] for i in range(len(nodes_list)) if labels[i] == 0]
                    partition1 = [nodes_list[i] for i in range(len(nodes_list)) if labels[i] == 1]
                    
                    # Choose the partition closer to the target size
                    if abs(len(partition0) - target_entities) <= abs(len(partition1) - target_entities):
                        if len(partition0) <= target_entities:
                            return set(partition0)
                        else:
                            return set(random.sample(partition0, target_entities))
                    else:
                        if len(partition1) <= target_entities:
                            return set(partition1)
                        else:
                            return set(random.sample(partition1, target_entities))
                
                subgraph_entities = extract_with_mincut()
                
                # If mincut failed, fall back to k_hop
                if not subgraph_entities:
                    method = "k_hop"
                
            except (ImportError, Exception) as e:
                logging.warning(f"Mincut method failed: {str(e)}. Falling back to k_hop method.")
                method = "k_hop"
        
        if method == "k_hop":
            # Choose a random seed entity
            seed_entity = random.choice(entities_list)
            
            # Start with the seed entity
            subgraph_entities.add(seed_entity)
            
            # Add entities using k-hop expansion until we reach the target size
            frontier = [seed_entity]
            visited = {seed_entity}
            
            while len(subgraph_entities) < target_entities and frontier:
                current = frontier.pop(0)
                neighbors = list(G.neighbors(current))
                random.shuffle(neighbors)
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        subgraph_entities.add(neighbor)
                        frontier.append(neighbor)
                        
                        if len(subgraph_entities) >= target_entities:
                            break
        
        # Create a new KnowledgeGraph object for the subgraph
        subgraph = KnowledgeGraph(f"{self.name}_subgraph")
        
        # Filter triples to include only those where both head and tail are in the subgraph
        for split in ['train', 'valid', 'test']:
            subgraph.triples[split] = [
                (head, relation, tail) for head, relation, tail in self.triples[split]
                if head in subgraph_entities and tail in subgraph_entities
            ]
        
        # Update total triples
        subgraph.triples['total'] = (
            subgraph.triples['train'] +
            subgraph.triples['valid'] +
            subgraph.triples['test']
        )
        
        # Rebuild entity and relation mappings
        entities_set = set()
        relations_set = set()
        for split in ['train', 'valid', 'test']:
            for head, relation, tail in subgraph.triples[split]:
                entities_set.add(head)
                entities_set.add(tail)
                relations_set.add(relation)
        
        subgraph.entities = {i: entity for i, entity in enumerate(sorted(entities_set))}
        subgraph.relations = {i: relation for i, relation in enumerate(sorted(relations_set))}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the subgraph in the same format as the original graph
        self._save_subgraph(subgraph, output_dir)
        
        return subgraph

    def _save_subgraph(self, subgraph, output_dir):
        """
        Save the subgraph in the same format as the original graph.
        
        Args:
            subgraph (KnowledgeGraph): The subgraph to save.
            output_dir (str): Directory to save the subgraph files.
        """
        
        # Different datasets have different file formats
        if self.name in ["wn18", "wn18rr", "fb15k237"]:
            # Save entity2id.txt
            with open(os.path.join(output_dir, "entity2id.txt"), "w", encoding="utf-8") as f:
                f.write(str(len(subgraph.entities)) + "\n")
                for idx, entity in subgraph.entities.items():
                    f.write(f"{entity}\t{idx}\n")
            
            # Save relation2id.txt
            with open(os.path.join(output_dir, "relation2id.txt"), "w", encoding="utf-8") as f:
                f.write(str(len(subgraph.relations)) + "\n")
                for idx, relation in subgraph.relations.items():
                    f.write(f"{relation}\t{idx}\n")
            
            # Save train2id.txt, valid2id.txt, test2id.txt
            for split in ['train', 'valid', 'test']:
                with open(os.path.join(output_dir, f"{split}2id.txt"), "w", encoding="utf-8") as f:
                    f.write(str(len(subgraph.triples[split])) + "\n")
                    for head, relation, tail in subgraph.triples[split]:
                        # Convert text back to IDs
                        head_id = next((idx for idx, ent in subgraph.entities.items() if ent == head), None)
                        rel_id = next((idx for idx, rel in subgraph.relations.items() if rel == relation), None)
                        tail_id = next((idx for idx, ent in subgraph.entities.items() if ent == tail), None)
                        if head_id is not None and rel_id is not None and tail_id is not None:
                            f.write(f"{head_id} {tail_id} {rel_id}\n")
                            
        elif self.name in ["nation", "umls"]:
            # Save entities.txt
            with open(os.path.join(output_dir, "entities.txt"), "w", encoding="utf-8") as f:
                for idx, entity in subgraph.entities.items():
                    f.write(f"{idx} {entity}\n")
                    
            # Save relations.txt
            with open(os.path.join(output_dir, "relations.txt"), "w", encoding="utf-8") as f:
                for idx, relation in subgraph.relations.items():
                    f.write(f"{idx} {relation}\n")
                    
            # Save triples.txt (all splits combined)
            with open(os.path.join(output_dir, "triples.txt"), "w", encoding="utf-8") as f:
                for head, relation, tail in subgraph.triples['total']:
                    # Convert text back to IDs
                    head_id = next((idx for idx, ent in subgraph.entities.items() if ent == head), None)
                    rel_id = next((idx for idx, rel in subgraph.relations.items() if rel == relation), None)
                    tail_id = next((idx for idx, ent in subgraph.entities.items() if ent == tail), None)
                    if head_id is not None and rel_id is not None and tail_id is not None:
                        f.write(f"{rel_id} {head_id} {tail_id}\n")
                        
        elif self.name == "yago3-10":
            # YAGO3-10 is in Hugging Face format
            # We'll save it as JSON files that can be easily loaded
            for split in ['train', 'valid', 'test']:
                split_data = []
                for head, relation, tail in subgraph.triples[split]:
                    split_data.append({"head": head, "relation": relation, "tail": tail})
                    
                with open(os.path.join(output_dir, f"{split}.json"), "w", encoding="utf-8") as f:
                    json.dump(split_data, f, ensure_ascii=False, indent=2)
                    
        elif self.name in ["codex-small", "codex-medium", "codex-large"]:            
            # Create necessary directories
            os.makedirs(os.path.join(output_dir, "entities", "en"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "relations", "en"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "triples", f"{self.name}_subgraph"), exist_ok=True)
            
            # Save entities.json
            entities_data = {}
            for idx, entity in subgraph.entities.items():
                entities_data[entity] = {"label": entity}
                
            with open(os.path.join(output_dir, "entities", "en", "entities.json"), "w", encoding="utf-8") as f:
                json.dump(entities_data, f, ensure_ascii=False, indent=2)
                
            # Save relations.json
            relations_data = {}
            for idx, relation in subgraph.relations.items():
                relations_data[relation] = {"label": relation}
                
            with open(os.path.join(output_dir, "relations", "en", "relations.json"), "w", encoding="utf-8") as f:
                json.dump(relations_data, f, ensure_ascii=False, indent=2)
                
            # Save train.txt, valid.txt, test.txt
            for split in ['train', 'valid', 'test']:
                with open(os.path.join(output_dir, "triples", f"{self.name}_subgraph", f"{split}.txt"), "w", encoding="utf-8") as f:
                    for head, relation, tail in subgraph.triples[split]:
                        f.write(f"{head}\t{relation}\t{tail}\n")
                        
        elif self.name == "wd-singer":
            # Save WD-singer format
            for split in ['train', 'valid', 'test']:
                split_name = "dev" if split == "valid" else split  # The dataset uses "dev" instead of "valid"
                with open(os.path.join(output_dir, f"{split_name}.triples"), "w", encoding="utf-8") as f:
                    for head, relation, tail in subgraph.triples[split]:
                        f.write(f"{head} {tail} {relation}\n")
        else:
            raise ValueError(f"Unsupported dataset for saving: {self.name}")
            
        logging.info(f"Subgraph saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display knowledge graph statistics.")
    parser.add_argument("--kg", type=str, required=True, 
                        help="Knowledge graph name, e.g. wn18rr, wn18, fb15k237, nation, umls, yago3-10")
    args = parser.parse_args()

    # Create a KnowledgeGraph instance and load data (after loading, data is in unified text format)
    kg = KnowledgeGraph(args.kg)
    kg.load(degree_threshold=20, calculate_close_relation=True, mapping_method='bert', m=10, batch_size=16, bert_model="distilbert-base-uncased")
    # kg.print_statistics()

    # #########################  Usage 1: Relation Selection #########################
    # relation_matrix, relation_mapping = kg.relation_graph(
    #     methods=["co-occurrence", "transitivity", "semantic"],
    #     weights={"co-occurrence": 0.4, "transitivity": 0.2, "semantic": 0.2},
    #     semantic_method="bert"  
    # )
    
    # results = kg.optimize_spectral_clustering_params(
    #     relation_matrix=relation_matrix,
    #     relation_mapping=relation_mapping,
    #     min_clusters=2,           
    #     max_clusters=15,          
    #     step=1,                  
    #     num_relations=15,   
    #     evaluation_metrics = [
    #         "silhouette_score",      # Silhouette coefficient - measures cluster cohesion and separation
    #         "calinski_harabasz",     # Calinski-Harabasz index - higher is better
    #         "davies_bouldin",        # Davies-Bouldin index - lower is better
    #         "eigenvalue_gap",        # Eigenvalue gap - identifies natural cluster separations
    #         "relation_coverage",     # Relation coverage - measures proportion of triples covered by selected relations
    #         "graph_connectivity"     # Graph connectivity - measures connectivity of subgraph formed by selected relations
    #         ],      
    #     trials=3,                 
    #     assign_strategy="max_internal", 
    #     visualize=True,           
    #     output_dir=os.path.join(kg.data_path, kg.name), 
    #     interactive=True       
    # )
    
    # best_n_clusters = results["best_n_clusters"]
    # selected_relations = results["selected_relations"]

    # logging.info(f"Best number of clusters: {best_n_clusters}")
    # logging.info(f"Selected relations :")
    # for i, rel in enumerate(selected_relations):
    #     logging.info(f"{i+1}. {rel}")

    # cluster_info = kg.relation_selection(
    #     relation_matrix=relation_matrix,
    #     relation_mapping=relation_mapping,
    #     method="spectral_clustering",
    #     num_relations=None,  # This is the key to get cluster mappings
    #     n_clusters=best_n_clusters,        # Specify desired number of clusters
    # )
    
    # # Print cluster information
    # logging.info("\nCluster Mapping Results:")
    # for cluster_id, relations in cluster_info["cluster_mapping"].items():
    #     stats = cluster_info["cluster_stats"][cluster_id]
    #     logging.info(f"Cluster {cluster_id} (Size: {stats['size']}, Avg similarity: {stats['avg_similarity']:.3f}):")
    #     for relation in relations:
    #         logging.info(f"  - {relation}")
    
    # # Visualize the clusters
    # kg.visualize_relation_clusters(
    #     relation_matrix=relation_matrix,
    #     relation_mapping=relation_mapping,
    #     weight_threshold=0.5,
    #     cluster_info=cluster_info,
    #     output_path=os.path.join(kg.data_path, kg.name, f"{kg.name}_relation_clusters.png")
    # )
    
    # # Perform detailed analysis of the clusters
    # analysis = kg.analyze_relation_clusters(
    #     relation_matrix=relation_matrix,
    #     relation_mapping=relation_mapping,
    #     cluster_info=cluster_info
    # )


    # #########################  Usage 2: Generate Relation Samples #########################
    # selected_relations = [
    #     "award received",
    #     "composer",
    #     "director",
    #     "lyricist",
    #     "notable work",
    #     "performer",
    #     "screenwriter",
    #     "residence",
    #     "country of citizenship",
    #     "collection",
    #     "producer",
    #     "significant event",
    #     "owner of",
    #     "member of",
    #     "nominated for"
    #     ]
    # openai_api_key = ""
    # _ = kg.generate_relation_samples(relations=selected_relations, k=10, output_path=os.path.join(kg.data_path, kg.name, f"{kg.name}_selected_relations.json"), openai_api_key=openai_api_key)

    #########################  Usage 3: Extract Subgraph #########################
    # subgraph = kg.extract_subgraph(os.path.join(kg.data_path, kg.name, f"{kg.name}_subgraph"), size_param=0.5)
    import pdb; pdb.set_trace()
