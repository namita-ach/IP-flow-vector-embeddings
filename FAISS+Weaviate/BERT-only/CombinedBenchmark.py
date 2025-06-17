import pandas as pd
import faiss
import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import weaviate
import warnings
from typing import List, Dict, Any, Tuple, Optional
import json
from datetime import datetime
import random
import re

warnings.filterwarnings('ignore')

class VectorDBBenchmark:
    def __init__(self, model_name='all-mpnet-base-v2', weaviate_url="http://localhost:8080"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.weaviate_url = weaviate_url
        
        # Initialize Weaviate client
        try:
            self.weaviate_client = weaviate.Client(weaviate_url)
            self.weaviate_available = True
            print(f"Weaviate connection established at {weaviate_url}")
        except Exception as e:
            print(f"Weaviate connection failed: {e}")
            self.weaviate_available = False
        
        # Performance tracking
        self.results = {
            'faiss': {
                'insertion_times': [], 'insertion_sizes': [], 'insertion_memory': [],
                'query_times': [], 'query_sizes': [], 'query_memory': [],
                'deletion_times': [], 'deletion_sizes': [], 'deletion_memory': [],
                'update_times': [], 'update_sizes': [], 'update_memory': [],
                'simple_query_times': [], 'simple_query_sizes': [], 'simple_query_memory': [],
                'complex_query_times': [], 'complex_query_sizes': [], 'complex_query_memory': [],
                'nl_query_times': [], 'nl_query_sizes': [], 'nl_query_memory': []
            },
            'weaviate': {
                'insertion_times': [], 'insertion_sizes': [], 'insertion_memory': [],
                'query_times': [], 'query_sizes': [], 'query_memory': [],
                'deletion_times': [], 'deletion_sizes': [], 'deletion_memory': [],
                'update_times': [], 'update_sizes': [], 'update_memory': [],
                'simple_query_times': [], 'simple_query_sizes': [], 'simple_query_memory': [],
                'complex_query_times': [], 'complex_query_sizes': [], 'complex_query_memory': [],
                'nl_query_times': [], 'nl_query_sizes': [], 'nl_query_memory': []
            }
        }
        
        self.faiss_index = None
        self.embeddings_cache = {}
        self.faiss_id_map = {}  # Map FAISS indices to original IDs
        self.faiss_data_store = {}  # Store original data for FAISS
        self.next_faiss_id = 0

    def setup_weaviate_schema(self):
        if not self.weaviate_available:
            return False
            
        try:
            # Delete existing class if it exists
            try:
                self.weaviate_client.schema.delete_class("IPFlow")
                print("Deleted existing IPFlow class")
            except:
                pass
            
            # Create new schema
            ip_flow_schema = {
                "class": "IPFlow",
                "vectorizer": "none",
                "properties": [
                    {"name": "frame_number", "dataType": ["int"]},
                    {"name": "frame_time", "dataType": ["string"]},
                    {"name": "source_ip", "dataType": ["string"]},
                    {"name": "destination_ip", "dataType": ["string"]},
                    {"name": "source_port", "dataType": ["int"]},
                    {"name": "destination_port", "dataType": ["int"]},
                    {"name": "protocol", "dataType": ["string"]},
                    {"name": "frame_length", "dataType": ["int"]},
                    {"name": "packet_text", "dataType": ["string"]},
                    {"name": "original_id", "dataType": ["string"]}  # For deletion tracking
                ]
            }
            
            self.weaviate_client.schema.create_class(ip_flow_schema)
            print("Weaviate schema created successfully")
            return True
        except Exception as e:
            print(f"Failed to setup Weaviate schema: {e}")
            return False

    def setup_faiss_index(self, dimension: int, num_vectors: int):
        try:
            nlist = min(100, max(1, int(np.sqrt(num_vectors))))
            quantizer = faiss.IndexFlatIP(dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            print(f"FAISS index created with dimension {dimension}")
            return True
        except Exception as e:
            print(f"Failed to setup FAISS index: {e}")
            return False

    def measure_performance(self, func, *args, **kwargs):
        process = psutil.Process()
        
        # Initial measurements
        start_time = time.time()
        start_memory = process.memory_info().rss / (1024 ** 2)  # MB
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            print(f"Function execution failed: {e}")
            result = None
            success = False
        
        # Final measurements
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 ** 2)  # MB
        
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        return result, execution_time, memory_delta, success

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        cache_key = str(hash(tuple(texts)))
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.embeddings_cache[cache_key] = embeddings
        return embeddings

    def benchmark_faiss_insertion(self, packet_texts: List[str], packet_data: List[Dict], batch_size: int):
        embeddings = self.generate_embeddings(packet_texts)
        
        if self.faiss_index is None:
            self.setup_faiss_index(embeddings.shape[1], len(embeddings))
            # Train the index
            self.faiss_index.train(embeddings)
        
        def insert_batch():
            start_id = self.next_faiss_id
            self.faiss_index.add(embeddings)
            
            # Store mapping and data
            for i, data in enumerate(packet_data):
                faiss_id = start_id + i
                original_id = data.get('original_id', str(faiss_id))
                self.faiss_id_map[faiss_id] = original_id
                self.faiss_data_store[faiss_id] = data
            
            self.next_faiss_id += len(embeddings)
            return len(embeddings)
        
        result, exec_time, memory_delta, success = self.measure_performance(insert_batch)
        
        if success:
            self.results['faiss']['insertion_times'].append(exec_time)
            self.results['faiss']['insertion_sizes'].append(batch_size)
            self.results['faiss']['insertion_memory'].append(memory_delta)
        
        return success

    def benchmark_weaviate_insertion(self, packet_data: List[Dict], batch_size: int):
        if not self.weaviate_available:
            return False
        
        def insert_batch():
            with self.weaviate_client.batch as batch:
                batch.batch_size = min(100, batch_size)
                for i, data in enumerate(packet_data):
                    # Generate embedding
                    embedding = self.generate_embeddings([data['packet_text']])[0]
                    
                    # Add original id for deletion tracking
                    data_with_id = data.copy()
                    data_with_id['original_id'] = data.get('original_id', str(i))
                    
                    batch.add_data_object(
                        data_object=data_with_id,
                        class_name="IPFlow",
                        vector=embedding.tolist()
                    )
            return len(packet_data)
        
        result, exec_time, memory_delta, success = self.measure_performance(insert_batch)
        
        if success:
            self.results['weaviate']['insertion_times'].append(exec_time)
            self.results['weaviate']['insertion_sizes'].append(batch_size)
            self.results['weaviate']['insertion_memory'].append(memory_delta)
        
        return success

    def benchmark_faiss_deletion(self, num_to_delete: int):
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return False
        
        def delete_items():
            # Get random ids to delete
            available_ids = list(self.faiss_data_store.keys())
            if len(available_ids) < num_to_delete:
                return 0
            
            ids_to_delete = random.sample(available_ids, num_to_delete)
            
            # Remove from data store
            for faiss_id in ids_to_delete:
                del self.faiss_data_store[faiss_id]
                if faiss_id in self.faiss_id_map:
                    del self.faiss_id_map[faiss_id]
            
            if self.faiss_data_store:
                remaining_texts = [data['packet_text'] for data in self.faiss_data_store.values()]
                embeddings = self.generate_embeddings(remaining_texts)
                
                dimension = embeddings.shape[1]
                nlist = min(100, max(1, int(np.sqrt(len(embeddings)))))
                quantizer = faiss.IndexFlatIP(dimension)
                new_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                new_index.train(embeddings)
                new_index.add(embeddings)
                
                self.faiss_index = new_index
                
                self.faiss_id_map = {}
                for i, (faiss_id, data) in enumerate(self.faiss_data_store.items()):
                    self.faiss_id_map[i] = data.get('original_id', str(faiss_id))
            else:
                self.faiss_index = None
            
            return len(ids_to_delete)
        
        result, exec_time, memory_delta, success = self.measure_performance(delete_items)
        
        if success and result > 0:
            self.results['faiss']['deletion_times'].append(exec_time)
            self.results['faiss']['deletion_sizes'].append(result)
            self.results['faiss']['deletion_memory'].append(memory_delta)
        
        return success

    def benchmark_weaviate_deletion(self, num_to_delete: int):
        if not self.weaviate_available:
            return False
        
        def delete_items():
            # Get some objects to delete
            result = self.weaviate_client.query.get("IPFlow", ["original_id"]).with_limit(num_to_delete).do()
            
            if not result.get('data', {}).get('Get', {}).get('IPFlow'):
                return 0
            
            objects_to_delete = result['data']['Get']['IPFlow']
            deleted_count = 0
            
            for obj in objects_to_delete:
                try:
                    # Get the object ID
                    obj_result = self.weaviate_client.query.get("IPFlow", ["original_id"]).with_where({
                        "path": ["original_id"],
                        "operator": "Equal",
                        "valueString": obj["original_id"]
                    }).with_additional(["id"]).with_limit(1).do()
                    
                    if obj_result.get('data', {}).get('Get', {}).get('IPFlow'):
                        obj_id = obj_result['data']['Get']['IPFlow'][0]['_additional']['id']
                        self.weaviate_client.data_object.delete(obj_id)
                        deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete object: {e}")
                    continue
            
            return deleted_count
        
        result, exec_time, memory_delta, success = self.measure_performance(delete_items)
        
        if success and result > 0:
            self.results['weaviate']['deletion_times'].append(exec_time)
            self.results['weaviate']['deletion_sizes'].append(result)
            self.results['weaviate']['deletion_memory'].append(memory_delta)
        
        return success

    def benchmark_faiss_simple_query(self, protocols: List[str], k: int = 5):
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return False
        
        def search_batch():
            results = []
            for protocol in protocols:
                # Create query embedding
                query_text = f"protocol {protocol}"
                query_embedding = self.generate_embeddings([query_text])
                
                # Search
                self.faiss_index.nprobe = 10
                distances, indices = self.faiss_index.search(query_embedding, k)
                
                # Filter results by protocol (post-processing)
                filtered_results = []
                for idx in indices[0]:
                    if idx != -1 and idx in self.faiss_data_store:
                        data = self.faiss_data_store[idx]
                        if data['protocol'].lower() == protocol.lower():
                            filtered_results.append((idx, data))
                
                results.append(filtered_results)
            return results
        
        result, exec_time, memory_delta, success = self.measure_performance(search_batch)
        
        if success:
            self.results['faiss']['simple_query_times'].append(exec_time)
            self.results['faiss']['simple_query_sizes'].append(len(protocols))
            self.results['faiss']['simple_query_memory'].append(memory_delta)
        
        return success

    def benchmark_weaviate_simple_query(self, protocols: List[str], k: int = 5):
        if not self.weaviate_available:
            return False
        
        def search_batch():
            results = []
            for protocol in protocols:
                result = self.weaviate_client.query.get("IPFlow", [
                    "frame_number", "source_ip", "destination_ip", "protocol", "frame_length"
                ]).with_where({
                    "path": ["protocol"],
                    "operator": "Equal",
                    "valueString": protocol
                }).with_limit(k).do()
                
                results.append(result)
            return results
        
        result, exec_time, memory_delta, success = self.measure_performance(search_batch)
        
        if success:
            self.results['weaviate']['simple_query_times'].append(exec_time)
            self.results['weaviate']['simple_query_sizes'].append(len(protocols))
            self.results['weaviate']['simple_query_memory'].append(memory_delta)
        
        return success

    def benchmark_faiss_complex_query(self, query_params: List[Dict], k: int = 5):
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return False
        
        def search_batch():
            results = []
            for params in query_params:
                # Create query embedding
                query_text = f"source {params['source_ip']} destination {params['destination_ip']}"
                query_embedding = self.generate_embeddings([query_text])
                
                # Search
                self.faiss_index.nprobe = 10
                distances, indices = self.faiss_index.search(query_embedding, k)
                
                # Filter results (post-processing)
                filtered_results = []
                for idx in indices[0]:
                    if idx != -1 and idx in self.faiss_data_store:
                        data = self.faiss_data_store[idx]
                        if (data['source_ip'] == params['source_ip'] and 
                            data['destination_ip'] == params['destination_ip']):
                            filtered_results.append((idx, data))
                
                results.append(filtered_results)
            return results
        
        result, exec_time, memory_delta, success = self.measure_performance(search_batch)
        
        if success:
            self.results['faiss']['complex_query_times'].append(exec_time)
            self.results['faiss']['complex_query_sizes'].append(len(query_params))
            self.results['faiss']['complex_query_memory'].append(memory_delta)
        
        return success

    def benchmark_weaviate_complex_query(self, query_params: List[Dict], k: int = 5):
        if not self.weaviate_available:
            return False
        
        def search_batch():
            results = []
            for params in query_params:
                result = self.weaviate_client.query.get("IPFlow", [
                    "frame_number", "source_ip", "destination_ip", "protocol", "frame_length"
                ]).with_where({
                    "operator": "And",
                    "operands": [
                        {
                            "path": ["source_ip"],
                            "operator": "Equal",
                            "valueString": params['source_ip']
                        },
                        {
                            "path": ["destination_ip"],
                            "operator": "Equal",
                            "valueString": params['destination_ip']
                        }
                    ]
                }).with_limit(k).do()
                
                results.append(result)
            return results
        
        result, exec_time, memory_delta, success = self.measure_performance(search_batch)
        
        if success:
            self.results['weaviate']['complex_query_times'].append(exec_time)
            self.results['weaviate']['complex_query_sizes'].append(len(query_params))
            self.results['weaviate']['complex_query_memory'].append(memory_delta)
        
        return success

    def parse_natural_language_query(self, nl_query: str) -> Dict:
        params = {}
        
        # Extract source IP
        source_match = re.search(r'(?:from|source)\s+(\d+\.\d+\.\d+\.\d+)', nl_query, re.IGNORECASE)
        if source_match:
            params['source_ip'] = source_match.group(1)
        
        # Extract destination IP
        dest_match = re.search(r'(?:to|destination)\s+(\d+\.\d+\.\d+\.\d+)', nl_query, re.IGNORECASE)
        if dest_match:
            params['destination_ip'] = dest_match.group(1)
        
        # Extract protocol
        protocol_match = re.search(r'(?:protocol|using)\s+(TCP|UDP|HTTP|HTTPS|DNS)', nl_query, re.IGNORECASE)
        if protocol_match:
            params['protocol'] = protocol_match.group(1).upper()
        
        # Extract packet length conditions
        length_match = re.search(r'(?:length|size)\s+(less than|greater than|equal to|=|<|>)\s+(\d+)', nl_query, re.IGNORECASE)
        if length_match:
            operator = length_match.group(1).lower()
            value = int(length_match.group(2))
            params['length_condition'] = {'operator': operator, 'value': value}
        
        return params

    def benchmark_faiss_nl_query(self, nl_queries: List[str], k: int = 5):
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return False
        
        def search_batch():
            results = []
            for nl_query in nl_queries:
                params = self.parse_natural_language_query(nl_query)
                
                query_embedding = self.generate_embeddings([nl_query])
                
                # Search
                self.faiss_index.nprobe = 10
                distances, indices = self.faiss_index.search(query_embedding, k)
                
                # Filter results based on parsed parameters
                filtered_results = []
                for idx in indices[0]:
                    if idx != -1 and idx in self.faiss_data_store:
                        data = self.faiss_data_store[idx]
                        match = True
                        
                        # Check source IP
                        if 'source_ip' in params and data['source_ip'] != params['source_ip']:
                            match = False
                        
                        # Check destination IP
                        if 'destination_ip' in params and data['destination_ip'] != params['destination_ip']:
                            match = False
                        
                        # Check protocol
                        if 'protocol' in params and data['protocol'].upper() != params['protocol']:
                            match = False
                        
                        # Check length condition
                        if 'length_condition' in params:
                            length_cond = params['length_condition']
                            data_length = data['frame_length']
                            if length_cond['operator'] in ['less than', '<']:
                                if data_length >= length_cond['value']:
                                    match = False
                            elif length_cond['operator'] in ['greater than', '>']:
                                if data_length <= length_cond['value']:
                                    match = False
                            elif length_cond['operator'] in ['equal to', '=']:
                                if data_length != length_cond['value']:
                                    match = False
                        
                        if match:
                            filtered_results.append((idx, data))
                
                results.append(filtered_results)
            return results
        
        result, exec_time, memory_delta, success = self.measure_performance(search_batch)
        
        if success:
            self.results['faiss']['nl_query_times'].append(exec_time)
            self.results['faiss']['nl_query_sizes'].append(len(nl_queries))
            self.results['faiss']['nl_query_memory'].append(memory_delta)
        
        return success

    def benchmark_weaviate_nl_query(self, nl_queries: List[str], k: int = 5):
        if not self.weaviate_available:
            return False
        
        def search_batch():
            results = []
            for nl_query in nl_queries:
                # Parse the natural language query
                params = self.parse_natural_language_query(nl_query)
                
                # Build Weaviate where clause
                where_conditions = []
                
                if 'source_ip' in params:
                    where_conditions.append({
                        "path": ["source_ip"],
                        "operator": "Equal",
                        "valueString": params['source_ip']
                    })
                
                if 'destination_ip' in params:
                    where_conditions.append({
                        "path": ["destination_ip"],
                        "operator": "Equal",
                        "valueString": params['destination_ip']
                    })
                
                if 'protocol' in params:
                    where_conditions.append({
                        "path": ["protocol"],
                        "operator": "Equal",
                        "valueString": params['protocol']
                    })
                
                if 'length_condition' in params:
                    length_cond = params['length_condition']
                    if length_cond['operator'] in ['less than', '<']:
                        where_conditions.append({
                            "path": ["frame_length"],
                            "operator": "LessThan",
                            "valueInt": length_cond['value']
                        })
                    elif length_cond['operator'] in ['greater than', '>']:
                        where_conditions.append({
                            "path": ["frame_length"],
                            "operator": "GreaterThan",
                            "valueInt": length_cond['value']
                        })
                    elif length_cond['operator'] in ['equal to', '=']:
                        where_conditions.append({
                            "path": ["frame_length"],
                            "operator": "Equal",
                            "valueInt": length_cond['value']
                        })
                
                # Execute query
                query_builder = self.weaviate_client.query.get("IPFlow", [
                    "frame_number", "source_ip", "destination_ip", "protocol", "frame_length"
                ])
                
                if where_conditions:
                    if len(where_conditions) == 1:
                        where_clause = where_conditions[0]
                    else:
                        where_clause = {
                            "operator": "And",
                            "operands": where_conditions
                        }
                    query_builder = query_builder.with_where(where_clause)
                
                result = query_builder.with_limit(k).do()
                results.append(result)
            
            return results
        
        result, exec_time, memory_delta, success = self.measure_performance(search_batch)
        
        if success:
            self.results['weaviate']['nl_query_times'].append(exec_time)
            self.results['weaviate']['nl_query_sizes'].append(len(nl_queries))
            self.results['weaviate']['nl_query_memory'].append(memory_delta)
        
        return success

    def benchmark_faiss_query(self, query_texts: List[str], k: int = 5):
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return False
        
        query_embeddings = self.generate_embeddings(query_texts)
        
        def search_batch():
            self.faiss_index.nprobe = 10
            distances, indices = self.faiss_index.search(query_embeddings, k)
            return distances, indices
        
        result, exec_time, memory_delta, success = self.measure_performance(search_batch)
        
        if success:
            self.results['faiss']['query_times'].append(exec_time)
            self.results['faiss']['query_sizes'].append(len(query_texts))
            self.results['faiss']['query_memory'].append(memory_delta)
        
        return success

    def benchmark_weaviate_query(self, query_texts: List[str], k: int = 5):
        if not self.weaviate_available:
            return False
        
        def search_batch():
            results = []
            for query_text in query_texts:
                query_embedding = self.generate_embeddings([query_text])[0]
                
                result = self.weaviate_client.query.get("IPFlow", [
                    "frame_number", "source_ip", "destination_ip", "protocol"
                ]).with_near_vector({
                    "vector": query_embedding.tolist()
                }).with_limit(k).do()
                
                results.append(result)
            return results
        
        result, exec_time, memory_delta, success = self.measure_performance(search_batch)
        
        if success:
            self.results['weaviate']['query_times'].append(exec_time)
            self.results['weaviate']['query_sizes'].append(len(query_texts))
            self.results['weaviate']['query_memory'].append(memory_delta)
        
        return success

    def create_sample_data(self, num_samples: int) -> Tuple[List[str], List[Dict]]:
        packet_texts = []
        packet_data = []
        
        protocols = ["TCP", "UDP", "HTTP", "HTTPS", "DNS"]
        
        for i in range(num_samples):
            source_ip = f"192.168.{i%256}.{(i*7)%256}"
            dest_ip = f"10.0.{(i*3)%256}.{(i*11)%256}"
            protocol = protocols[i % len(protocols)]
            src_port = (i * 13) % 65535
            dst_port = (i * 17) % 65535
            frame_len = (i * 100) % 1500 + 64  # Ensure minimum frame size
            
            packet_text = f"{source_ip} {dest_ip} {protocol} {src_port} {dst_port} {frame_len}"
            packet_texts.append(packet_text)
            
            packet_data.append({
                "frame_number": i,
                "frame_time": f"2024-01-01 10:{i%60:02d}:{(i*2)%60:02d}",
                "source_ip": source_ip,
                "destination_ip": dest_ip,
                "source_port": src_port,
                "destination_port": dst_port,
                "protocol": protocol,
                "frame_length": frame_len,
                "packet_text": packet_text,
                "original_id": f"packet_{i}"
            })
        
        return packet_texts, packet_data

    def create_query_sets(self, packet_data: List[Dict]) -> Tuple[List[str], List[Dict], List[str]]:
        if not packet_data:
            return [], [], []
        
        protocols = list(set([data['protocol'] for data in packet_data[:10]]))
        
        complex_queries = []
        for data in packet_data[:5]:
            complex_queries.append({
                'source_ip': data['source_ip'],
                'destination_ip': data['destination_ip']
            })
        
        # Natural language queries
        nl_queries = []
        for i, data in enumerate(packet_data[:5]):
            if i == 0:
                nl_queries.append(f"find datapoints from {data['source_ip']} to {data['destination_ip']} with {data['protocol']} protocol")
            elif i == 1:
                nl_queries.append(f"show traffic from source {data['source_ip']} to destination {data['destination_ip']} using protocol {data['protocol']} and packet length greater than {data['frame_length'] - 100}")
            elif i == 2:
                nl_queries.append(f"find packets from {data['source_ip']} to {data['destination_ip']} with length less than {data['frame_length'] + 100}")
            elif i == 3:
                nl_queries.append(f"get flows from source {data['source_ip']} using {data['protocol']} protocol with packet size equal to {data['frame_length']}")
            else:
                nl_queries.append(f"find all traffic from {data['source_ip']} to {data['destination_ip']} with {data['protocol']} protocol and length greater than 500")
        
        return protocols, complex_queries, nl_queries

    def run_benchmark(self, batch_sizes: List[int] = [100, 500, 1000, 2000, 5000]):
        print(f"\n{'='*60}")
        print(f"BENCHMARKING MODEL: {self.model_name}")
        print(f"{'='*60}")
        
        # Setup
        if self.weaviate_available:
            self.setup_weaviate_schema()
        
        # Store data for queries
        all_packet_data = []
        
        # Run benchmarks for different batch sizes
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Create sample data
            packet_texts, packet_data = self.create_sample_data(batch_size)
            all_packet_data.extend(packet_data)
            
            # FAISS insertion
            print("  FAISS insertion...", end=" ")
            faiss_insert_success = self.benchmark_faiss_insertion(packet_texts, packet_data, batch_size)
            print("done" if faiss_insert_success else "ERROR")
            
            # Weaviate insertion
            if self.weaviate_available:
                print("  Weaviate insertion...", end=" ")
                weaviate_insert_success = self.benchmark_weaviate_insertion(packet_data, batch_size)
                print("done" if weaviate_insert_success else "ERROR")
            
            # Basic query benchmarks
            query_texts = [
                "TCP flow 192.168.1.1 80",
                "UDP DNS traffic",
                "HTTP web traffic",
                "HTTPS secure connection",
                "192.168.0.1 10.0.0.1 TCP"
            ]
            
            if faiss_insert_success:
                print("  FAISS basic query...", end=" ")
                faiss_query_success = self.benchmark_faiss_query(query_texts)
                print("done" if faiss_query_success else "ERROR")
            
            if self.weaviate_available and weaviate_insert_success:
                print("  Weaviate basic query...", end=" ")
                weaviate_query_success = self.benchmark_weaviate_query(query_texts)
                print("done" if weaviate_query_success else "ERROR")
            
            # make query sets from current data
            protocols, complex_queries, nl_queries = self.create_query_sets(packet_data)
            
            if protocols:
                if faiss_insert_success:
                    print("  FAISS simple queries...", end=" ")
                    faiss_simple_success = self.benchmark_faiss_simple_query(protocols)
                    print("done" if faiss_simple_success else "ERROR")
                
                if self.weaviate_available and weaviate_insert_success:
                    print("  Weaviate simple queries...", end=" ")
                    weaviate_simple_success = self.benchmark_weaviate_simple_query(protocols)
                    print("done" if weaviate_simple_success else "ERROR")
            
            # Complex queries (source + destination)
            if complex_queries:
                if faiss_insert_success:
                    print("  FAISS complex queries...", end=" ")
                    faiss_complex_success = self.benchmark_faiss_complex_query(complex_queries)
                    print("done" if faiss_complex_success else "ERROR")
                
                if self.weaviate_available and weaviate_insert_success:
                    print("  Weaviate complex queries...", end=" ")
                    weaviate_complex_success = self.benchmark_weaviate_complex_query(complex_queries)
                    print("done" if weaviate_complex_success else "ERROR")
            
            # Natural language queries
            if nl_queries:
                if faiss_insert_success:
                    print("  FAISS NL queries...", end=" ")
                    faiss_nl_success = self.benchmark_faiss_nl_query(nl_queries)
                    print("done" if faiss_nl_success else "ERROR")
                
                if self.weaviate_available and weaviate_insert_success:
                    print("  Weaviate NL queries...", end=" ")
                    weaviate_nl_success = self.benchmark_weaviate_nl_query(nl_queries)
                    print("done" if weaviate_nl_success else "ERROR")
        
        # Deletion benchmarks (after all insertions)
        deletion_sizes = [50, 100, 200]
        
        for del_size in deletion_sizes:
            print(f"\nTesting deletion size: {del_size}")
            
            if self.faiss_index and self.faiss_index.ntotal > del_size:
                print("  FAISS deletion...", end=" ")
                faiss_del_success = self.benchmark_faiss_deletion(del_size)
                print("done" if faiss_del_success else "ERROR")
            
            if self.weaviate_available:
                print("  Weaviate deletion...", end=" ")
                weaviate_del_success = self.benchmark_weaviate_deletion(del_size)
                print("done" if weaviate_del_success else "ERROR")

    def save_results_to_csv(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for CSV
        csv_data = []
        
        operation_types = [
            'insertion', 'query', 'deletion', 'update',
            'simple_query', 'complex_query', 'nl_query'
        ]
        
        for db_type in ['faiss', 'weaviate']:
            for operation in operation_types:
                times = self.results[db_type].get(f'{operation}_times', [])
                sizes = self.results[db_type].get(f'{operation}_sizes', [])
                memory = self.results[db_type].get(f'{operation}_memory', [])
                
                for i in range(len(times)):
                    csv_data.append({
                        'model': self.model_name,
                        'database': db_type,
                        'operation': operation,
                        'batch_size': sizes[i] if i < len(sizes) else 0,
                        'execution_time': times[i],
                        'memory_delta_mb': memory[i] if i < len(memory) else 0,
                        'throughput': sizes[i] / times[i] if i < len(sizes) and times[i] > 0 else 0
                    })
        
        df = pd.DataFrame(csv_data)
        clean_model_name = self.model_name.replace('/', '_').replace('-', '_')
        csv_path = os.path.join(output_dir, f"{clean_model_name}_benchmark.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    def plot_results(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle(f'FAISS vs Weaviate Performance - {self.model_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Insertion Time vs Batch Size
        ax = axes[0, 0]
        if self.results['faiss']['insertion_times']:
            ax.plot(self.results['faiss']['insertion_sizes'], self.results['faiss']['insertion_times'], 
                   'o-', label='FAISS', linewidth=2, markersize=6)
        if self.results['weaviate']['insertion_times']:
            ax.plot(self.results['weaviate']['insertion_sizes'], self.results['weaviate']['insertion_times'], 
                   's-', label='Weaviate', linewidth=2, markersize=6)
        ax.set_title('Insertion Time vs Batch Size', fontweight='bold')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Time (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Query Time vs Query Count
        ax = axes[0, 1]
        if self.results['faiss']['query_times']:
            ax.plot(self.results['faiss']['query_sizes'], self.results['faiss']['query_times'], 
                   'o-', label='FAISS', linewidth=2, markersize=6)
        if self.results['weaviate']['query_times']:
            ax.plot(self.results['weaviate']['query_sizes'], self.results['weaviate']['query_times'], 
                   's-', label='Weaviate', linewidth=2, markersize=6)
        ax.set_title('Basic Query Time', fontweight='bold')
        ax.set_xlabel('Number of Queries')
        ax.set_ylabel('Time (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Deletion Time
        ax = axes[0, 2]
        if self.results['faiss']['deletion_times']:
            ax.plot(self.results['faiss']['deletion_sizes'], self.results['faiss']['deletion_times'], 
                   'o-', label='FAISS', linewidth=2, markersize=6)
        if self.results['weaviate']['deletion_times']:
            ax.plot(self.results['weaviate']['deletion_sizes'], self.results['weaviate']['deletion_times'], 
                   's-', label='Weaviate', linewidth=2, markersize=6)
        ax.set_title('Deletion Time', fontweight='bold')
        ax.set_xlabel('Items Deleted')
        ax.set_ylabel('Time (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Simple Query Time
        ax = axes[1, 0]
        if self.results['faiss']['simple_query_times']:
            ax.plot(self.results['faiss']['simple_query_sizes'], self.results['faiss']['simple_query_times'], 
                   'o-', label='FAISS', linewidth=2, markersize=6)
        if self.results['weaviate']['simple_query_times']:
            ax.plot(self.results['weaviate']['simple_query_sizes'], self.results['weaviate']['simple_query_times'], 
                   's-', label='Weaviate', linewidth=2, markersize=6)
        ax.set_title('Simple Query Time (Protocol)', fontweight='bold')
        ax.set_xlabel('Number of Queries')
        ax.set_ylabel('Time (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Complex Query Time
        ax = axes[1, 1]
        if self.results['faiss']['complex_query_times']:
            ax.plot(self.results['faiss']['complex_query_sizes'], self.results['faiss']['complex_query_times'], 
                   'o-', label='FAISS', linewidth=2, markersize=6)
        if self.results['weaviate']['complex_query_times']:
            ax.plot(self.results['weaviate']['complex_query_sizes'], self.results['weaviate']['complex_query_times'], 
                   's-', label='Weaviate', linewidth=2, markersize=6)
        ax.set_title('Complex Query Time (Source+Dest)', fontweight='bold')
        ax.set_xlabel('Number of Queries')
        ax.set_ylabel('Time (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Natural Language Query Time
        ax = axes[1, 2]
        if self.results['faiss']['nl_query_times']:
            ax.plot(self.results['faiss']['nl_query_sizes'], self.results['faiss']['nl_query_times'], 
                   'o-', label='FAISS', linewidth=2, markersize=6)
        if self.results['weaviate']['nl_query_times']:
            ax.plot(self.results['weaviate']['nl_query_sizes'], self.results['weaviate']['nl_query_times'], 
                   's-', label='Weaviate', linewidth=2, markersize=6)
        ax.set_title('Natural Language Query Time', fontweight='bold')
        ax.set_xlabel('Number of Queries')
        ax.set_ylabel('Time (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 7: Insertion Throughput
        ax = axes[2, 0]
        if self.results['faiss']['insertion_times']:
            throughput_faiss = [s/t for s, t in zip(self.results['faiss']['insertion_sizes'], 
                                                   self.results['faiss']['insertion_times'])]
            ax.plot(self.results['faiss']['insertion_sizes'], throughput_faiss, 
                   'o-', label='FAISS', linewidth=2, markersize=6)
        if self.results['weaviate']['insertion_times']:
            throughput_weaviate = [s/t for s, t in zip(self.results['weaviate']['insertion_sizes'], 
                                                      self.results['weaviate']['insertion_times'])]
            ax.plot(self.results['weaviate']['insertion_sizes'], throughput_weaviate, 
                   's-', label='Weaviate', linewidth=2, markersize=6)
        ax.set_title('Insertion Throughput', fontweight='bold')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Records/Second')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 8: Memory Usage Comparison
        ax = axes[2, 1]
        operations = ['Insert', 'Query', 'Delete', 'Simple Q', 'Complex Q', 'NL Q']
        faiss_mem = []
        weaviate_mem = []
        
        # Calculate average memory usage for each operation
        for op in ['insertion', 'query', 'deletion', 'simple_query', 'complex_query', 'nl_query']:
            faiss_mem.append(np.mean(self.results['faiss'].get(f'{op}_memory', [0])))
            weaviate_mem.append(np.mean(self.results['weaviate'].get(f'{op}_memory', [0])))
        
        x = np.arange(len(operations))
        width = 0.35
        
        ax.bar(x - width/2, faiss_mem, width, label='FAISS', alpha=0.8)
        ax.bar(x + width/2, weaviate_mem, width, label='Weaviate', alpha=0.8)
        
        ax.set_title('Average Memory Usage by Operation', fontweight='bold')
        ax.set_xlabel('Operations')
        ax.set_ylabel('Memory Delta (MB)')
        ax.set_xticks(x)
        ax.set_xticklabels(operations, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 9: Query Performance Comparison
        ax = axes[2, 2]
        query_types = ['Basic', 'Simple', 'Complex', 'NL']
        faiss_query_times = []
        weaviate_query_times = []
        
        for op in ['query', 'simple_query', 'complex_query', 'nl_query']:
            faiss_query_times.append(np.mean(self.results['faiss'].get(f'{op}_times', [0])))
            weaviate_query_times.append(np.mean(self.results['weaviate'].get(f'{op}_times', [0])))
        
        x = np.arange(len(query_types))
        width = 0.35
        
        ax.bar(x - width/2, faiss_query_times, width, label='FAISS', alpha=0.8)
        ax.bar(x + width/2, weaviate_query_times, width, label='Weaviate', alpha=0.8)
        
        ax.set_title('Query Performance by Type', fontweight='bold')
        ax.set_xlabel('Query Types')
        ax.set_ylabel('Average Time (seconds)')
        ax.set_xticks(x)
        ax.set_xticklabels(query_types)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        clean_model_name = self.model_name.replace('/', '_').replace('-', '_')
        plot_path = os.path.join(output_dir, f"{clean_model_name}_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {plot_path}")

    def print_summary_report(self):
        print(f"\n{'='*80}")
        print(f"DETAILED BENCHMARK SUMMARY - {self.model_name}")
        print(f"{'='*80}")
        
        operations = [
            ('Insertion', 'insertion'),
            ('Basic Query', 'query'),
            ('Simple Query', 'simple_query'),
            ('Complex Query', 'complex_query'),
            ('Natural Language Query', 'nl_query'),
            ('Deletion', 'deletion')
        ]
        
        for op_name, op_key in operations:
            print(f"\n{op_name}:")
            print("-" * (len(op_name) + 1))
            
            # FAISS results
            faiss_times = self.results['faiss'].get(f'{op_key}_times', [])
            faiss_memory = self.results['faiss'].get(f'{op_key}_memory', [])
            
            if faiss_times:
                print(f"  FAISS - Avg Time: {np.mean(faiss_times):.4f}s, "
                      f"Avg Memory: {np.mean(faiss_memory):.2f}MB, "
                      f"Operations: {len(faiss_times)}")
            else:
                print(f"  FAISS - No data")
            
            # Weaviate results
            weaviate_times = self.results['weaviate'].get(f'{op_key}_times', [])
            weaviate_memory = self.results['weaviate'].get(f'{op_key}_memory', [])
            
            if weaviate_times:
                print(f"  Weaviate - Avg Time: {np.mean(weaviate_times):.4f}s, "
                      f"Avg Memory: {np.mean(weaviate_memory):.2f}MB, "
                      f"Operations: {len(weaviate_times)}")
            else:
                print(f"  Weaviate - No data")
            
            # Performance comparison
            if faiss_times and weaviate_times:
                faiss_avg = np.mean(faiss_times)
                weaviate_avg = np.mean(weaviate_times)
                if faiss_avg < weaviate_avg:
                    speedup = weaviate_avg / faiss_avg
                    print(f"  * FAISS is {speedup:.2f}x faster")
                else:
                    speedup = faiss_avg / weaviate_avg
                    print(f"  * Weaviate is {speedup:.2f}x faster")

def main():
    # Configuration
    models = [
        'distilbert-base-nli-stsb-mean-tokens',
        'microsoft/codebert-base',
        'bert-base-nli-mean-tokens',
        'sentence-transformers/average_word_embeddings_komninos',
        'all-mpnet-base-v2'
    ]
    
    batch_sizes = [100, 500, 1000, 2000, 5000]
    output_dir = "benchmark_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track overall results
    all_results = []
    
    print("ENHANCED FAISS vs Weaviate IP Flow Benchmark")
    print("=" * 50)
    print(f"Models to test: {len(models)}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Output directory: {output_dir}")
    print(f"Features: Insertion, Deletion, Basic/Simple/Complex/NL Queries")
    
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Processing model: {model}")
        
        try:
            benchmark = VectorDBBenchmark(model)
            
            benchmark.run_benchmark(batch_sizes)
            
            benchmark.print_summary_report()
            
            benchmark.save_results_to_csv(output_dir)
            benchmark.plot_results(output_dir)
            
            all_results.append({
                'model': model,
                'faiss_avg_insert_time': np.mean(benchmark.results['faiss']['insertion_times']) if benchmark.results['faiss']['insertion_times'] else 0,
                'faiss_avg_query_time': np.mean(benchmark.results['faiss']['query_times']) if benchmark.results['faiss']['query_times'] else 0,
                'faiss_avg_simple_query_time': np.mean(benchmark.results['faiss']['simple_query_times']) if benchmark.results['faiss']['simple_query_times'] else 0,
                'faiss_avg_complex_query_time': np.mean(benchmark.results['faiss']['complex_query_times']) if benchmark.results['faiss']['complex_query_times'] else 0,
                'faiss_avg_nl_query_time': np.mean(benchmark.results['faiss']['nl_query_times']) if benchmark.results['faiss']['nl_query_times'] else 0,
                'faiss_avg_deletion_time': np.mean(benchmark.results['faiss']['deletion_times']) if benchmark.results['faiss']['deletion_times'] else 0,
                'weaviate_avg_insert_time': np.mean(benchmark.results['weaviate']['insertion_times']) if benchmark.results['weaviate']['insertion_times'] else 0,
                'weaviate_avg_query_time': np.mean(benchmark.results['weaviate']['query_times']) if benchmark.results['weaviate']['query_times'] else 0,
                'weaviate_avg_simple_query_time': np.mean(benchmark.results['weaviate']['simple_query_times']) if benchmark.results['weaviate']['simple_query_times'] else 0,
                'weaviate_avg_complex_query_time': np.mean(benchmark.results['weaviate']['complex_query_times']) if benchmark.results['weaviate']['complex_query_times'] else 0,
                'weaviate_avg_nl_query_time': np.mean(benchmark.results['weaviate']['nl_query_times']) if benchmark.results['weaviate']['nl_query_times'] else 0,
                'weaviate_avg_deletion_time': np.mean(benchmark.results['weaviate']['deletion_times']) if benchmark.results['weaviate']['deletion_times'] else 0,
            })
            
            print(f"Completed model: {model}")
            
        except Exception as e:
            print(f"Failed to process model {model}: {e}")
            continue
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(output_dir, "enhanced_benchmark_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary report saved to {summary_path}")
        
        # Print summary
        print("\n" + "="*120)
        print("ENHANCED BENCHMARK SUMMARY")
        print("="*120)
        print(summary_df.to_string(index=False))
    
    print(f"\nEnhanced benchmark completed! Results saved in '{output_dir}' directory")
    print("\nBenchmark includes:")
    print("- Insertion performance")
    print("- Deletion performance") 
    print("- Basic vector similarity queries")
    print("- Simple structured queries (by protocol)")
    print("- Complex structured queries (by source + destination)")
    print("- Natural language queries with parameter extraction")

if __name__ == "__main__":
    main()