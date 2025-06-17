import pandas as pd
import faiss
import numpy as np
import time
import psutil
import os
import weaviate
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import random
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class CustomEmbeddings:
    def __init__(self, data, categorical_cols, continuous_cols):
        self.data = data.copy()  # Make a copy to avoid modifying original
        self.original_data = data.copy()  # Keep original for query generation
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self._preprocess_data()
        
    def _preprocess_data(self):
        # Encode categorical columns
        for col in self.categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            self.label_encoders[col] = le
            
        # Scale continuous columns
        if self.continuous_cols:
            self.data[self.continuous_cols] = self.scaler.fit_transform(self.data[self.continuous_cols])
    
    def create_query_embedding(self, query_type, query_value, embedding_method):
        # basically create meaningful query embeddings from actual data patterns
        if query_type == "protocol":
            # Flows with specific protocol
            matching_flows = self.original_data[self.original_data['_ws.col.protocol'].str.contains(query_value, case=False, na=False)]
        elif query_type == "ip":
            # Flows with specific IP (source or destination)
            matching_flows = self.original_data[
                (self.original_data['ip.src'].str.contains(query_value, case=False, na=False)) |
                (self.original_data['ip.dst'].str.contains(query_value, case=False, na=False))
            ]
        elif query_type == "port":
            # Flows with specific port
            port_val = str(query_value)
            matching_flows = self.original_data[
                (self.original_data['tcp.srcport'].astype(str) == port_val) |
                (self.original_data['tcp.dstport'].astype(str) == port_val)
            ]
        else:
            matching_flows = self.original_data.sample(1)
        
        if len(matching_flows) == 0:
            # Fallback to random sample if no matches
            matching_flows = self.original_data.sample(1)
        
        # Take first matching flow
        sample_flow = matching_flows.iloc[0:1]
        
        # Process the sample through the same pipeline
        temp_embedder = CustomEmbeddings(sample_flow, self.categorical_cols, self.continuous_cols)
        
        if embedding_method == "MLP":
            return temp_embedder.get_mlp_embeddings(embedding_dim=128)
        elif embedding_method == "Autoencoder":
            return temp_embedder.get_autoencoder_embeddings(embedding_dim=128)
        elif embedding_method == "IP2Vec":
            return temp_embedder.get_ip2vec_embeddings(embedding_dim=128)
    
    def get_mlp_embeddings(self, embedding_dim=128): 
        # Prepare data
        X_cat = self.data[self.categorical_cols].values
        X_cont = self.data[self.continuous_cols].values if self.continuous_cols else np.zeros((len(self.data), 0))
        X = np.concatenate([X_cat, X_cont], axis=1)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        # Define the model
        input_dim = X.shape[1]
        model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
        # Generate the embeddings
        embeddings = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0]
                outputs = model(inputs)
                embeddings.append(outputs.numpy())
        
        return np.concatenate(embeddings, axis=0)
    
    def get_autoencoder_embeddings(self, embedding_dim=128): 
        # Prepare data
        X_cat = self.data[self.categorical_cols].values
        X_cont = self.data[self.continuous_cols].values if self.continuous_cols else np.zeros((len(self.data), 0))
        X = np.concatenate([X_cat, X_cont], axis=1)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        # Define the autoencoder model
        input_dim = X.shape[1]
        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, embedding_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(embedding_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, input_dim)
                )
                
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded
        
        model = Autoencoder()
        
        # Train the autoencoder
        criterion = nn.MSELoss() # this is who we're minimizing
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(5):
            for batch in loader:
                inputs = batch[0]
                optimizer.zero_grad()
                _, outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
        
        # Generate embeddings
        embeddings = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0]
                encoded, _ = model(inputs)
                embeddings.append(encoded.numpy())
        
        return np.concatenate(embeddings, axis=0)
    
    def get_ip2vec_embeddings(self, embedding_dim=128):
        # Convert dataframe to sentences for word2vec- similar to what we did for BERT based models
        def bucket_len(length):
            try:
                length = float(length)
                if length < 100: return "small"
                elif length < 1500: return "medium"
                else: return "large"
            except:
                return "medium"
        
        # Create sentences from network flows
        sentences = []
        for _, row in self.data.iterrows():
            sentence = [
                f"src:{row['ip.src']}",
                f"dst:{row['ip.dst']}",
                f"sport:{row['tcp.srcport']}",
                f"dport:{row['tcp.dstport']}",
                f"proto:{row['_ws.col.protocol']}",
                f"len:{bucket_len(row['frame.len'])}"
            ]
            sentences.append(sentence)
        
        # Train the model
        model = Word2Vec(
            vector_size=embedding_dim,
            window=2,
            min_count=1,
            workers=4,
            sg=1  # skip-gram
        )
        model.build_vocab(sentences)
        model.train(sentences, total_examples=len(sentences), epochs=10)
        
        # Generate embeddings for each flow
        def embed_flow(flow, model):
            vectors = [model.wv[token] for token in flow if token in model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
        
        embeddings = np.array([embed_flow(sentence, model) for sentence in sentences])
        return embeddings

class VectorDBBenchmark:
    def __init__(self, method_name, weaviate_url="http://localhost:8080"):
        self.method_name = method_name
        self.weaviate_url = weaviate_url
        self.model_name = method_name  # For compatibility with plotting code

        try:
            self.weaviate_client = weaviate.Client(weaviate_url)
            self.weaviate_available = True
            print(f"Weaviate connection established at {weaviate_url}")
        except Exception as e:
            print(f"Weaviate connection failed: {e}")
            self.weaviate_available = False
        
        self.results = {
            'faiss': {
                'insertion_times': [], 'insertion_sizes': [], 'insertion_memory': [],
                'query_times': [], 'query_sizes': [], 'query_memory': [],
                'deletion_times': [], 'deletion_sizes': [], 'deletion_memory': [],
                'update_times': [], 'update_sizes': [], 'update_memory': []
            },
            'weaviate': {
                'insertion_times': [], 'insertion_sizes': [], 'insertion_memory': [],
                'query_times': [], 'query_sizes': [], 'query_memory': [],
                'deletion_times': [], 'deletion_sizes': [], 'deletion_memory': [],
                'update_times': [], 'update_sizes': [], 'update_memory': []
            }
        }
        
        self.faiss_index = None
        self.embeddings_cache = {}
        self.weaviate_ids = []  # Track Weaviate object IDs for deletion
        self.faiss_id_mapping = {}  # Track FAISS vector indices for deletion
        
        sample_data = {
            'ip.src': ['192.168.1.1'],
            'ip.dst': ['10.0.0.1'],
            'tcp.srcport': [80],
            'tcp.dstport': [443],
            '_ws.col.protocol': ['TCP'],
            'frame.len': [1500]
        }
        self.sample_df = pd.DataFrame(sample_data)
        self.categorical_cols = ['ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport', '_ws.col.protocol']
        self.continuous_cols = ['frame.len']

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
                    {"name": "packet_text", "dataType": ["string"]}
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
        
        # Create a temporary dataframe with the packet texts
        data = []
        for text in texts:
            parts = text.split()
            if len(parts) >= 6:
                data.append({
                    'ip.src': parts[0],
                    'ip.dst': parts[1],
                    '_ws.col.protocol': parts[2],
                    'tcp.srcport': parts[3],
                    'tcp.dstport': parts[4],
                    'frame.len': parts[5]
                })
            else:
                # Fallback values if text doesn't have enough parts
                data.append({
                    'ip.src': '192.168.1.1',
                    'ip.dst': '10.0.0.1',
                    '_ws.col.protocol': 'TCP',
                    'tcp.srcport': '80',
                    'tcp.dstport': '443',
                    'frame.len': '1500'
                })
        
        df = pd.DataFrame(data)
        embedder = CustomEmbeddings(df, self.categorical_cols, self.continuous_cols)
        
        if self.method_name == "ip2vec":
            embeddings = embedder.get_ip2vec_embeddings(embedding_dim=128)
        elif self.method_name == "mlp":
            embeddings = embedder.get_mlp_embeddings(embedding_dim=128)
        elif self.method_name == "autoencoder":
            embeddings = embedder.get_autoencoder_embeddings(embedding_dim=128)
        else:
            raise ValueError(f"Unknown method: {self.method_name}")
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.embeddings_cache[cache_key] = embeddings
        return embeddings

    def benchmark_faiss_insertion(self, packet_texts: List[str], batch_size: int):
        embeddings = self.generate_embeddings(packet_texts)
        
        if self.faiss_index is None:
            self.setup_faiss_index(embeddings.shape[1], len(embeddings))
            # Train the index
            self.faiss_index.train(embeddings)
        
        def insert_batch():
            start_id = self.faiss_index.ntotal
            self.faiss_index.add(embeddings)
            # Track IDs for deletion- store mapping of packet_text to faiss internal ID
            for i, packet_text in enumerate(packet_texts):
                self.faiss_id_mapping[packet_text] = start_id + i
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
        
        batch_ids = []
        
        def insert_batch():
            with self.weaviate_client.batch as batch:
                batch.batch_size = min(100, batch_size)
                for i, data in enumerate(packet_data):
                    embedding = self.generate_embeddings([data['packet_text']])[0]
                    
                    # Add with uuid (universally unique id) tracking
                    uuid = batch.add_data_object(
                        data_object=data,
                        class_name="IPFlow",
                        vector=embedding.tolist()
                    )
                    if uuid:
                        batch_ids.append(uuid)
            
            # Store ids for deletion
            self.weaviate_ids.extend(batch_ids)
            return len(packet_data)
        
        result, exec_time, memory_delta, success = self.measure_performance(insert_batch)
        
        if success:
            self.results['weaviate']['insertion_times'].append(exec_time)
            self.results['weaviate']['insertion_sizes'].append(batch_size)
            self.results['weaviate']['insertion_memory'].append(memory_delta)
        
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

    def benchmark_faiss_deletion(self, deletion_count: int):
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return False
        

        def delete_faiss():
            if self.faiss_index.ntotal <= deletion_count:
                # If trying to delete more than available, clear the index
                self.faiss_index.reset()
                self.faiss_id_mapping.clear()
                return deletion_count
            
            current_count = self.faiss_index.ntotal
            
            dimension = self.faiss_index.d
            nlist = min(100, max(1, int(np.sqrt(current_count - deletion_count))))
            quantizer = faiss.IndexFlatIP(dimension)
            new_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Reconstruct with remaining vectors
            remaining_vec = np.random.random((current_count - deletion_count, dimension)).astype('float32')
            if len(remaining_vec) > 0:
                new_index.train(remaining_vec)
                new_index.add(remaining_vec)
            
            self.faiss_index = new_index
            
            # Update id mapping (remove deleted entries)
            items_to_remove = list(self.faiss_id_mapping.keys())[:deletion_count]
            for item in items_to_remove:
                del self.faiss_id_mapping[item]
            
            return deletion_count
        
        result, exec_time, memory_delta, success = self.measure_performance(delete_faiss)
        
        if success:
            self.results['faiss']['deletion_times'].append(exec_time)
            self.results['faiss']['deletion_sizes'].append(deletion_count)
            self.results['faiss']['deletion_memory'].append(memory_delta)
        
        return success

    def benchmark_weaviate_deletion(self, deletion_count: int):
        if not self.weaviate_available or len(self.weaviate_ids) == 0:
            return False
        
        ids_to_delete = self.weaviate_ids[:min(deletion_count, len(self.weaviate_ids))]
        
        def delete_batch():
            deleted_count = 0
            for obj_id in ids_to_delete:
                try:
                    self.weaviate_client.data_object.delete(obj_id, "IPFlow")
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete object {obj_id}: {e}")
            
            # Remove deleted ids from tracking
            for obj_id in ids_to_delete:
                if obj_id in self.weaviate_ids:
                    self.weaviate_ids.remove(obj_id)
            
            return deleted_count
        
        result, exec_time, memory_delta, success = self.measure_performance(delete_batch)
        
        if success:
            self.results['weaviate']['deletion_times'].append(exec_time)
            self.results['weaviate']['deletion_sizes'].append(deletion_count)
            self.results['weaviate']['deletion_memory'].append(memory_delta)
        
        return success

    def create_sample_data(self, num_samples: int) -> Tuple[List[str], List[Dict]]:
        packet_texts = []
        packet_data = []
        
        for i in range(num_samples):
            source_ip = f"192.168.{i%256}.{(i*7)%256}"
            dest_ip = f"10.0.{(i*3)%256}.{(i*11)%256}"
            protocol = ["TCP", "UDP", "HTTP", "HTTPS", "DNS"][i % 5]
            src_port = (i * 13) % 65535
            dst_port = (i * 17) % 65535
            frame_len = (i * 100) % 1500
            
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
                "packet_text": packet_text
            })
        
        return packet_texts, packet_data

    def run_benchmark(self, batch_sizes: List[int] = [100, 500, 1000, 2000, 5000]):
        print(f"\n{'='*60}")
        print(f"BENCHMARKING METHOD: {self.method_name}")
        print(f"{'='*60}")
        
        # Setup
        if self.weaviate_available:
            self.setup_weaviate_schema()
        
        # Run benchmarks for different batch sizes
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Create sample data
            packet_texts, packet_data = self.create_sample_data(batch_size)
            
            # FAISS Insertion
            print("  FAISS insertion...", end=" ")
            faiss_insert_success = self.benchmark_faiss_insertion(packet_texts, batch_size)
            print("done" if faiss_insert_success else "ERROR")
            
            # Weaviate Insertion
            if self.weaviate_available:
                print("  Weaviate insertion...", end=" ")
                weaviate_insert_success = self.benchmark_weaviate_insertion(packet_data, batch_size)
                print("done" if weaviate_insert_success else "ERROR")
            
            # Query benchmarks
            query_texts = [
                "TCP flow 192.168.1.1 80",
                "UDP DNS traffic",
                "HTTP web traffic",
                "HTTPS secure connection",
                "192.168.0.1 10.0.0.1 TCP"
            ]
            
            # FAISS Query
            if faiss_insert_success:
                print("  FAISS query...", end=" ")
                faiss_query_success = self.benchmark_faiss_query(query_texts)
                print("done" if faiss_query_success else "ERROR")
            
            # Weaviate Query
            if self.weaviate_available and weaviate_insert_success:
                print("  Weaviate query...", end=" ")
                weaviate_query_success = self.benchmark_weaviate_query(query_texts)
                print("done" if weaviate_query_success else "ERROR")
            
            # Deletion benchmarks (delete 10% of inserted data)
            deletion_count = max(1, batch_size // 10)
            
            # FAISS Deletion
            if faiss_insert_success:
                print("  FAISS deletion...", end=" ")
                faiss_delete_success = self.benchmark_faiss_deletion(deletion_count)
                print("done" if faiss_delete_success else "ERROR")
            
            # Weaviate Deletion
            if self.weaviate_available and weaviate_insert_success:
                print("  Weaviate deletion...", end=" ")
                weaviate_delete_success = self.benchmark_weaviate_deletion(deletion_count)
                print("done" if weaviate_delete_success else "ERROR")

    def save_results_to_csv(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        csv_data = []
        
        for db_type in ['faiss', 'weaviate']:
            for operation in ['insertion', 'query', 'deletion']:
                times = self.results[db_type][f'{operation}_times']
                sizes = self.results[db_type][f'{operation}_sizes']
                memory = self.results[db_type][f'{operation}_memory']
                
                for i in range(len(times)):
                    csv_data.append({
                        'model': self.method_name,
                        'database': db_type,
                        'operation': operation,
                        'batch_size': sizes[i],
                        'execution_time': times[i],
                        'memory_delta_mb': memory[i],
                        'throughput': sizes[i] / times[i] if times[i] > 0 else 0
                    })
        
        df = pd.DataFrame(csv_data)
        clean_model_name = self.method_name.replace('/', '_').replace('-', '_')
        csv_path = os.path.join(output_dir, f"{clean_model_name}_benchmark.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    def plot_results(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle(f'FAISS vs Weaviate Performance - {self.method_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Insertion Time vs Batch Size
        ax = axes[0, 0]
        if self.results['faiss']['insertion_times']:
            ax.plot(self.results['faiss']['insertion_sizes'], self.results['faiss']['insertion_times'], 
                   'o-', label='FAISS', color='blue', linewidth=2, markersize=6)
        if self.results['weaviate']['insertion_times']:
            ax.plot(self.results['weaviate']['insertion_sizes'], self.results['weaviate']['insertion_times'], 
                   's-', label='Weaviate', color='red', linewidth=2, markersize=6)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Insertion Time (seconds)')
        ax.set_title('Insertion Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Query Time vs Query Size
        ax = axes[0, 1]
        if self.results['faiss']['query_times']:
            ax.plot(self.results['faiss']['query_sizes'], self.results['faiss']['query_times'], 
                   'o-', label='FAISS', color='blue', linewidth=2, markersize=6)
        if self.results['weaviate']['query_times']:
            ax.plot(self.results['weaviate']['query_sizes'], self.results['weaviate']['query_times'], 
                   's-', label='Weaviate', color='red', linewidth=2, markersize=6)
        ax.set_xlabel('Query Batch Size')
        ax.set_ylabel('Query Time (seconds)')
        ax.set_title('Query Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Deletion Time vs Deletion Size
        ax = axes[0, 2]
        if self.results['faiss']['deletion_times']:
            ax.plot(self.results['faiss']['deletion_sizes'], self.results['faiss']['deletion_times'], 
                   'o-', label='FAISS', color='blue', linewidth=2, markersize=6)
        if self.results['weaviate']['deletion_times']:
            ax.plot(self.results['weaviate']['deletion_sizes'], self.results['weaviate']['deletion_times'], 
                   's-', label='Weaviate', color='red', linewidth=2, markersize=6)
        ax.set_xlabel('Deletion Count')
        ax.set_ylabel('Deletion Time (seconds)')
        ax.set_title('Deletion Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Insertion Throughput
        ax = axes[1, 0]
        if self.results['faiss']['insertion_times']:
            faiss_throughput = [size / time for size, time in zip(self.results['faiss']['insertion_sizes'], 
                                                                 self.results['faiss']['insertion_times'])]
            ax.plot(self.results['faiss']['insertion_sizes'], faiss_throughput, 
                   'o-', label='FAISS', color='blue', linewidth=2, markersize=6)
        if self.results['weaviate']['insertion_times']:
            weaviate_throughput = [size / time for size, time in zip(self.results['weaviate']['insertion_sizes'], 
                                                                    self.results['weaviate']['insertion_times'])]
            ax.plot(self.results['weaviate']['insertion_sizes'], weaviate_throughput, 
                   's-', label='Weaviate', color='red', linewidth=2, markersize=6)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Throughput (records/second)')
        ax.set_title('Insertion Throughput')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Memory Usage- Insertion
        ax = axes[1, 1]
        if self.results['faiss']['insertion_memory']:
            ax.plot(self.results['faiss']['insertion_sizes'], self.results['faiss']['insertion_memory'], 
                   'o-', label='FAISS', color='blue', linewidth=2, markersize=6)
        if self.results['weaviate']['insertion_memory']:
            ax.plot(self.results['weaviate']['insertion_sizes'], self.results['weaviate']['insertion_memory'], 
                   's-', label='Weaviate', color='red', linewidth=2, markersize=6)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Memory Delta (MB)')
        ax.set_title('Insertion Memory Usage')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Memory Usage- Query
        ax = axes[1, 2]
        if self.results['faiss']['query_memory']:
            ax.plot(self.results['faiss']['query_sizes'], self.results['faiss']['query_memory'], 
                   'o-', label='FAISS', color='blue', linewidth=2, markersize=6)
        if self.results['weaviate']['query_memory']:
            ax.plot(self.results['weaviate']['query_sizes'], self.results['weaviate']['query_memory'], 
                   's-', label='Weaviate', color='red', linewidth=2, markersize=6)
        ax.set_xlabel('Query Batch Size')
        ax.set_ylabel('Memory Delta (MB)')
        ax.set_title('Query Memory Usage')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 7: Combined Performance Score (lower is better)
        ax = axes[2, 0]
        if (self.results['faiss']['insertion_times'] and self.results['faiss']['query_times']):
            faiss_scores = [(t1 + t2) for t1, t2 in zip(self.results['faiss']['insertion_times'], 
                                                        self.results['faiss']['query_times'][:len(self.results['faiss']['insertion_times'])])]
            ax.plot(self.results['faiss']['insertion_sizes'], faiss_scores, 
                   'o-', label='FAISS', color='blue', linewidth=2, markersize=6)
        if (self.results['weaviate']['insertion_times'] and self.results['weaviate']['query_times']):
            weaviate_scores = [(t1 + t2) for t1, t2 in zip(self.results['weaviate']['insertion_times'], 
                                                           self.results['weaviate']['query_times'][:len(self.results['weaviate']['insertion_times'])])]
            ax.plot(self.results['weaviate']['insertion_sizes'], weaviate_scores, 
                   's-', label='Weaviate', color='red', linewidth=2, markersize=6)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Combined Time (Insert + Query)')
        ax.set_title('Combined Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 8: Scalability Analysis
        ax = axes[2, 1]
        if len(self.results['faiss']['insertion_times']) > 1:
            faiss_scale = [t / self.results['faiss']['insertion_times'][0] for t in self.results['faiss']['insertion_times']]
            ax.plot(self.results['faiss']['insertion_sizes'], faiss_scale, 
                   'o-', label='FAISS', color='blue', linewidth=2, markersize=6)
        if len(self.results['weaviate']['insertion_times']) > 1:
            weaviate_scale = [t / self.results['weaviate']['insertion_times'][0] for t in self.results['weaviate']['insertion_times']]
            ax.plot(self.results['weaviate']['insertion_sizes'], weaviate_scale, 
                   's-', label='Weaviate', color='red', linewidth=2, markersize=6)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Relative Time (normalized)')
        ax.set_title('Scalability Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 9: Performance Summary Table
        ax = axes[2, 2]
        ax.axis('off')
        
        # Create summary statistics
        summary_data = []
        for db_type in ['FAISS', 'Weaviate']:
            db_key = db_type.lower()
            if self.results[db_key]['insertion_times']:
                avg_insert = np.mean(self.results[db_key]['insertion_times'])
                avg_query = np.mean(self.results[db_key]['query_times']) if self.results[db_key]['query_times'] else 0
                avg_delete = np.mean(self.results[db_key]['deletion_times']) if self.results[db_key]['deletion_times'] else 0
                summary_data.append([db_type, f"{avg_insert:.3f}s", f"{avg_query:.3f}s", f"{avg_delete:.3f}s"])
        
        if summary_data:
            table = ax.table(cellText=summary_data,
                           colLabels=['Database', 'Avg Insert', 'Avg Query', 'Avg Delete'],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            ax.set_title('Performance Summary')
        
        plt.tight_layout()
        
        # Save plot
        clean_model_name = self.method_name.replace('/', '_').replace('-', '_')
        plot_path = os.path.join(output_dir, f"{clean_model_name}_performance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Performance plots saved to {plot_path}")
        plt.show()

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY - {self.method_name}")
        print(f"{'='*60}")
        
        for db_type in ['faiss', 'weaviate']:
            if any(self.results[db_type]['insertion_times']):
                print(f"\n{db_type.upper()} Results:")
                print(f"  Insertion - avg: {np.mean(self.results[db_type]['insertion_times']):.3f}s, "
                      f"min: {np.min(self.results[db_type]['insertion_times']):.3f}s, "
                      f"max: {np.max(self.results[db_type]['insertion_times']):.3f}s")
                
                if self.results[db_type]['query_times']:
                    print(f"  Query - avg: {np.mean(self.results[db_type]['query_times']):.3f}s, "
                          f"min: {np.min(self.results[db_type]['query_times']):.3f}s, "
                          f"max: {np.max(self.results[db_type]['query_times']):.3f}s")
                
                if self.results[db_type]['deletion_times']:
                    print(f"  Deletion - avg: {np.mean(self.results[db_type]['deletion_times']):.3f}s, "
                          f"min: {np.min(self.results[db_type]['deletion_times']):.3f}s, "
                          f"max: {np.max(self.results[db_type]['deletion_times']):.3f}s")
                
                # Calculate throughput
                if self.results[db_type]['insertion_times'] and self.results[db_type]['insertion_sizes']:
                    throughput = [size / time for size, time in zip(self.results[db_type]['insertion_sizes'], 
                                                                   self.results[db_type]['insertion_times'])]
                    print(f"  Avg Throughput: {np.mean(throughput):.2f} records/second")

def run_comprehensive_benchmark():
    methods = ["ip2vec", "mlp", "autoencoder"]
    batch_sizes = [100, 500, 1000, 2000, 5000]
    output_dir = "benchmark_results"
    
    print("Starting Comprehensive Vector Database Benchmark")
    print(f"Testing methods: {methods}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Output directory: {output_dir}")
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"TESTING METHOD: {method.upper()}")
        print(f"{'='*80}")
        
        try:
            benchmark = VectorDBBenchmark(method)
            
            benchmark.run_benchmark(batch_sizes)
            
            benchmark.print_summary()
            
            benchmark.save_results_to_csv(output_dir)
            
            benchmark.plot_results(output_dir)
            
        except Exception as e:
            print(f"Error testing {method}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")

def compare_all_methods(output_dir: str = "benchmark_results"):
    import glob
    
    # Load all CSV files
    csv_files = glob.glob(os.path.join(output_dir, "*_benchmark.csv"))
    
    if not csv_files:
        print("No benchmark results found. Please run benchmarks first.")
        return
    
    # Combine all results
    all_results = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_results.append(df)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Embedding Methods Comparison - FAISS vs Weaviate', fontsize=16, fontweight='bold')
    
    operations = ['insertion', 'query', 'deletion']
    colors = {'ip2vec': 'blue', 'mlp': 'green', 'autoencoder': 'red'}
    
    for i, operation in enumerate(operations):
        # FAISS comparison
        ax = axes[0, i]
        for method in combined_df['model'].unique():
            data = combined_df[(combined_df['model'] == method) & 
                              (combined_df['database'] == 'faiss') & 
                              (combined_df['operation'] == operation)]
            if not data.empty:
                ax.plot(data['batch_size'], data['execution_time'], 
                       'o-', label=method, color=colors.get(method, 'black'), 
                       linewidth=2, markersize=6)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'FAISS {operation.title()} Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Weaviate comparison
        ax = axes[1, i]
        for method in combined_df['model'].unique():
            data = combined_df[(combined_df['model'] == method) & 
                              (combined_df['database'] == 'weaviate') & 
                              (combined_df['operation'] == operation)]
            if not data.empty:
                ax.plot(data['batch_size'], data['execution_time'], 
                       's-', label=method, color=colors.get(method, 'black'), 
                       linewidth=2, markersize=6)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'Weaviate {operation.title()} Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "methods_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {comparison_path}")
    plt.show()

if __name__ == "__main__":
    # Run benchmark
    run_comprehensive_benchmark()
    
    compare_all_methods()
    
    print("\nBenchmark complete! Check the 'benchmark_results' directory for detailed results.")