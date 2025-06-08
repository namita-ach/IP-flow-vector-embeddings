import pandas as pd
import faiss
import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import random

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
            # Random sample as fallback
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

class FAISSPerformanceTracker:
    def __init__(self, method_name):
        self.method_name = method_name
        
        # Lists for our graohs
        self.insertion_sizes = []
        self.deletion_sizes = []
        self.update_sizes = []
        self.query_sizes = []
        
        self.insertion_times = []
        self.deletion_times = []
        self.update_times = []
        self.query_times = []
        
        self.insertion_cpu = []
        self.deletion_cpu = []
        self.update_cpu = []
        self.query_cpu = []
        
        self.insertion_memory = []
        self.deletion_memory = []
        self.update_memory = []
        self.query_memory = []

    def _measure_performance(self, func, *args, **kwargs):
        pid = os.getpid()
        process = psutil.Process(pid)
        
        # Initial resource usage
        start_cpu = process.cpu_percent(interval=None)
        start_mem = process.memory_info().rss / (1024 ** 2)  # Convert to MB
        start_time = time.time()
        
        # Run function
        result = func(*args, **kwargs)
        
        # Final resource usage
        end_cpu = process.cpu_percent(interval=None)
        end_mem = process.memory_info().rss / (1024 ** 2)  # Convert to MB
        end_time = time.time()
        
        # Compute differences
        cpu_usage = end_cpu - start_cpu
        mem_usage = end_mem - start_mem
        execution_time = end_time - start_time
        
        return result, execution_time, cpu_usage, mem_usage

    def track_insertion(self, index, new_embeddings):
        # Normalize embeddings
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._insert_embeddings, index, new_embeddings
        )
        
        # Track metrics
        self.insertion_sizes.append(len(new_embeddings))
        self.insertion_times.append(time_taken)
        self.insertion_cpu.append(cpu_usage)
        self.insertion_memory.append(mem_usage)

    def track_deletion(self, index, num_deletions):
        # Select random indices to delete
        delete_indices = np.random.choice(index.ntotal, num_deletions, replace=False)
        
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._delete_embeddings, index, delete_indices
        )
        
        # Track metrics
        self.deletion_sizes.append(num_deletions)
        self.deletion_times.append(time_taken)
        self.deletion_cpu.append(cpu_usage)
        self.deletion_memory.append(mem_usage)

    def track_update(self, index, num_updates, new_embeddings):
        # Select random indices to update
        update_indices = np.random.choice(index.ntotal, num_updates, replace=False)
        
        # Normalize new embeddings
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._update_embeddings, index, update_indices, new_embeddings
        )
        
        # Track metrics
        self.update_sizes.append(num_updates)
        self.update_times.append(time_taken)
        self.update_cpu.append(cpu_usage)
        self.update_memory.append(mem_usage)

    def _insert_embeddings(self, index, new_embeddings):
        index.add(new_embeddings)
        return index

    def _delete_embeddings(self, index, delete_indices):
        index.remove_ids(delete_indices)
        return index

    def _update_embeddings(self, index, update_indices, new_embeddings):
        index.remove_ids(update_indices)
        index.add(new_embeddings)
        return index

    def plot_performance_metrics(self, save_path=None):
        plt.figure(figsize=(15, 5))
        
        # Time plots
        plt.subplot(1, 3, 1)
        plt.plot(self.insertion_sizes, self.insertion_times, label='Insertion', marker='o')
        plt.plot(self.deletion_sizes, self.deletion_times, label='Deletion', marker='s')
        plt.plot(self.update_sizes, self.update_times, label='Update', marker='^')
        plt.plot(self.query_sizes, self.query_times, label='Query', marker='x')
        plt.title('Execution Time')
        plt.xlabel('Number of Embeddings / Queries')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.grid(True)

        # CPU usage
        plt.subplot(1, 3, 2)
        plt.plot(self.insertion_sizes, self.insertion_cpu, label='Insertion', marker='o')
        plt.plot(self.deletion_sizes, self.deletion_cpu, label='Deletion', marker='s')
        plt.plot(self.update_sizes, self.update_cpu, label='Update', marker='^')
        plt.plot(self.query_sizes, self.query_cpu, label='Query', marker='x')
        plt.title('CPU Usage')
        plt.xlabel('Number of Embeddings / Queries')
        plt.ylabel('CPU Percentage')
        plt.legend()
        plt.grid(True)

        # Memory usage
        plt.subplot(1, 3, 3)
        plt.plot(self.insertion_sizes, self.insertion_memory, label='Insertion', marker='o')
        plt.plot(self.deletion_sizes, self.deletion_memory, label='Deletion', marker='s')
        plt.plot(self.update_sizes, self.update_memory, label='Update', marker='^')
        plt.plot(self.query_sizes, self.query_memory, label='Query', marker='x')
        plt.title('Memory Usage')
        plt.xlabel('Number of Embeddings / Queries')
        plt.ylabel('Memory (MB)')
        plt.legend()
        plt.grid(True)

        plt.suptitle(f'Performance Metrics for {self.method_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {save_path}")
        
        plt.show()

    def query_top_k(self, index, query_embeddings, k=5):
        # Normalize query embeddings
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        distances, indices = index.search(query_embeddings, k)
        
        # For normalized vectors: cosine_similarity = 1 - (L2_distance^2 / 2)
        cosine_similarities = 1 - (distances ** 2) / 2
        
        return cosine_similarities, indices
    
    def track_query(self, index, query_embeddings, k=5):
        num_queries = len(query_embeddings)
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._query_embeddings, index, query_embeddings, k
        )
        # Track metrics
        self.query_sizes.append(num_queries)
        self.query_times.append(time_taken)
        self.query_cpu.append(cpu_usage)
        self.query_memory.append(mem_usage)

    def _query_embeddings(self, index, query_embeddings, k):
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        index.nprobe = 10
        return index.search(query_embeddings, k)

def main(): 
    csv_path = "/kaggle/input/5-tuple-ip-flow-data/ip_flow_dataset.csv" 
    
    df = pd.read_csv(
        csv_path,
        header=0,
        names=[
            "frame.number", "frame.time", "ip.src", "ip.dst",
            "tcp.srcport", "tcp.dstport", "_ws.col.protocol", "frame.len"
        ],
        dtype=str,
        skiprows=1
    )

    # Convert some of the columns to numeric
    df["frame.len"] = pd.to_numeric(df["frame.len"], errors='coerce').fillna(0)
    df["tcp.srcport"] = pd.to_numeric(df["tcp.srcport"].fillna(0), errors='coerce')
    df["tcp.dstport"] = pd.to_numeric(df["tcp.dstport"].fillna(0), errors='coerce')
    
    # Define categorical and continuous columns
    categorical_cols = ["ip.src", "ip.dst", "_ws.col.protocol"]
    continuous_cols = ["frame.len", "tcp.srcport", "tcp.dstport"]
    
    # Initialize custom embeddings
    custom_embedder = CustomEmbeddings(df, categorical_cols, continuous_cols)
    
    # Define embedding methods to test
    embedding_methods = {
        "MLP": custom_embedder.get_mlp_embeddings,
        "Autoencoder": custom_embedder.get_autoencoder_embeddings,
        "IP2Vec": custom_embedder.get_ip2vec_embeddings
    }
    
    plots_dir = "PipelineResults"
    os.makedirs(plots_dir, exist_ok=True)
    
    for method_name, embed_method in embedding_methods.items():
        print(f"\nProcessing embedding method: {method_name}")
        
        # Generate embeddings
        embeddings = embed_method(embedding_dim=128)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Creating FAISS Inner Product index for cosine similarity
        dimension = embeddings.shape[1]
        nlist = min(100, int(np.sqrt(len(embeddings))))
        quantizer = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        
        # Initialize performance tracker
        tracker = FAISSPerformanceTracker(method_name)
        
        # Track performance for x no of operations
        for num_ops in [2500, 5000, 7500, 10000, 20000, 30000]:
            print(f"  Running {num_ops} operations")
            
            # synthetic embeddings for operations
            synthetic_embeddings = np.random.randn(num_ops, dimension).astype('float32')
            
            # Insertion tracking
            tracker.track_insertion(index, synthetic_embeddings)
            
            # Deletion tracking
            tracker.track_deletion(index, num_ops)
            
            # Update tracking
            update_embeddings = np.random.randn(num_ops, dimension).astype('float32')
            tracker.track_update(index, num_ops, update_embeddings)
            
            # Query tracking with actual meaningful queries
            query_embeddings = np.random.randn(num_ops, dimension).astype('float32')
            tracker.track_query(index, query_embeddings, k=5)
            
            if num_ops == 10000:
                
                # Query 1: Find flows with TCP
                try:
                    tcp_query = custom_embedder.create_query_embedding("protocol", "TCP", method_name)
                    if tcp_query is not None:
                        similarities, indices = tracker.query_top_k(index, tcp_query, k=5)
                        print(f"Method: {method_name}")
                        print(f"Query: 'TCP flows'")
                        print(f"Top 5 Similar Flow Indices: {indices[0]}")
                        print(f"Cosine Similarities: {similarities[0]}")
                        print(f"Average Cosine Similarity: {sum(similarities[0])/5}")
                except Exception as e:
                    print(f"TCP query failed: {e}")
                
                print()
                
                # Query 2: Find flows with specific IP pattern
                try:
                    ip_query = custom_embedder.create_query_embedding("ip", "192.168", method_name)
                    if ip_query is not None:
                        similarities, indices = tracker.query_top_k(index, ip_query, k=5)
                        print(f"Query: 'Flows with 192.168.x.x IPs'")
                        print(f"Top 5 Similar Flow Indices: {indices[0]}")
                        print(f"Cosine Similarities: {similarities[0]}")
                except Exception as e:
                    print(f"IP query failed: {e}")
                
                print()
                
                # Query 3: Find flows with specific port
                try:
                    port_query = custom_embedder.create_query_embedding("port", "80", method_name)
                    if port_query is not None:
                        similarities, indices = tracker.query_top_k(index, port_query, k=5)
                        print(f"Query: 'Flows using port 80'")
                        print(f"Top 5 Similar Flow Indices: {indices[0]}")
                        print(f"Cosine Similarities: {similarities[0]}")
                except Exception as e:
                    print(f"Port query failed: {e}")
        
        plot_filename = f"{plots_dir}/{method_name}.png"
        tracker.plot_performance_metrics(save_path=plot_filename)

if __name__ == "__main__":
    main()
