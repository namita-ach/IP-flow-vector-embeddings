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
import random

class AutoencoderEmbeddings:
    def __init__(self, data, categorical_cols, continuous_cols):
        self.data = data.copy()
        self.original_data = data.copy()
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
    
    def create_query_embedding(self, query_type, query_value):
        if query_type == "protocol":
            matching_flows = self.original_data[self.original_data['_ws.col.protocol'].str.contains(query_value, case=False, na=False)]
        elif query_type == "ip":
            matching_flows = self.original_data[
                (self.original_data['ip.src'].str.contains(query_value, case=False, na=False)) |
                (self.original_data['ip.dst'].str.contains(query_value, case=False, na=False))
            ]
        elif query_type == "port":
            port_val = str(query_value)
            matching_flows = self.original_data[
                (self.original_data['tcp.srcport'].astype(str) == port_val) |
                (self.original_data['tcp.dstport'].astype(str) == port_val)
            ]
        else:
            matching_flows = self.original_data.sample(1)
        
        if len(matching_flows) == 0:
            matching_flows = self.original_data.sample(1)
        
        sample_flow = matching_flows.iloc[0:1]
        temp_embedder = AutoencoderEmbeddings(sample_flow, self.categorical_cols, self.continuous_cols)
        return temp_embedder.get_autoencoder_embeddings(embedding_dim=128)
    
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
        criterion = nn.MSELoss()
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

class FAISSPerformanceTracker:
    def __init__(self, method_name):
        self.method_name = method_name
        
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
        
        start_cpu = process.cpu_percent(interval=None)
        start_mem = process.memory_info().rss / (1024 ** 2)
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_cpu = process.cpu_percent(interval=None)
        end_mem = process.memory_info().rss / (1024 ** 2)
        end_time = time.time()
        
        cpu_usage = end_cpu - start_cpu
        mem_usage = end_mem - start_mem
        execution_time = end_time - start_time
        
        return result, execution_time, cpu_usage, mem_usage

    def track_insertion(self, index, new_embeddings):
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._insert_embeddings, index, new_embeddings
        )
        
        self.insertion_sizes.append(len(new_embeddings))
        self.insertion_times.append(time_taken)
        self.insertion_cpu.append(cpu_usage)
        self.insertion_memory.append(mem_usage)

    def track_deletion(self, index, num_deletions):
        delete_indices = np.random.choice(index.ntotal, num_deletions, replace=False)
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._delete_embeddings, index, delete_indices
        )
        
        self.deletion_sizes.append(num_deletions)
        self.deletion_times.append(time_taken)
        self.deletion_cpu.append(cpu_usage)
        self.deletion_memory.append(mem_usage)

    def track_update(self, index, num_updates, new_embeddings):
        update_indices = np.random.choice(index.ntotal, num_updates, replace=False)
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._update_embeddings, index, update_indices, new_embeddings
        )
        
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
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        distances, indices = index.search(query_embeddings, k)
        cosine_similarities = 1 - (distances ** 2) / 2
        return cosine_similarities, indices
    
    def track_query(self, index, query_embeddings, k=5):
        num_queries = len(query_embeddings)
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._query_embeddings, index, query_embeddings, k
        )
        self.query_sizes.append(num_queries)
        self.query_times.append(time_taken)
        self.query_cpu.append(cpu_usage)
        self.query_memory.append(mem_usage)

    def _query_embeddings(self, index, query_embeddings, k):
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        index.nprobe = 10
        return index.search(query_embeddings, k)

def main(): 
    csv_path = "ip_flow_dataset.csv"
    
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

    # Convert columns to numeric
    df["frame.len"] = pd.to_numeric(df["frame.len"], errors='coerce').fillna(0)
    df["tcp.srcport"] = pd.to_numeric(df["tcp.srcport"].fillna(0), errors='coerce')
    df["tcp.dstport"] = pd.to_numeric(df["tcp.dstport"].fillna(0), errors='coerce')
    
    categorical_cols = ["ip.src", "ip.dst", "_ws.col.protocol"]
    continuous_cols = ["frame.len", "tcp.srcport", "tcp.dstport"]
    
    # Initialize autoencoder embeddings
    autoencoder_embedder = AutoencoderEmbeddings(df, categorical_cols, continuous_cols)
    
    plots_dir = "AutoencoderResults"
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Processing Autoencoder embedding method")
    
    # Generate embeddings
    embeddings = autoencoder_embedder.get_autoencoder_embeddings(embedding_dim=128)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    nlist = min(100, int(np.sqrt(len(embeddings))))
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    
    # Initialize performance tracker
    tracker = FAISSPerformanceTracker("Autoencoder")
    
    # Track performance for different operation sizes
    for num_ops in [2500, 5000, 7500, 10000, 20000, 30000]:
        print(f"  Running {num_ops} operations")
        
        synthetic_embeddings = np.random.randn(num_ops, dimension).astype('float32')
        
        # Track operations
        tracker.track_insertion(index, synthetic_embeddings)
        tracker.track_deletion(index, num_ops)
        
        update_embeddings = np.random.randn(num_ops, dimension).astype('float32')
        tracker.track_update(index, num_ops, update_embeddings)
        
        query_embeddings = np.random.randn(num_ops, dimension).astype('float32')
        tracker.track_query(index, query_embeddings, k=5)
        
        if num_ops == 10000:
            # Test meaningful queries
            try:
                tcp_query = autoencoder_embedder.create_query_embedding("protocol", "TCP")
                if tcp_query is not None:
                    similarities, indices = tracker.query_top_k(index, tcp_query, k=5)
                    print(f"Query: 'TCP flows'")
                    print(f"Top 5 Similar Flow Indices: {indices[0]}")
                    print(f"Cosine Similarities: {similarities[0]}")
                    print(f"Average Cosine Similarity: {sum(similarities[0])/5}")
            except Exception as e:
                print(f"TCP query failed: {e}")
            
            print()
            
            try:
                ip_query = autoencoder_embedder.create_query_embedding("ip", "192.168")
                if ip_query is not None:
                    similarities, indices = tracker.query_top_k(index, ip_query, k=5)
                    print(f"Query: 'Flows with 192.168.x.x IPs'")
                    print(f"Top 5 Similar Flow Indices: {indices[0]}")
                    print(f"Cosine Similarities: {similarities[0]}")
            except Exception as e:
                print(f"IP query failed: {e}")
            
            print()
            
            try:
                port_query = autoencoder_embedder.create_query_embedding("port", "80")
                if port_query is not None:
                    similarities, indices = tracker.query_top_k(index, port_query, k=5)
                    print(f"Query: 'Flows using port 80'")
                    print(f"Top 5 Similar Flow Indices: {indices[0]}")
                    print(f"Cosine Similarities: {similarities[0]}")
            except Exception as e:
                print(f"Port query failed: {e}")
    
    plot_filename = f"{plots_dir}/Autoencoder_Performance.png"
    tracker.plot_performance_metrics(save_path=plot_filename)

if __name__ == "__main__":
    main()
