import pandas as pd
import faiss
import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

class FAISSPerformanceTracker:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        # Performance tracking lists
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

    def track_insertion(self, index, new_packet_texts):
        # Generate and normalize embeddings
        new_embeddings = self.model.encode(new_packet_texts, convert_to_numpy=True)
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

    def track_update(self, index, num_updates, new_packet_texts):
        # Select random indices to update
        update_indices = np.random.choice(index.ntotal, num_updates, replace=False)
        
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._update_embeddings, index, update_indices, new_packet_texts
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

    def _update_embeddings(self, index, update_indices, new_packet_texts):
        index.remove_ids(update_indices)
        
        # Compute and normalize new embeddings
        new_embeddings = self.model.encode(new_packet_texts, convert_to_numpy=True)
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        
        # Add new embeddings
        index.add(new_embeddings)
        return index

    def plot_performance_metrics(self, save_path=None):
        plt.figure(figsize=(15, 5))
        
        # Time 
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

        plt.suptitle(f'Performance Metrics for {self.model_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {save_path}")
        
        plt.show()

    def query_top_k(self, index, query_texts, k=5):
        # Generate and normalize query embeddings
        query_embeddings = self.model.encode(query_texts, convert_to_numpy=True)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        # Search for top-k using inner product (cosine similarity for normalized vectors)
        distances, indices = index.search(query_embeddings, k)
        
        return distances, indices  # distances are cosine similarities

    def track_query(self, index, query_texts, k=5):
        num_queries = len(query_texts)
        
        # Measure performance
        _, time_taken, cpu_usage, mem_usage = self._measure_performance(
            self._query_embeddings, index, query_texts, k
        )
        
        # Track metrics
        self.query_sizes.append(num_queries)
        self.query_times.append(time_taken)
        self.query_cpu.append(cpu_usage)
        self.query_memory.append(mem_usage)

    def _query_embeddings(self, index, query_texts, k):
        query_embeddings = self.model.encode(query_texts, convert_to_numpy=True)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        index.nprobe = 10
        return index.search(query_embeddings, k)

    def create_meaningful_queries(self, df):
        queries = []
        
        # Query 1: TCP flows
        tcp_flows = df[df['_ws.col.protocol'].str.contains('TCP', case=False, na=False)]
        if len(tcp_flows) > 0:
            sample = tcp_flows.iloc[0]
            queries.append(f"{sample['ip.src']} {sample['ip.dst']} TCP {sample['tcp.srcport']} {sample['tcp.dstport']}")
        
        # Query 2: HTTP flows (port 80)
        http_flows = df[(df['tcp.dstport'] == '80') | (df['tcp.srcport'] == '80')]
        if len(http_flows) > 0:
            sample = http_flows.iloc[0]
            queries.append(f"{sample['ip.src']} {sample['ip.dst']} {sample['_ws.col.protocol']} 80")
        
        # Query 3: Specific IP pattern
        local_ips = df[df['ip.src'].str.contains('192.168', case=False, na=False)]
        if len(local_ips) > 0:
            sample = local_ips.iloc[0]
            queries.append(f"192.168 {sample['ip.dst']} {sample['_ws.col.protocol']}")
        
        # Add generic queries if no specific patterns found
        if len(queries) == 0:
            queries = [
                "192.168.1.1 192.168.1.2 TCP 80",
                "10.0.0.1 10.0.0.2 UDP 53",
                "172.16.1.1 172.16.1.2 HTTP 443"
            ]
        
        return queries

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
        skiprows=1,
        nrows=30000
    )

    # Convert relevant columns to appropriate types
    df["frame.len"] = pd.to_numeric(df["frame.len"], errors='coerce').fillna(0)
    df["tcp.srcport"] = df["tcp.srcport"].fillna('0')
    df["tcp.dstport"] = df["tcp.dstport"].fillna('0')

    # Create packet text representation
    df["packet_text"] = (
        df["ip.src"].fillna('') + " " +
        df["ip.dst"].fillna('') + " " +
        df["_ws.col.protocol"].fillna('') + " " +
        df["tcp.srcport"].fillna('') + " " +
        df["tcp.dstport"].fillna('') + " " +
        df["frame.len"].astype(str)
    )

    modelList = [
        'distilbert-base-nli-stsb-mean-tokens',
        'microsoft/codebert-base',
        'bert-base-nli-mean-tokens',
        'sentence-transformers/average_word_embeddings_komninos',
        'all-mpnet-base-v2'
    ]
    
    plots_dir = "PipelineResults"
    os.makedirs(plots_dir, exist_ok=True)
    
    for mod in modelList:
        print(f"\nProcessing model: {mod}")
        
        # Clean the model name for filename
        clean_model_name = mod.replace('/', '_').replace('-', '_')

        # Initialize performance tracker
        tracker = FAISSPerformanceTracker(mod)

        # Generate embeddings
        embeddings = tracker.model.encode(df["packet_text"].tolist(), convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Creating FAISS Inner Product index for proper cosine similarity
        dimension = embeddings.shape[1]
        nlist = min(100, int(np.sqrt(len(embeddings))))
        quantizer = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)

        # Track performance for various operations (matching first code's scale)
        for num_ops in [2500, 5000, 7500, 10000, 20000, 30000]:
            print(f"  Running {num_ops} operations...")
            
            # Insertion tracking
            new_packet_texts = [f"192.168.1.{i%255} 192.168.1.{(i+1)%255} TCP {i%65535} {(i*10)%65535} {i*100}" for i in range(num_ops)]
            tracker.track_insertion(index, new_packet_texts)

            # Deletion tracking
            tracker.track_deletion(index, num_ops)

            # Update tracking
            update_texts = [f"10.0.0.{i%255} 10.0.0.{(i+1)%255} UDP {i%65535} {(i*5)%65535} {i*50}" for i in range(num_ops)]
            tracker.track_update(index, num_ops, update_texts)
            
            # Query tracking
            query_texts = [f"172.16.1.{i%255} 172.16.1.{(i+1)%255} TCP {i%65535}" for i in range(min(num_ops, 100))]
            tracker.track_query(index, query_texts, k=5)

            # Show meaningful query results at the largest batch
            if num_ops == 10000:                
                # Create meaningful queries based on actual data
                meaningful_queries = tracker.create_meaningful_queries(df)
                
                for i, query in enumerate(meaningful_queries[:3]):  # Show first 3 queries
                    try:
                        distances, indices = tracker.query_top_k(index, [query], k=5)
                        print(f"Query #{i+1}: '{query}'")
                        print(f"Top 5 Similar Flow Indices: {indices[0]}")
                        print(f"Cosine Similarities: {distances[0]}")
                    except Exception as e:
                        print(f"Query {i+1} failed: {e}")
                        print()
                # Query for TCP flows
                tcp_query = ["TCP flow analysis 192.168.1.1 80"]
                try:
                    distances, indices = tracker.query_top_k(index, tcp_query, k=5)
                    print(f"Query: 'TCP flows analysis'")
                    print(f"Top 5 Similar Flow Indices: {indices[0]}")
                    print(f"Cosine Similarities: {distances[0]}")
                except Exception as e:
                    print(f"TCP query failed: {e}")
                    print()
                
                # Query for HTTP traffic
                http_query = ["HTTP web traffic port 80 443"]
                try:
                    distances, indices = tracker.query_top_k(index, http_query, k=5)
                    print(f"Query: 'HTTP web traffic'")
                    print(f"Top 5 Similar Flow Indices: {indices[0]}")
                    print(f"Cosine Similarities: {distances[0]}")
                except Exception as e:
                    print(f"HTTP query failed: {e}")
                    print()

        plot_filename = f"{plots_dir}/{clean_model_name}.png"
        tracker.plot_performance_metrics(save_path=plot_filename)

if __name__ == "__main__":
    main()
