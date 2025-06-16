models = [
    'distilbert-base-nli-stsb-mean-tokens',
    'microsoft/codebert-base',
    'bert-base-nli-mean-tokens',
    'sentence-transformers/average_word_embeddings_komninos',
    'all-mpnet-base-v2'
]

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sentence_transformers import SentenceTransformer
import warnings
from typing import List, Dict, Tuple
import faiss
import weaviate
import os
from tqdm import tqdm
import re

warnings.filterwarnings('ignore')

class EmbeddingQuerySystem:
    def __init__(self, data_path=None, data_df=None):
        """
        Initialize the query system with network flow data
        
        Args:
            data_path: Path to CSV file containing network flow data
            data_df: DataFrame containing network flow data (alternative to data_path)
        """
        # Load data
        if data_df is not None:
            self.data = data_df.copy()
        elif data_path is not None:
            self.data = pd.read_csv(data_path)
        else:
            # Create sample data if none provided
            self.data = self._create_sample_data()
        
        # Define columns based on your original code structure
        self.categorical_cols = ['ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport', '_ws.col.protocol']
        self.continuous_cols = ['frame.len']
        
        # Ensure required columns exist
        self._validate_columns()
        
        # Initialize embeddings storage
        self.embeddings = {
            'distilbert': None,
            'codebert': None,
            'bert-nli': None,
            'komninos': None,
            'mpnet': None
        }
        
        # Initialize vector databases
        self.faiss_indexes = {}
        self.weaviate_clients = {}
        
        # Initialize preprocessors
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Preprocess data
        self.processed_data = self._preprocess_data()
        
        # Initialize BERT models
        self.bert_models = {
            'distilbert': SentenceTransformer('distilbert-base-nli-stsb-mean-tokens'),
            'codebert': SentenceTransformer('microsoft/codebert-base'),
            'bert-nli': SentenceTransformer('bert-base-nli-mean-tokens'),
            'komninos': SentenceTransformer('sentence-transformers/average_word_embeddings_komninos'),
            'mpnet': SentenceTransformer('all-mpnet-base-v2')
        }
        
        print(f"✓ Loaded {len(self.data)} network flow records")
        print(f"✓ Columns: {list(self.data.columns)}")
    
    def embed_all_data(self):
        """Generate embeddings using all BERT-based methods"""
        print("Generating embeddings for all BERT models...")
        self.generate_bert_embeddings('distilbert')
        self.generate_bert_embeddings('codebert')
        self.generate_bert_embeddings('bert-nli')
        self.generate_bert_embeddings('komninos')
        self.generate_bert_embeddings('mpnet')
        print("✓ All embeddings generated successfully!")
    
    def _create_sample_data(self):
        """Create sample network flow data for testing"""
        np.random.seed(42)
        n_samples = 1000
        
        protocols = ['TCP', 'UDP', 'HTTP', 'HTTPS', 'DNS', 'FTP', 'SSH']
        
        data = []
        for i in range(n_samples):
            data.append({
                'ip.src': f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                'ip.dst': f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                'tcp.srcport': np.random.randint(1024, 65535),
                'tcp.dstport': np.random.choice([80, 443, 22, 21, 53, 25, np.random.randint(1024, 65535)]),
                '_ws.col.protocol': np.random.choice(protocols),
                'frame.len': np.random.randint(64, 1500)
            })
        
        return pd.DataFrame(data)
    
    def _validate_columns(self):
        """Ensure all required columns exist in the data"""
        required_cols = self.categorical_cols + self.continuous_cols
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}. Using sample data structure.")
            self.data = self._create_sample_data()
    
    def _preprocess_data(self):
        """Preprocess the data for embedding generation"""
        processed_data = self.data.copy()
        
        # Encode categorical columns
        for col in self.categorical_cols:
            if col in processed_data.columns:
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale continuous columns
        if self.continuous_cols:
            continuous_data = processed_data[self.continuous_cols]
            processed_data[self.continuous_cols] = self.scaler.fit_transform(continuous_data)
        
        return processed_data
    
    def generate_bert_embeddings(self, model_name: str):
        """Generate embeddings using the specified BERT model"""
        print(f"Generating {model_name} embeddings...")
        
        # Create text representations of network flows
        texts = []
        for _, row in self.data.iterrows():
            text = (f"Source IP: {row['ip.src']}, Destination IP: {row['ip.dst']}, "
                   f"Source Port: {row['tcp.srcport']}, Destination Port: {row['tcp.dstport']}, "
                   f"Protocol: {row['_ws.col.protocol']}, Frame Length: {row['frame.len']}")
            texts.append(text)
        
        # Generate embeddings
        model = self.bert_models[model_name]
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        
        self.embeddings[model_name] = embeddings
        print(f"✓ {model_name} embeddings generated: {self.embeddings[model_name].shape}")
        
        # Create FAISS index
        self._create_faiss_index(model_name)
        
        # Create Weaviate collection
        self._create_weaviate_collection(model_name)
    
    def _create_faiss_index(self, method: str):
        """Create FAISS index for the specified embedding method"""
        if method not in self.embeddings or self.embeddings[method] is None:
            raise ValueError(f"No embeddings found for method {method}")
        
        embeddings = self.embeddings[method].astype('float32')
        dimension = embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create and train the index
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        self.faiss_indexes[method] = index
        print(f"✓ FAISS index created for {method} embeddings")
    
    def _create_weaviate_collection(self, method: str):
        """Create Weaviate collection for the specified embedding method"""
        if method not in self.embeddings or self.embeddings[method] is None:
            raise ValueError(f"No embeddings found for method {method}")
        
        # Clean the method name for Weaviate class naming
        clean_method_name = method.replace('-', '_').capitalize()
        class_name = f"NetworkFlows_{clean_method_name}"
        
        # Initialize Weaviate client
        client = weaviate.Client(
            url="http://localhost:8080",
            additional_headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
            }
        )
        
        # Define collection schema
        class_obj = {
            "class": class_name,
            "vectorizer": "none",
            "properties": [
                {
                    "name": "source_ip",
                    "dataType": ["text"]
                },
                {
                    "name": "destination_ip",
                    "dataType": ["text"]
                },
                {
                    "name": "protocol",
                    "dataType": ["text"]
                },
                {
                    "name": "source_port",
                    "dataType": ["int"]
                },
                {
                    "name": "destination_port",
                    "dataType": ["int"]
                },
                {
                    "name": "frame_length",
                    "dataType": ["int"]
                },
                {
                    "name": "original_index",
                    "dataType": ["int"]
                },
                {
                    "name": "text_description",
                    "dataType": ["text"]
                }
            ]
        }
        
        # Delete existing collection if it exists
        if client.schema.exists(class_name):
            client.schema.delete_class(class_name)
        
        # Create new collection
        client.schema.create_class(class_obj)
        
        # Add data to Weaviate with embeddings
        with client.batch as batch:
            batch.batch_size = 100
            for idx, (_, row) in enumerate(tqdm(self.data.iterrows(), total=len(self.data), desc=f"Adding {method} to Weaviate")):
                text_desc = (f"Source IP: {row['ip.src']}, Destination IP: {row['ip.dst']}, "
                            f"Source Port: {row['tcp.srcport']}, Destination Port: {row['tcp.dstport']}, "
                            f"Protocol: {row['_ws.col.protocol']}, Frame Length: {row['frame.len']}")
                
                properties = {
                    "source_ip": row['ip.src'],
                    "destination_ip": row['ip.dst'],
                    "protocol": row['_ws.col.protocol'],
                    "source_port": int(row['tcp.srcport']),
                    "destination_port": int(row['tcp.dstport']),
                    "frame_length": int(row['frame.len']),
                    "original_index": idx,
                    "text_description": text_desc
                }
                
                vector = self.embeddings[method][idx].astype('float32').tolist()
                
                client.batch.add_data_object(
                    properties,
                    class_name,
                    vector=vector
                )
        
        self.weaviate_clients[method] = client
        print(f"✓ Weaviate collection created for {method} embeddings")
    
    def create_query_embedding(self, query_text: str, method: str):
        """Create embedding for a query string using the specified BERT model"""
        if method not in self.bert_models:
            raise ValueError(f"Unknown BERT model: {method}")
        
        # Generate embedding using the appropriate model
        model = self.bert_models[method]
        embedding = model.encode([query_text], show_progress_bar=False)
        return embedding
    
    def _parse_query(self, query_text: str) -> Dict:
        """Parse a natural language query into network flow components"""
        query_lower = query_text.lower()
        
        # Default values
        query_components = {
            'ip.src': '192.168.1.1',
            'ip.dst': '10.0.0.1',
            'tcp.srcport': 80,
            'tcp.dstport': 443,
            '_ws.col.protocol': 'TCP',
            'frame.len': 1500
        }
        
        # Extract IP addresses
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = re.findall(ip_pattern, query_text)
        if len(ips) >= 1:
            query_components['ip.src'] = ips[0]
        if len(ips) >= 2:
            query_components['ip.dst'] = ips[1]
        
        # Extract ports
        port_pattern = r'\b(?:port|:)\s*(\d+)\b'
        ports = re.findall(port_pattern, query_text)
        if ports:
            query_components['tcp.dstport'] = int(ports[0])
        
        # Extract protocols
        protocols = ['TCP', 'UDP', 'HTTP', 'HTTPS', 'DNS', 'FTP', 'SSH']
        for protocol in protocols:
            if protocol.lower() in query_lower:
                query_components['_ws.col.protocol'] = protocol
                break
        
        # Extract frame length hints
        if 'large' in query_lower:
            query_components['frame.len'] = 1400
        elif 'small' in query_lower:
            query_components['frame.len'] = 64
        
        return query_components
    
    def query_similar(self, query_text: str, method: str, top_k: int = 5) -> Dict[str, List[Tuple[int, float, Dict]]]:
        """
        Find similar network flows using all available databases
        
        Args:
            query_text: Natural language query
            method: Embedding method ('distilbert', 'codebert', 'bert-nli', 'komninos', 'mpnet')
            top_k: Number of similar flows to return
            
        Returns:
            Dictionary with results from each database:
            {
                'faiss': [(index, score, flow_data), ...],
                'weaviate': [(index, score, flow_data), ...]
            }
        """
        if method not in self.embeddings or self.embeddings[method] is None:
            raise ValueError(f"Embeddings for method '{method}' not generated. Call embed_all_data() first.")
        
        # Generate query embedding
        query_embedding = self.create_query_embedding(query_text, method)
        
        results = {
            'faiss': [],
            'weaviate': []
        }
        
        # FAISS search
        if method in self.faiss_indexes:
            query_vector = query_embedding.astype('float32')
            faiss.normalize_L2(query_vector)
            
            D, I = self.faiss_indexes[method].search(query_vector, top_k)
            
            for score, idx in zip(D[0], I[0]):
                if idx >= 0:  # -1 indicates no result
                    flow_data = self.data.iloc[idx].to_dict()
                    results['faiss'].append((idx, float(score), flow_data))
        
        # Weaviate search
        if method in self.weaviate_clients:
            clean_method_name = method.replace('-', '_').capitalize()
            class_name = f"NetworkFlows_{clean_method_name}"
            
            query_vector = query_embedding.astype('float32').tolist()
            
            weaviate_results = self.weaviate_clients[method].query\
                .get(class_name, [
                    "source_ip",
                    "destination_ip",
                    "protocol",
                    "source_port",
                    "destination_port",
                    "frame_length",
                    "original_index",
                    "text_description"
                ])\
                .with_near_vector({
                    "vector": query_vector,
                    "certainty": 0.7
                })\
                .with_limit(top_k)\
                .do()
            
            if 'data' in weaviate_results and 'Get' in weaviate_results['data']:
                for item in weaviate_results['data']['Get'][f"NetworkFlows_{method.capitalize()}"]:
                    idx = item['original_index']
                    score = item['_additional']['certainty']  # Weaviate's certainty score
                    flow_data = {
                        'ip.src': item['source_ip'],
                        'ip.dst': item['destination_ip'],
                        '_ws.col.protocol': item['protocol'],
                        'tcp.srcport': item['source_port'],
                        'tcp.dstport': item['destination_port'],
                        'frame.len': item['frame_length'],
                        'text_description': item['text_description']
                    }
                    results['weaviate'].append((idx, float(score), flow_data))
        
        return results
    
    def interactive_query(self):
        """Interactive query interface"""
        print("\n" + "="*60)
        print("🔍 NETWORK FLOW EMBEDDING QUERY SYSTEM (BERT-based)")
        print("="*60)
        print("Available methods: distilbert, codebert, bert-nli, komninos, mpnet")
        print("Example queries:")
        print("  - 'TCP traffic from 192.168.1.1'")
        print("  - 'HTTP traffic to port 80'")
        print("  - 'DNS traffic'")
        print("  - 'Large packets'")
        print("  - '10.0.0.1 192.168.1.1 HTTPS'")
        print("\nType 'quit' to exit")
        print("-" * 60)
        
        while True:
            try:
                # Get user input
                query = input("\nEnter your query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                method = input("Choose method (distilbert/codebert/bert-nli/komninos/mpnet): ").strip().lower()
                if method not in ['distilbert', 'codebert', 'bert-nli', 'komninos', 'mpnet']:
                    print("Invalid method. Please choose from: distilbert, codebert, bert-nli, komninos, mpnet")
                    continue
                
                # Perform query
                print(f"\n🔍 Searching for: '{query}' using {method.upper()} embeddings...")
                results = self.query_similar(query, method, top_k=5)
                
                # Display results
                for db_name, db_results in results.items():
                    print(f"\n📊 Top 5 Similar Network Flows from {db_name.upper()}:")
                    print("-" * 80)
                    
                    for i, (idx, score, flow_data) in enumerate(db_results, 1):
                        print(f"\n{i}. Similarity Score: {score:.4f}")
                        print(f"   Index: {idx}")
                        print(f"   Source IP: {flow_data['ip.src']}")
                        print(f"   Destination IP: {flow_data['ip.dst']}")
                        print(f"   Protocol: {flow_data['_ws.col.protocol']}")
                        print(f"   Source Port: {flow_data['tcp.srcport']}")
                        print(f"   Destination Port: {flow_data['tcp.dstport']}")
                        print(f"   Frame Length: {flow_data['frame.len']}")
                        if 'text_description' in flow_data:
                            print(f"   Description: {flow_data['text_description']}")
                        print("-" * 40)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

def main():
    """Main function to run the query system"""
    print("Initializing Network Flow Embedding Query System (BERT-based)...")
    
    # Initialize the system (with sample data)
    system = EmbeddingQuerySystem()
    
    # Generate all embeddings
    system.embed_all_data()
    
    # Start interactive query
    system.interactive_query()

if __name__ == "__main__":
    main()