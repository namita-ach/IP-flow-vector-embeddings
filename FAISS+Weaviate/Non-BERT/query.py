import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from gensim.models import Word2Vec
import warnings
from typing import List, Dict, Tuple
import faiss
import weaviate
import os
from tqdm import tqdm

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
            'mlp': None,
            'autoencoder': None,
            'ip2vec': None
        }
        
        # Initialize vector databases
        self.faiss_indexes = {}
        self.weaviate_clients = {}
        
        # Initialize preprocessors
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Preprocess data
        self.processed_data = self._preprocess_data()
        
        print(f"✓ Loaded {len(self.data)} network flow records")
        print(f"✓ Columns: {list(self.data.columns)}")
    
    def embed_all_data(self):
        """Generate embeddings using all three methods"""
        print("Generating embeddings for all methods...")
        self.generate_mlp_embeddings()
        self.generate_autoencoder_embeddings()
        self.generate_ip2vec_embeddings()
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
    
    def generate_mlp_embeddings(self, embedding_dim=128):
        """Generate MLP-based embeddings"""
        print("Generating MLP embeddings...")
        
        # Prepare data
        X_cat = self.processed_data[self.categorical_cols].values
        X_cont = self.processed_data[self.continuous_cols].values if self.continuous_cols else np.zeros((len(self.processed_data), 0))
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
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Generate embeddings
        embeddings = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0]
                outputs = model(inputs)
                embeddings.append(outputs.numpy())
        
        self.embeddings['mlp'] = np.concatenate(embeddings, axis=0)
        print(f"✓ MLP embeddings generated: {self.embeddings['mlp'].shape}")
        
        # Create FAISS index
        self._create_faiss_index('mlp')
        
        # Create Weaviate collection
        self._create_weaviate_collection('mlp')
    
    def generate_autoencoder_embeddings(self, embedding_dim=128):
        """Generate Autoencoder-based embeddings"""
        print("Generating Autoencoder embeddings...")
        
        # Prepare data
        X_cat = self.processed_data[self.categorical_cols].values
        X_cont = self.processed_data[self.continuous_cols].values if self.continuous_cols else np.zeros((len(self.processed_data), 0))
        X = np.concatenate([X_cat, X_cont], axis=1)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        # Define autoencoder
        input_dim = X.shape[1]
        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, embedding_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(embedding_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded
        
        model = Autoencoder()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train the autoencoder
        print("Training autoencoder...")
        for epoch in range(10):
            total_loss = 0
            for batch in loader:
                inputs = batch[0]
                optimizer.zero_grad()
                _, outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")
        
        # Generate embeddings
        embeddings = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0]
                encoded, _ = model(inputs)
                embeddings.append(encoded.numpy())
        
        self.embeddings['autoencoder'] = np.concatenate(embeddings, axis=0)
        print(f"✓ Autoencoder embeddings generated: {self.embeddings['autoencoder'].shape}")
        
        # Create FAISS index
        self._create_faiss_index('autoencoder')
        
        # Create Weaviate collection
        self._create_weaviate_collection('autoencoder')
    
    def generate_ip2vec_embeddings(self, embedding_dim=128):
        """Generate IP2Vec embeddings using Word2Vec"""
        print("Generating IP2Vec embeddings...")
        
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
        
        # Train Word2Vec model
        print("Training Word2Vec model...")
        model = Word2Vec(
            sentences,
            vector_size=embedding_dim,
            window=3,
            min_count=1,
            workers=4,
            sg=1,  # skip-gram
            epochs=10
        )
        
        # Generate embeddings for each flow
        def embed_flow(flow_sentence, w2v_model):
            vectors = [w2v_model.wv[token] for token in flow_sentence if token in w2v_model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)
        
        embeddings = np.array([embed_flow(sentence, model) for sentence in sentences])
        self.embeddings['ip2vec'] = embeddings
        print(f"✓ IP2Vec embeddings generated: {self.embeddings['ip2vec'].shape}")
        
        # Create FAISS index
        self._create_faiss_index('ip2vec')
        
        # Create Weaviate collection
        self._create_weaviate_collection('ip2vec')
    
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
        
        # Initialize Weaviate client
        client = weaviate.Client(
            url="http://localhost:8080",  # Replace with your Weaviate instance
            additional_headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")  # Optional, for generative search
            }
        )
        
        # Define collection schema
        class_obj = {
            "class": f"NetworkFlows_{method.capitalize()}",
            "vectorizer": "none",  # We'll provide our own vectors
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
                }
            ]
        }
        
        # Delete existing collection if it exists
        if client.schema.exists(f"NetworkFlows_{method.capitalize()}"):
            client.schema.delete_class(f"NetworkFlows_{method.capitalize()}")
        
        # Create new collection
        client.schema.create_class(class_obj)
        
        # Add data to Weaviate with embeddings
        with client.batch as batch:
            batch.batch_size = 100
            for idx, (_, row) in enumerate(tqdm(self.data.iterrows(), total=len(self.data), desc=f"Adding {method} to Weaviate")):
                properties = {
                    "source_ip": row['ip.src'],
                    "destination_ip": row['ip.dst'],
                    "protocol": row['_ws.col.protocol'],
                    "source_port": int(row['tcp.srcport']),
                    "destination_port": int(row['tcp.dstport']),
                    "frame_length": int(row['frame.len']),
                    "original_index": idx
                }
                
                vector = self.embeddings[method][idx].astype('float32').tolist()
                
                client.batch.add_data_object(
                    properties,
                    f"NetworkFlows_{method.capitalize()}",
                    vector=vector
                )
        
        self.weaviate_clients[method] = client
        print(f"✓ Weaviate collection created for {method} embeddings")
    
    def create_query_embedding(self, query_text: str, method: str):
        """Create embedding for a query string"""
        # Parse the query into components
        query_components = self._parse_query(query_text)
        
        # Create a temporary dataframe with the query
        query_df = pd.DataFrame([query_components])
        
        # Process the query through the same pipeline
        query_processed = query_df.copy()
        
        # Encode categorical columns using existing encoders
        for col in self.categorical_cols:
            if col in query_processed.columns:
                le = self.label_encoders[col]
                try:
                    query_processed[col] = le.transform(query_processed[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    query_processed[col] = 0
        
        # Scale continuous columns
        if self.continuous_cols:
            query_processed[self.continuous_cols] = self.scaler.transform(query_processed[self.continuous_cols])
        
        if method == 'mlp':
            X_cat = query_processed[self.categorical_cols].values
            X_cont = query_processed[self.continuous_cols].values if self.continuous_cols else np.zeros((len(query_processed), 0))
            X = np.concatenate([X_cat, X_cont], axis=1)
            X_tensor = torch.FloatTensor(X)
            
            # Use the same model structure
            input_dim = X.shape[1]
            model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            )
            
            with torch.no_grad():
                embedding = model(X_tensor).numpy()
            return embedding
        
        elif method == 'autoencoder':
            # Similar to MLP but using encoder part
            X_cat = query_processed[self.categorical_cols].values
            X_cont = query_processed[self.continuous_cols].values if self.continuous_cols else np.zeros((len(query_processed), 0))
            X = np.concatenate([X_cat, X_cont], axis=1)
            X_tensor = torch.FloatTensor(X)
            
            # Use encoder part
            encoder = nn.Sequential(
                nn.Linear(X.shape[1], 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            )
            
            with torch.no_grad():
                embedding = encoder(X_tensor).numpy()
            return embedding
        
        elif method == 'ip2vec':
            # Create sentence from query
            sentence = [
                f"src:{query_components['ip.src']}",
                f"dst:{query_components['ip.dst']}",
                f"sport:{query_components['tcp.srcport']}",
                f"dport:{query_components['tcp.dstport']}",
                f"proto:{query_components['_ws.col.protocol']}",
                f"len:{self._bucket_len(query_components['frame.len'])}"
            ]
            
            # For simplicity, return average of existing embeddings
            # In practice, you'd retrain or use the existing Word2Vec model
            return np.mean(self.embeddings['ip2vec'], axis=0).reshape(1, -1)
    
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
        import re
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
    
    def _bucket_len(self, length):
        """Bucket frame length for IP2Vec"""
        try:
            length = float(length)
            if length < 100: return "small"
            elif length < 1500: return "medium"
            else: return "large"
        except:
            return "medium"
    
    def query_similar(self, query_text: str, method: str, top_k: int = 5) -> Dict[str, List[Tuple[int, float, Dict]]]:
        """
        Find similar network flows using all available databases
        
        Args:
            query_text: Natural language query
            method: Embedding method ('mlp', 'autoencoder', 'ip2vec')
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
            query_vector = query_embedding.astype('float32').tolist()
            
            weaviate_results = self.weaviate_clients[method].query\
                .get(f"NetworkFlows_{method.capitalize()}", [
                    "source_ip",
                    "destination_ip",
                    "protocol",
                    "source_port",
                    "destination_port",
                    "frame_length",
                    "original_index"
                ])\
                .with_near_vector({
                    "vector": query_vector,
                    "certainty": 0.7  # Adjust as needed
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
                        'frame.len': item['frame_length']
                    }
                    results['weaviate'].append((idx, float(score), flow_data))
        
        return results
    
    def interactive_query(self):
        """Interactive query interface"""
        print("\n" + "="*60)
        print("🔍 NETWORK FLOW EMBEDDING QUERY SYSTEM")
        print("="*60)
        print("Available methods: mlp, autoencoder, ip2vec")
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
                
                method = input("Choose method (mlp/autoencoder/ip2vec): ").strip().lower()
                if method not in ['mlp', 'autoencoder', 'ip2vec']:
                    print("Invalid method. Please choose: mlp, autoencoder, or ip2vec")
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
                        print("-" * 40)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

def main():
    """Main function to run the query system"""
    print("Initializing Network Flow Embedding Query System...")
    
    # Initialize the system (with sample data)
    system = EmbeddingQuerySystem()
    
    # Generate all embeddings
    system.embed_all_data()
    
    # Start interactive query
    system.interactive_query()

if __name__ == "__main__":
    main()