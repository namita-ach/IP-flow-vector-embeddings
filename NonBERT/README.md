## Method 1: Multi-Layer Perceptron (MLP)
### The Concept

A Multi-Layer Perceptron is like a simplified version of a neural network brain. Think of it as a series of mathematical filters that take in raw data and gradually transform it into a more useful representation.

The MLP works in layers:
- Input layer: receives the processed flow data
- Hidden layer: applies mathematical transformations to find patterns
- Output layer: produces the final embedding vector

### How It Works With Network Flows

First, the code prepares the network flow data. Categorical information like IP addresses and protocols get converted to numbers using label encoding. This means each unique IP address or protocol gets assigned a specific number. Continuous data like port numbers and packet lengths get normalized so they all exist on similar scales.

The MLP then takes this numerical representation and passes it through two layers of processing. The first layer has 256 neurons that apply linear transformations followed by ReLU activation functions. ReLU simply means if a value is negative, make it zero, otherwise keep it as is. This helps the network learn non-linear patterns.

The second layer reduces the 256 values down to 128 values, which becomes the final embedding. These 128 numbers capture the essential characteristics of each network flow in a format that can be easily compared and searched.

### Why This Approach Works

MLPs are good at finding complex relationships between different features of the data. For network flows, this means it can learn that certain combinations of IP addresses, ports, and protocols tend to go together. The embedding preserves these relationships so similar flows end up close to each other in the vector space.

## Method 2: Autoencoder

### The Concept

An autoencoder is like a compression and decompression system. Imagine trying to compress a photograph into a smaller file size and then expanding it back. The autoencoder learns to compress network flow data into a smaller representation while trying to preserve all the important information.

The autoencoder has two parts:
- Encoder: compresses the input data into a smaller representation
- Decoder: tries to reconstruct the original data from the compressed version

### How It Works With Network Flows

The autoencoder starts with the same data preprocessing as the MLP method. Network flows get converted to numerical vectors through label encoding and normalization.

The encoder part takes the full network flow representation and compresses it down to 128 dimensions through a hidden layer of 256 neurons. This 128-dimensional representation is the embedding we want.

The decoder part tries to reconstruct the original network flow data from just these 128 numbers. It expands the 128 dimensions back up through another hidden layer to match the original input size.

During training, the autoencoder learns by comparing its reconstructed output to the original input. The goal is to minimize the difference between them. This forces the 128-dimensional embedding to capture all the essential information needed to recreate the original flow.

### Why This Approach Works

The compression requirement forces the autoencoder to learn the most important features of network flows. If two flows are similar, their embeddings will be similar because they compress to similar representations. The reconstruction requirement ensures that important distinguishing features are preserved in the embedding.

This method is particularly good at capturing the underlying structure of the data because it must learn to represent each flow efficiently while maintaining the ability to reconstruct it accurately.

## Method 3: IP2Vec (Word2Vec Adaptation)

### The Concept

IP2Vec adapts the Word2Vec algorithm, which was originally designed for understanding relationships between words in text. Word2Vec learns that words appearing in similar contexts tend to have similar meanings. IP2Vec applies this same principle to network flows by treating flow components as "words" and flows as "sentences."

### How It Works With Network Flows

Instead of preprocessing numerical data like the other methods, IP2Vec takes a completely different approach. It converts each network flow into a sentence-like structure made up of tokens.

For each flow, it creates tokens like:
- src:192.168.1.1 (source IP)
- dst:10.0.0.5 (destination IP) 
- sport:80 (source port)
- dport:443 (destination port)
- proto:TCP (protocol)
- len:medium (packet length category)

Notice that packet lengths get converted to categories like small, medium, or large rather than exact numbers. This helps group similar flows together.

Each flow becomes a sentence of these tokens. The Word2Vec algorithm then learns relationships between these tokens by analyzing which tokens tend to appear together in the same flows.

The algorithm uses a technique called skip-gram, which tries to predict surrounding tokens given a center token. For example, if it sees "proto:TCP", it learns to predict that tokens like "sport:80" or "dport:443" might appear in the same flow.

Once trained, each token has its own vector representation. To create an embedding for a complete flow, the method takes all the token vectors for that flow and averages them together.

### Why This Approach Works

This method captures the contextual relationships between different components of network flows. It learns that certain protocols tend to use certain ports, that certain IP ranges communicate with each other, and that certain combinations of features tend to occur together.

The advantage is that it can capture semantic relationships that might not be obvious from the raw numerical data. For example, it might learn that flows using ports 80 and 443 are similar because they both represent web traffic, even though the port numbers themselves are quite different.

## Performance Tracking and Testing

The code includes a comprehensive performance tracking system that measures how well each embedding method works with the FAISS vector database. FAISS is a system designed for efficient similarity search in large collections of vectors.

### What Gets Measured

The performance tracker measures four types of operations:

- Insertion: Adding new flow embeddings to the database
- Deletion: Removing existing embeddings from the database  
- Update: Changing existing embeddings (implemented as delete then insert)
- Query: Searching for similar flows given a query embedding

For each operation, the system tracks execution time, CPU usage, and memory consumption at different scales from 2,500 to 30,000 operations.

### Query Generation and Testing

The system tests realistic queries by generating query embeddings based on actual patterns in the data:

- Protocol queries look for flows using specific protocols like TCP
- IP queries look for flows involving specific IP address patterns
- Port queries look for flows using specific port numbers

Each query embedding gets created using the same method as the database embeddings, ensuring consistency. The system then searches for the top 5 most similar flows and measures the cosine similarity scores.

### FAISS Index Configuration

The code uses FAISS with an Inverted File index structure combined with flat vectors for exact similarity computation. This provides a good balance between search speed and accuracy. The index gets configured to use inner product similarity, which works well with normalized vectors to compute cosine similarity.

## Data Processing Pipeline

The entire pipeline follows a consistent structure regardless of which embedding method is used:

- Data loading reads network flow records from a CSV file and handles various data types appropriately. Missing values get filled with sensible defaults, and data types get converted as needed.

- Preprocessing converts categorical data to numerical representations and normalizes continuous values. This ensures all embedding methods receive data in a consistent format.

- Embedding generation creates vector representations using one of the three methods described above. All embeddings get normalized to unit length to enable cosine similarity comparisons.

- Index construction builds the FAISS search structure and trains it on the embedding data. The index gets configured for the specific characteristics of the embedding vectors.

- Performance evaluation runs comprehensive tests measuring the efficiency of different operations at various scales. Results get visualized in graphs showing how performance changes with the number of operations.

## Comparison of Methods

- Each embedding method has different strengths and characteristics:

- The MLP method is straightforward and fast. It learns direct mappings from input features to embeddings without requiring iterative training. However, it might miss complex relationships between features since it does not have a specific objective related to the data structure.

- The Autoencoder method ensures that embeddings preserve important information through its reconstruction requirement. This often leads to higher quality embeddings that maintain the distinguishing characteristics of different flows. The training process takes longer but typically produces better results for similarity search tasks.

- The IP2Vec method captures semantic relationships between flow components that the other methods might miss. It can identify conceptual similarities between flows that use different but related protocols or ports. However, it requires careful design of the tokenization scheme and may not work as well with purely numerical relationships.

- The choice between methods depends on the specific requirements of the application, the characteristics of the network data, and the trade-offs between embedding quality and computational efficiency.
