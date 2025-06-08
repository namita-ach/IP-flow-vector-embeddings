## Understanding the Data Source

(tried to stay true to the paper here)

The code works with network traffic data from the ISCX Tor/Non-Tor 2017 dataset, which contains information about internet traffic flows. This dataset includes two scenarios that get combined together to create a larger training set.

Each network flow record contains essential information like source IP address, source port, destination IP address, destination port, and flow packets per second. These five pieces of information form what network security experts call a "five-tuple"- the basic identifier for any network connection.

## The BERT Approach to Network Flows

### Converting Network Flows to Text

* The first major step is transforming numerical and categorical network data into text that BERT can understand. Each network flow gets converted into a sentence-like format.

* The conversion process takes each row of network data and creates a standardized text representation. For example, a flow with source IP 192.168.1.1, source port 80, destination IP 10.0.0.5, destination port 443, and 1000 packets per second becomes the text string "SRC_IP: 192.168.1.1, SRC_PORT: 80, DST_IP: 10.0.0.5, DST_PORT: 443, PACKETS/s: 1000".

* This text format serves several important purposes. It preserves all the original information while making it accessible to BERT's text processing capabilities. The structured format with clear labels helps BERT learn the relationships between different components of network flows. The consistent formatting ensures that similar flows produce similar text representations.

## Data Preparation Pipeline

### Loading and Combining Data

- The code starts by loading two separate CSV files containing network flow data from different scenarios. These files get combined into a single dataset to provide more training examples for the BERT model.

- Data cleaning involves removing extra spaces from column names and selecting only the relevant columns needed for the five-tuple representation. This focused approach helps the model concentrate on the most important features of network flows.

### Text Conversion Process

- Each network flow record gets transformed into a natural language sentence using a consistent template. The template includes clear labels for each component, making it easy for BERT to identify and learn patterns in the different parts of network flows.

- The text conversion preserves the original data relationships while making them accessible to language model processing. This bridge between network data and natural language processing is what makes the BERT approach unique.

### Dataset Preparation

- The processed text data gets converted into a format compatible with the Hugging Face datasets library. This standardization allows the code to use powerful training tools designed for transformer models like BERT.

### Masked Language Modeling

- BERT fine-tuning uses a technique called Masked Language Modeling. During training, the system randomly hides about 15% of the tokens in each sentence and asks BERT to predict what the hidden tokens should be.

- For network flows, this means BERT might see "SRC_IP: 192.168.1.1, SRC_PORT: [MASK], DST_IP: 10.0.0.5, DST_PORT: 443, PACKETS/s: 1000" and need to predict that the masked token should be "80". This forces BERT to learn the relationships between different parts of network flows.

- The masking process happens dynamically during training, meaning different tokens get masked in each training iteration. This provides extensive practice for BERT to understand all possible relationships in the network flow data.

### Training Configuration

- The training process uses carefully chosen parameters to optimize learning while preventing overfitting. A batch size of 8 samples per device provides a good balance between training stability and computational efficiency.

- The model trains for 3 epochs, meaning it sees the entire dataset 3 times. This is usually sufficient for BERT to adapt its pre-trained knowledge to the specific patterns in network flow data without losing its general language understanding capabilities.

- Weight decay adds a small penalty for large model parameters, which helps prevent the model from becoming too specialized on the training data and improves its ability to generalize to new network flows.

### Evaluation Strategy

- The training process reserves 10% of the data for evaluation, allowing continuous monitoring of how well the model learns to understand network flows. Evaluation happens after each epoch, providing insight into the training progress.

- This evaluation data never gets used for training, so it provides an unbiased measure of how well the fine-tuned model will perform on new, unseen network flows.

## Embedding Generation

### CLS Token Approach

- After fine-tuning, the model generates embeddings using the CLS token representation. The CLS token is a special token that BERT adds to the beginning of every sentence. During training, this token learns to represent the meaning of the entire sentence.

- For network flows, the CLS token embedding captures the essential characteristics of the entire flow in a single vector. This vector representation preserves the relationships that BERT learned during fine-tuning while providing a format suitable for similarity search and other machine learning tasks.

### Extraction Process

- Embedding generation involves passing network flow text through the fine-tuned BERT model and extracting the CLS token representation from the final hidden layer. This process happens without updating the model weights, ensuring consistent embeddings for identical inputs.

- The extraction process handles batches of flows efficiently, making it practical to generate embeddings for large datasets. The resulting embeddings maintain the semantic relationships learned during fine-tuning.
  
### Loss Function and Objectives

- The masked language modeling objective provides a self-supervised learning approach that does not require manual labeling of network flows. The model learns by predicting masked components based on context from other parts of the flow.

- This self-supervised approach is particularly valuable for network security applications where labeled data can be expensive or difficult to obtain. The model learns meaningful representations directly from the structure of the network flow data.
