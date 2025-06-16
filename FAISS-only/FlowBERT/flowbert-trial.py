import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from datasets import DatasetDict



data1 = pd.read_csv("/kaggle/input/iscx-tor-nontor-2017-csvs/Scenario-A-merged_5s.csv")
data1.head()

data1.shape

data2 = pd.read_csv("/kaggle/input/iscx-tor-nontor-2017-csvs/Scenario-B-merged_5s.csv")

combined = pd.concat([data1, data2])
print(combined.shape)

combined.head()

combined.columns = combined.columns.str.strip(' ')
print(combined.columns.tolist())

data = combined[['Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Flow Packets/s']]

print(data.head())

print(data.shape)

data.to_csv('fivetupleflow.csv')

# finetuning

#convert the flows to text- maybe one sentence for one row?

# first we find a way to process the csv to merge the rows together
def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df["text"] = df.apply(lambda row: f"SRC_IP: {row['Source IP']}, SRC_PORT: {row['Source Port']}, "
                                    f"DST_IP: {row['Destination IP']}, DST_PORT: {row['Destination Port']}, "
                                    f"PACKETS/s: {row['Flow Packets/s']}", axis=1)
    return df[["text"]]

csv= '/kaggle/working/fivetupleflow.csv'
df = preprocess_data(csv)
df.head()
# so now we have all the columns merged into one for each row
# each row is like one sentence

# okay so next we'd have to convert this to a hugginface dataset so that we can finetune a basic bert model
def convertDataset(df):
    return Dataset.from_pandas(df)

dataset = convertDataset(df)

# next we will have to tokenise
# we'll be initializing the tokenizer for the bert model using the hugging face transformers library
# we're basically loading the pretrained tokenizer that's associated with the model that we're trying to finetune
# this tokenizer uses the wordpiece algorithm to split text into smalelr units (tokens)
# it maps the tokens to a unique integer IS based off the model's vocab

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

def finetune_bert(dataset):
    tokenized_dataset = dataset.map(tokenize, batched=True)  # this applies the tokenization we just discussed

    # gotta split the dataset with a 10% eval so that the embeddings are alright
    dataset_split = tokenized_dataset.train_test_split(test_size=0.1)  # 90% train, 10% eval

    model = BertForMaskedLM.from_pretrained("bert-base-uncased")  # here we load the generic bert model SPECIFICALLY for masked language modelling

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    # this just prepares batches of data with dynamic masking
    # it randomly masks 15% of the tokens per batch and also uses tokenizer to handle the padding and masking

    training_args = TrainingArguments(
        output_dir="./bert_finetuned",  # this is where the training outputs will go
        evaluation_strategy="epoch",  # we're evaluating the model after every training epoch
        save_strategy="epoch",  # save model ka checkpoint after every epoch
        per_device_train_batch_size=8,  # 8 samples per core during training
        per_device_eval_batch_size=8,  # 8 samples per core during evaluation
        num_train_epochs=3,  # trains the model for this many passes through the dataset
        weight_decay=0.01,  # l2 regularization to prevent overfitting
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],  # Added this!
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()  # Start training

    model.save_pretrained("./bert_finetuned")
    tokenizer.save_pretrained("./bert_finetuned")

finetune_bert(dataset)

# now we just test the generated metrics
def generate_embeddings(texts, model_path="./bert_finetuned"):
    model = BertForMaskedLM.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model.bert(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # CLS token embedding
    return embeddings

# embeddings = generate_embeddings(["SRC_IP: 10.0.2.15, SRC_PORT: 53913, DST_IP: 216.58.208.46, DST_PORT: 80, PACKETS/s: 4597.701149"])

