import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split

# Load your CSV dataset
df = pd.read_csv('D:\\novathonnew\\records\\healthcare_dataset\\healthcare_dataset.csv')  # Replace with the path to your CSV file
df = df.head(1000)  # Limiting data for testing (you can remove this for full data)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Label mapping: Convert string labels to numeric labels
label_map = {
    "O": 0,           # Outside any entity
    "B-PERSON": 1,    # Beginning of a PERSON entity
    "B-AGE": 2,       # Beginning of an AGE entity
}

# Preprocess data: create 'Text' from the relevant columns and assign labels
def preprocess_data(df):
    texts = []
    labels = []

    for _, row in df.iterrows():
        # Create the 'text' column by combining multiple fields from the dataset
        text = f"Name: {row['Name']}, Age: {row['Age']}, Gender: {row['Gender']}, Medical Condition: {row['Medical Condition']}, " \
               f"Blood Type: {row['Blood Type']}, Doctor: {row['Doctor']}, Hospital: {row['Hospital']}, " \
               f"Admission Type: {row['Admission Type']}, Billing Amount: {row['Billing Amount']}, " \
               f"Medication: {row['Medication']}, Test Results: {row['Test Results']}"
        
        # Labeling is an example - you should define how to label the entities (e.g., medical conditions, names)
        label = ['O' for _ in range(len(text.split()))]  # Initialize all labels as 'O' (outside entity)

        # Example label assignment (adjust based on your data's needs)
        if 'John' in row['Name']:  # Example condition to identify a name
            label[0] = 'B-PERSON'  # Assign 'B-PERSON' to the first word if 'John' is in Name
        if row['Age'] == 45:  # Example condition to identify age
            label[2] = 'B-AGE'  # Assign 'B-AGE' for the age part
        
        # Convert labels to numeric
        label_ids = [label_map[l] for l in label]  # Convert string labels to numeric

        # Add the processed text and labels
        texts.append(text)
        labels.append(label_ids)

    return texts, labels

# Process the data
texts, labels = preprocess_data(df)

# Tokenize the data and align labels
def tokenize_data(texts, labels):
    tokenized_inputs = []
    tokenized_labels = []

    for text, label in zip(texts, labels):
        encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt", return_attention_mask=True)
        
        # Initialize label_ids with -100 for padding tokens
        label_ids = [-100] * len(encoding['input_ids'][0])

        # Align the labels with tokens in the text
        for idx, token in enumerate(text.split()):
            label_ids[idx] = label[idx]  # Assign numeric label to each token

        tokenized_inputs.append(encoding)
        tokenized_labels.append(label_ids)

    return tokenized_inputs, tokenized_labels

# Tokenize the data
tokenized_inputs, tokenized_labels = tokenize_data(texts, labels)

# Convert to torch tensors
train_inputs = torch.cat([item['input_ids'] for item in tokenized_inputs])
train_labels = torch.tensor(tokenized_labels)
attention_masks = torch.cat([item['attention_mask'] for item in tokenized_inputs])

# Split the data into train and test sets
train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks = train_test_split(train_inputs, train_labels, attention_masks, test_size=0.2)

# Create DataLoader for training
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=8)

# Create DataLoader for testing
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=8)

# Define the BERT model for token classification
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(5):  # Adjust epochs here
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch_inputs, batch_masks, batch_labels = batch
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        batch_masks = batch_masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_inputs, attention_mask=batch_masks, labels=batch_labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} Loss: {total_loss}")

# Save the trained model
model.save_pretrained('trained_bert_model')
tokenizer.save_pretrained('trained_bert_model')

# Evaluation loop (on test set)
model.eval()
total_correct = 0
total_tokens = 0

with torch.no_grad():
    for batch in test_dataloader:
        batch_inputs, batch_masks, batch_labels = batch
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        batch_masks = batch_masks.to(device)

        # Forward pass (get predictions)
        outputs = model(batch_inputs, attention_mask=batch_masks)
        predictions = torch.argmax(outputs.logits, dim=-1)

        # Compare predictions with actual labels (ignoring padding tokens)
        for pred, label in zip(predictions, batch_labels):
            mask = label != -100  # Mask to ignore padding tokens
            total_correct += (pred[mask] == label[mask]).sum().item()
            total_tokens += mask.sum().item()

accuracy = total_correct / total_tokens
print(f'Accuracy on the test set: {accuracy}')
