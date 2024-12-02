import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pytesseract
from PIL import Image

# Load the pre-trained ClinicalBERT model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=5)  # Adjust labels if model is fine-tuned

# Function to extract text from an image using Tesseract
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Function to extract information from text using the fine-tuned model
def extract_information(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=-1)

    # Align tokens and labels
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = predicted_labels[0].tolist()

    extracted_info = {}
    current_entity = ""
    current_label = None

    for token, label in zip(tokens, labels):
        if token.startswith("##"):  # Handle subwords
            current_entity += token[2:]
        else:
            if current_label is not None:
                extracted_info[current_label] = current_entity
            current_entity = token
            current_label = label_mapping.get(label, None)

    # Add the final entity
    if current_label is not None:
        extracted_info[current_label] = current_entity

    return extracted_info

# Define label mapping
label_mapping = {
    0: "name",
    1: "age",
    2: "gender",
    3: "Blood Type",
    4: "Medical Condition",
    5: "Date of Admission",
}

# Example usage
image_path = "D:\\novathonnew\\records\\hello.png"
text = extract_text_from_image(image_path)
extracted_info = extract_information(text)

print(extracted_info)
