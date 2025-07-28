import os
import torch
import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModelForTokenClassification

from llmware.parsers import ImageParser
from llmware.models import ModelCatalog
from llmware.prompts import Prompt

# -----------------------------
# Step 1: OCR using pytesseract
# -----------------------------
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)

# -----------------------------
# Step 2: Entity Extraction with ClinicalBERT
# -----------------------------
def extract_with_bert(text):
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=6)  # Adjust as needed

    label_mapping = {
        0: "name",
        1: "age",
        2: "gender",
        3: "Blood Type",
        4: "Medical Condition",
        5: "Date"
    }

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = predicted_labels[0].tolist()

    extracted_info = {}
    current_entity = ""
    current_label = None

    for token, label in zip(tokens, labels):
        if token.startswith("##"):
            current_entity += token[2:]
        else:
            if current_label:
                extracted_info.setdefault(current_label, []).append(current_entity)
            current_entity = token
            current_label = label_mapping.get(label)

    if current_label:
        extracted_info.setdefault(current_label, []).append(current_entity)

    return {k: ' '.join(v) for k, v in extracted_info.items()}

# -----------------------------
# Step 3: Structured Extraction with LLMWare
# -----------------------------
def extract_with_llmware(text):
    model = ModelCatalog().load_model("bling-tiny")  # lightweight LLM
    prompt = Prompt(model)
    response = prompt.prompt_with_context(
        context=text,
        prompt="Extract the following fields from the text: Name, Age, Gender, Date, Blood Type, Medical Condition.",
        max_output_tokens=200
    )
    return response["llm_response"]

# -----------------------------
# Step 4: Merge Outputs
# -----------------------------
def merge_extractions(bert_info, llmware_text):
    # Convert LLMWare output to dict
    structured_info = {}
    for line in llmware_text.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key and value and key not in bert_info:
                structured_info[key] = value

    return {**bert_info, **structured_info}

# -----------------------------
# Main Pipeline
# -----------------------------
def process_image(image_path):
    print("[*] Performing OCR...")
    raw_text = extract_text_from_image(image_path)

    print("[*] Extracting entities with BERT...")
    bert_info = extract_with_bert(raw_text)

    print("[*] Extracting structure with LLMWare...")
    llmware_info_text = extract_with_llmware(raw_text)

    print("[*] Merging results...")
    final_info = merge_extractions(bert_info, llmware_info_text)

    return final_info

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    image_path = "D:\\novathonnew\\records\\hello.png"
    output = process_image(image_path)

    print("\n[FINAL EXTRACTED INFO]")
    for key, value in output.items():
        print(f"{key}: {value}")
