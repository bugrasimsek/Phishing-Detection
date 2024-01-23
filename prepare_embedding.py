import sys
import torch
import numpy as np
import trafilatura
import pickle
from sentence_transformers import SentenceTransformer
from transformers import ElectraModel, ElectraTokenizer
from googletrans import Translator
import os
import time

def extract_trafilatura(sourceDirectory):
    # Create target directory with "_Extracted" suffix if it doesn't exist
    targetDirectory = sourceDirectory.strip() + "_Extracted"
    os.makedirs(targetDirectory, exist_ok=True)

    filenames = sorted(os.listdir(sourceDirectory))  # Ensure a consistent order

    for filename in filenames:
        # file_path is like "./Legitimate/" (directory) + "legit_0.txt" (filename)
        sourceFilePath = sourceDirectory + "/" + filename
        targetFilePath = targetDirectory + "/" + filename

        # Open the file
        with open(sourceFilePath, 'r', encoding='utf-8', errors="replace") as file:
            # Read the file content as string
            html_content = file.read()

            # Use trafilatura to extract text from HTML
            text = trafilatura.extract(html_content, encoding_errors="replace", engine='utf-8')

            # If text is not empty, write it to a new file in the target directory
            if text:
                with open(targetFilePath, "w", encoding='utf-8') as f:
                    f.write(text)


# splits into 4500 char chunks and translates each chunk with for loop
def translate(text):
    translator = Translator()

    # Google Translate API has a limit of 5000 characters per request
    # So, if the text is longer than 4500 characters, split it into chunks of 4500 characters
    if len(text) > 4500:
        translation = ""

        for i in range(0, len(text), 4500):
            time.sleep(0.15)
            translation += translator.translate(text[i:i + 4000], dest='en').text

    else:
        time.sleep(0.1)
        translation = translator.translate(text, dest='en').text

    return translation


# batch translate
def batch_translate(sourceDirectory):
    # Create target directory with "_Translated" suffix if it doesn't exist
    targetDirectory = sourceDirectory.strip() + "_Translated"
    os.makedirs(targetDirectory, exist_ok=True)

    filenames = sorted(os.listdir(sourceDirectory))  # Ensure a consistent order
    size = len(filenames)

    for i in range(size):
        filename = filenames[i]
        sourceFilePath = sourceDirectory + "/" + filename
        targetFilePath = targetDirectory + "/" + filename

        with open(sourceFilePath, 'r', encoding='utf-8', errors="replace") as sourceFile:
            text = sourceFile.read()
            translation = translate(text)

            with open(targetFilePath, "w", encoding='utf-8') as targetFile:
                targetFile.write(translation)


def encodeTextElectra(text, model, tokenizer, device):
    # use tokenizer to convert text to tokens
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    # move inputs to device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()


# Function to process each folder and get embeddings with labels
def createEmbeddings(sourceDirectory, label, device, transformerType):
    if transformerType == "xlm-roberta":
        targetDirectory = sourceDirectory + "_Extracted"  # If xlm is selected, use extracted text directly
        model = SentenceTransformer('xlm-roberta-base').to(device)


    elif transformerType == "sbert":
        targetDirectory = sourceDirectory + "_Extracted_Translated"  # If sbert or electra is selected, use translated text
        model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens').to(device)

    elif transformerType == "electra":
        targetDirectory = sourceDirectory + "_Extracted_Translated"  # If sbert or electra is selected, use translated text
        model = ElectraModel.from_pretrained('google/electra-small-discriminator').to(device)
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')


    embeddings = []
    labels = []

    filenames = sorted(os.listdir(sourceDirectory))  # Ensure a consistent order

    for filename in filenames:
        targetFilePath = targetDirectory + "/" + filename

        with open(targetFilePath, 'r', encoding='utf-8', errors="replace") as file:
            text = file.read()
            if text:
                if transformerType == "electra":
                    embedding = encodeTextElectra(text, model, tokenizer, device)
                    embeddings.append(embedding)

                else:
                    curEmbedding = model.encode(text, convert_to_tensor=True, device=device).cpu().numpy()
                    embeddings.append(curEmbedding)

                labels.append(label)

    return embeddings, labels


# Check if GPU is available, use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Get transformer argument: sbert, xlm-roberta or electra
transformerType = sys.argv[1]

# Extract and Translate samples
extract_trafilatura("./Legitimate")
extract_trafilatura("./Phishing")
batch_translate("./Legitimate_Extracted")
batch_translate("./Phishing_Extracted")

# Create embeddings and labels
legitEmbeddings, legitLabels = createEmbeddings("./Legitimate", 0, device, transformerType)
phishEmbeddings, phishLabels = createEmbeddings("./Phishing", 1, device, transformerType)

# Combine embeddings and labels
embeddings = legitEmbeddings + phishEmbeddings
labels = legitLabels + phishLabels

# Save embeddings and labels to a pickle file
with open('embeddings-' + transformerType + '.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'labels': labels}, f)
