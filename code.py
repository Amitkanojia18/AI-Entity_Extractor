import pandas as pd
import os
import requests
from PIL import Image
import pytesseract
import re
from google.colab import drive

# Importing drive
drive.mount('/content/drive')

# Paths
image_folder = '/content/drive/MyDrive/dwnld_images/'
train_data_path = '/content/drive/MyDrive/train_data_path/train.csv'
output_csv_path = '/content/drive/MyDrive/extracted_text.csv'

# Create directory if not exists
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    print(f"Directory {image_folder} created.")
else:
    print(f"Directory {image_folder} already exists.")

# Load or create the CSV file for extracted text
if os.path.exists(output_csv_path):
    extracted_text_df = pd.read_csv(output_csv_path)
    print(f"CSV loaded from {output_csv_path}")
else:
    extracted_text_df = pd.DataFrame(columns=['image_name', 'extracted_text'])
    extracted_text_df.to_csv(output_csv_path, index=False)
    print(f"Created a new CSV at {output_csv_path}")

# Function to download images
def download_images(image_url, save_directory, image_name):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(os.path.join(save_directory, image_name), 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {image_name}")
        else:
            print(f"Failed to download {image_name}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading {image_name}: {e}")

# Image processing functions
def Preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert('L')      # Convert to grayscale
        img = img.resize((800, 800))
        return img
    except Exception as e:
        print(f"Error Processing {image_path}: {e}")
        return None

def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()        # Remove extra spaces
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def cleaning_extracted_text(text):
    try:
        text = re.sub(r'[^0-9a-zA-Z.\S]', '', text)
        match = re.search(r'(\d+(\.\d+)?\s*[a-zA-Z]+)', text)
        if match:
            cleaned_text = match.group(0)
        else:
            cleaned_text = ""
        return cleaned_text.strip()
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ""

# Load dataset
train_df = pd.read_csv(train_data_path)

# Initialize the list to collect new rows
new_rows = []

# Process images and save progress in batches
batch_size = 1001  # Adjust the batch size as needed

for index, row in train_df.iterrows():
    image_name = f"{row.name}.jpg"
    image_path = os.path.join(image_folder, image_name)

    # Download image if not present
    if not os.path.exists(image_path):
        image_url = row['image_link']
        download_images(image_url, image_folder, image_name)
    else:
        print(f"Image {image_name} already exists. Skip downloading.")

    # Process image if it is present
    if os.path.exists(image_path):
        img = Preprocess_image(image_path)
        if img is not None:
            extracted_text = extract_text_from_image(img)
            cleaned_text = cleaning_extracted_text(extracted_text)
            print(f"Cleaned text from {image_name}: {cleaned_text}")

            # Append new row regardless of whether cleaned_text is empty
            new_rows.append({'image_name': image_name, 'extracted_text': cleaned_text})
        else:
            print(f"Skipping {image_name} due to processing issues.")
    else:
        print(f"{image_name} not found. Skipping.")

    # Save progress to CSV in batches
    if len(new_rows) >= batch_size:
        print(f"Saving batch of {len(new_rows)} rows to CSV.")
        new_rows_df = pd.DataFrame(new_rows)
        if os.path.exists(output_csv_path):
            extracted_text_df = pd.read_csv(output_csv_path)
            extracted_text_df = pd.concat([extracted_text_df, new_rows_df], ignore_index=True)
        else:
            extracted_text_df = new_rows_df
        extracted_text_df.to_csv(output_csv_path, index=False)
        new_rows = []  # Clear the list after saving

# Save any remaining rows after the loop
if new_rows:
    print(f"Saving final batch of {len(new_rows)} rows to CSV.")
    new_rows_df = pd.DataFrame(new_rows)
    if os.path.exists(output_csv_path):
        extracted_text_df = pd.read_csv(output_csv_path)
        extracted_text_df = pd.concat([extracted_text_df, new_rows_df], ignore_index=True)
    else:
        extracted_text_df = new_rows_df
    extracted_text_df.to_csv(output_csv_path, index=False)
