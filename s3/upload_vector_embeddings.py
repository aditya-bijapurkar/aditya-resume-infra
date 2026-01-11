import json
import boto3
import os
import openai

# create a file in the same directory named personal_data.txt and add data to be used in the knowledge base

MODEL = "text-embedding-3-small"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TEXT_FILENAME = os.path.join(BASE_DIR, "personal_data.txt")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_BUCKET_VECTOR_EMBEDDINGS_KEY = os.getenv("S3_BUCKET_VECTOR_EMBEDDINGS_KEY")

if not all([OPENAI_API_KEY, AWS_REGION, S3_BUCKET_NAME, S3_BUCKET_VECTOR_EMBEDDINGS_KEY]):
    raise ValueError("Missing required environment variables")

client = openai.OpenAI(api_key=OPENAI_API_KEY)
s3_client = boto3.client("s3", region_name=AWS_REGION)

with open(LOCAL_TEXT_FILENAME, "r") as file:
    text_chunks = file.read().splitlines()

vector_embeddings = []

index = 0
for chunk in text_chunks:
    chunk = chunk.strip()
    if not chunk or chunk.startswith("#"):
        continue
    
    print(f"Processing chunk {index+1}...: {chunk}")
    index += 1
    response = client.embeddings.create(
        model=MODEL,
        input=chunk
    )
    vector_embedding = response.data[0].embedding    

    vector_embeddings.append({
        "text": chunk,
        "vectorEmbeddings": vector_embedding
    })

with open("local_vector_embeddings.json", "w") as file:
    json.dump(vector_embeddings, file)

try:
    s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=S3_BUCKET_VECTOR_EMBEDDINGS_KEY)
    print(f"Deleted existing file from s3://{S3_BUCKET_NAME}/{S3_BUCKET_VECTOR_EMBEDDINGS_KEY}")
except s3_client.exceptions.NoSuchKey:
    print(f"No existing file found at s3://{S3_BUCKET_NAME}/{S3_BUCKET_VECTOR_EMBEDDINGS_KEY}")

s3_client.upload_file("local_vector_embeddings.json", S3_BUCKET_NAME, S3_BUCKET_VECTOR_EMBEDDINGS_KEY)
print(f"Uploaded to s3://{S3_BUCKET_NAME}/{S3_BUCKET_VECTOR_EMBEDDINGS_KEY}")

os.remove("local_vector_embeddings.json")