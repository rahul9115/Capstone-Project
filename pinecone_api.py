import pandas as pd
import pinecone
from sentence_transformers import SentenceTransformer
import os
pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment="us-west1-gcp")
index_name = "job-matching-index"
dim = 768  

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=dim)
index = pinecone.Index(index_name)

jobs_df = pd.read_csv("stackshare_jobs.csv")

model = SentenceTransformer('all-MiniLM-L6-v2')

job_vectors = []
for i, row in jobs_df.iterrows():
    job_desc = row['Job Description']
    vector = model.encode(job_desc)
    job_vectors.append({
        'id': str(i),
        'values': vector,
        'metadata': {
            'job_title': row['Job Title'],
            'skills': row['Required Skills'],
        }
    })

index.upsert(vectors=job_vectors)

resume_text = "Your extracted resume text here"
resume_vector = model.encode(resume_text)

query_result = index.query([resume_vector], top_k=5)

for match in query_result['matches']:
    print(f"Match Score: {match['score']}")
    print(f"Job Title: {match['metadata']['job_title']}")
    print(f"Required Skills: {match['metadata']['skills']}\n")
