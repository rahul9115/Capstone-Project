from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="********-****-****-****-************")
pc.create_index(
    name="quickstart",
    dimension=2, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)