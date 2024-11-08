from transformers import pipeline
import torch 
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device used is", device)
classifier=pipeline("zero-shot-classification",device=device)
text="What is this about"
candidate_labels=["positive","neutral","negative"]
result=classifier(text,candidate_labels)
print(result)