from huggingface_hub import HfApi
from huggingface_hub import login
import os
from dotenv import load_dotenv
load_dotenv()
login(os.environ.get("HUGGING_FACE_TOKEN"))
api = HfApi()
print(api.list_models())
from huggingface_hub import snapshot_download
snapshot_download(repo_id="meta-llama/Llama-2-7b-chat-hf", cache_dir="./meta-llama/Llama-2-7b-chat-hf",token=os.environ.get("HUGGING_FACE_TOKEN"))