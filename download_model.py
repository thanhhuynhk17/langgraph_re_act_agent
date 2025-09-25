# download model from huggingface
import os # Optional for faster downloading
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import snapshot_download
snapshot_download(
  repo_id = "BAAI/bge-reranker-v2-m3",
  local_dir = "./models/bge-reranker-v2-m3",
#   allow_patterns = ["*Q6_K_XL*"],
)