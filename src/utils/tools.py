import os
from typing import Literal, Optional, List
from langchain.tools import tool
from src.utils.schemas import HybridSearchInput
from src.utils.toolhelper import run_hybrid_search, run_load_data_to_embedding, run_normalization_data
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

search_tool = TavilySearch()

# module-level cache
_MODEL: Optional[SentenceTransformer] = None
_DOC_EMBEDDINGS = None
_SEQUENCES: Optional[List[str]] = None


def get_model(device: str = "cuda:0") -> SentenceTransformer:
    """Khởi tạo model 1 lần và cache vào _MODEL."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            device=device,
            # model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
        )
    return _MODEL


def prepare_embeddings(sequences: List[str], device: str = "cuda:0"):
    """
    Tạo embeddings cho sequences và cache chúng.
    Gọi lần đầu tiên với danh sách documents (sequences). Lần sau sẽ dùng cache.
    """
    global _DOC_EMBEDDINGS, _SEQUENCES

    if _DOC_EMBEDDINGS is None:
        model_instance = get_model(device)
        _DOC_EMBEDDINGS = model_instance.encode(sequences)
        _SEQUENCES = sequences
    return _DOC_EMBEDDINGS


@tool("hybrid_search", args_schema=HybridSearchInput)
def hybrid_search(
    query: str,
    k: Literal[10, 20] = 10,
    sequences: Optional[List[str]] = None,
    doc_embeddings=None,
) -> str:
    """
    Hybrid search tool:
    - Tự động load dữ liệu nếu chưa có embeddings.
    - Nếu đã có embeddings trong cache thì không encode lại.
    - Có thể override bằng cách truyền `doc_embeddings` từ ngoài.
    """
    global _DOC_EMBEDDINGS, _SEQUENCES

    model_instance = get_model()

    # --- ưu tiên doc_embeddings được truyền vào ---
    if doc_embeddings is not None:
        emb = doc_embeddings
    else:
        if _DOC_EMBEDDINGS is None:
            # nếu chưa có embeddings thì load dữ liệu mặc định
            if sequences is None:
                path = "src/hoanghamobile.csv"
                sequences = run_load_data_to_embedding(path)
                sequences = run_normalization_data(sequences)

            emb = prepare_embeddings(sequences)
        else:
            emb = _DOC_EMBEDDINGS

    # --- Nếu sequences không truyền vào thì fallback dùng cache ---
    docs = sequences or _SEQUENCES
    if docs is None:
        raise ValueError("No documents available for search. Could not build embeddings.")

    # Thực hiện hybrid search
    result = run_hybrid_search(model_instance, query, emb, k)

    return "\n".join(
        [f"{id_result.get('_id', 'N/A')} - {id_result.get('_content', '')}" for id_result in result]
    )
