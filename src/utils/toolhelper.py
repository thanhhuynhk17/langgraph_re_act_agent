import pandas as pd
import re
import numpy as np
from underthesea import word_tokenize
import os
import random
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch
import torchaudio
import torchvision

def load_excel(path:str) -> pd.DataFrame:
    data = pd.read_csv(path)
    df = pd.DataFrame(data)

    # Chuẩn hóa cột current_price
    df["current_price"] = (
        df["current_price"]
        .astype(str)                                     # Đảm bảo dạng chuỗi
        .str.replace(r"[^\d]", "", regex=True)           # Chỉ giữ số
        .replace("", np.nan)                             # Chuỗi rỗng -> NaN
    )

    # Chuyển sang số (bỏ qua lỗi), rồi thêm hậu tố "vnd"
    df["current_price"] = df["current_price"].astype(float).dropna().astype("Int64").astype(str) + " vnd"

    return df

def convert_table_to_rows(df: pd.DataFrame) -> list:
    """
    Chuyển đổi DataFrame (bảng) thành một danh sách các chuỗi.
    Mỗi chuỗi đại diện cho một hàng, với các giá trị được nối lại với nhau.

    Args:
        df (pd.DataFrame): DataFrame đầu vào.

    Returns:
        list: Danh sách các chuỗi, mỗi chuỗi là một hàng.
    """
    result_list = []
    for index, row in df.iterrows():
        # Chuyển tất cả giá trị trong hàng thành chuỗi và nối chúng lại với nhau
        row_string = ", ".join(str(value) for value in row)
        result_list.append(row_string)
        
    cleaned_text = [" ".join(row_string.replace("<br>", "").split()) for row_string in result_list if row_string.strip() != ""]
    result_list = [re.sub(r'<[^>]*>|\s+', ' ', row_string).strip() for row_string in cleaned_text]
    
    return result_list

def remove_stopwords_vi(text: str = 'inputs là đoạn văn bản cụ thể của bạn' , path_documents_vi:str='src\stopwords-vietnamese.txt'):
    id_str = text.split(',')[0] + ','
    link_https_str = text.split(',')[1] + ','
    name_object = text.split(',')[2] + ','
    # print(id_str, link_https_str, name_object)
    
    text = ','.join(text.split(',')[3:])
    stop_words = set(open(str(path_documents_vi), encoding="utf-8").read().splitlines())
    tokens = str(word_tokenize(text, format="text")).split(',')
    filtered = [w.strip() for w in tokens if w.lower() not in stop_words]
    return id_str + link_https_str + name_object + ', '.join(filtered)

def clean_text(text):
    """
    Làm sạch văn bản:
    - Giữ nguyên ID (token đầu tiên) và link (https, http, www, .com, .vn...).
    - Xóa ký tự đặc biệt, xuống dòng (\r, \n).
    - Xóa toàn bộ thẻ HTML (<br>, <div>...</div>, ...).
    - Thay thế '_' thành khoảng trắng.
    - Chuẩn hóa khoảng trắng.
    """
    if not text:
        return text

    # 1. Xóa thẻ HTML
    text = re.sub(r'<.*?>', ' ', text)

    # 2. Xóa ký tự xuống dòng và tab
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # 3. Thay thế dấu "_" thành khoảng trắng
    text = text.replace("_", " ")

    # 4. Tách token
    tokens = text.split(',')
    if not tokens:
        return text

    # Giữ lại ID (token đầu tiên)
    cleaned_tokens = [tokens[0]+',' + tokens[1]+',' + tokens[2] +',']

    # Regex nhận diện link (http, https, www, .com, .vn, .net, .org...)
    url_pattern = re.compile(r'(https?://\S+|www\.\S+|\S+\.(com|vn|net|org)\S*)')

    for token in tokens[3:]:
        if url_pattern.match(token):
            cleaned_tokens.append(token + ', ')  # Giữ nguyên link
        else:
            # Giữ chữ cái + số + khoảng trắng (loại bỏ ký tự đặc biệt)
            cleaned = re.sub(r'[^0-9a-zA-ZÀ-Ỹà-ỹ\s]', '', token)
            if cleaned:
                cleaned_tokens.append(cleaned + ',')

    # 5. Ghép lại và chuẩn hóa khoảng trắng
    return re.sub(r'\s+', ' ', " ".join(cleaned_tokens)).strip()


def normalize_record(text, fix_inch_heu=False):
    """
    Normalize a product text line:
    - remove extra commas, trailing commas
    - remove token 'nan' (case-insensitive)
    - replace underscores, remove HTML/newlines if any
    - normalize units: MB, GB, MP, mAh, 4G
    - collapse adjacent duplicate tokens (case-insensitive)
    - optionally apply heuristic to fix large 'inch' numbers (fix_inch_heu=True will convert e.g. 667->6.67 if detected)
    """
    if not text:
        return text

    # 1. basic cleanup
    text = re.sub(r'<.*?>', ' ', text)               # remove html tags
    text = text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. remove standalone 'nan'
    text = re.sub(r'\b[nN][aA][nN]\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 3. remove commas that are not in urls (simple approach: replace comma with space)
    #    keep url/ID intact by doing token-wise handling later
    # We'll split tokens and treat first token (ID) and tokens that match URL pattern specially.
    tokens = text.split()

    if not tokens:
        return text

    # Save leading ID (first token) and keep URLs intact
    id_token = tokens[0]
    url_pattern = re.compile(r'^(https?://\S+|www\.\S+|\S+\.(com|vn|net|org)\S*)$', flags=re.IGNORECASE)

    cleaned = [id_token]
    prev_tok_lower = None

    for tok in tokens[1:]:
        # keep url as-is
        if url_pattern.match(tok):
            cur = tok
        else:
            # remove commas / trailing punctuation
            cur = re.sub(r'[,\u200b]+', ' ', tok)
            cur = re.sub(r'^[\W_]+|[\W_]+$', '', cur)  # trim non-word at ends
            cur = cur.strip()

            # normalize units
            cur = re.sub(r'(?i)\b(\d+)\s*[gG][bB]\b', r'\1 GB', cur)
            cur = re.sub(r'(?i)\b(\d+)\s*[mM][bB]\b', r'\1 MB', cur)
            cur = re.sub(r'(?i)\b(\d+)\s*[mM][pP]\b', r'\1 MP', cur)
            cur = re.sub(r'(?i)\b(\d+)\s*[mM][aA][hH]\b', r'\1 mAh', cur)
            cur = re.sub(r'(?i)\b(\d+)\s*[gG]\b', r'\1G', cur)  # 4G

            # fix patterns like '128MB3' -> '128 MB' (remove stray trailing digits after unit)
            cur = re.sub(r'(\b\d+\s*(?:MB|GB|MP|mAh))3\b', r'\1', cur)

            # normalize common tokens
            cur = re.sub(r'(?i)\bips\s*lcd\b', 'IPS LCD', cur)
            cur = re.sub(r'(?i)\bfull\s*hd\s*\+?\b', 'Full HD+', cur)
            cur = re.sub(r'(?i)\bsnapdragon\s*(\d+)\s*g\b', lambda m: f"Snapdragon {m.group(1)}G", cur)

            # optional heuristic fix for inch numbers that are clearly too large for phones
            if fix_inch_heu:
                m = re.match(r'^(\d{2,3})\s*(?:inch|in)$', cur, flags=re.IGNORECASE)
                if m:
                    val = int(m.group(1))
                    if 50 <= val <= 999:
                        cur = f"{val/100:.2f} inch"

        if not cur:
            continue

        # collapse adjacent duplicate tokens (case-insensitive)
        if prev_tok_lower is not None and cur.lower() == prev_tok_lower:
            continue

        cleaned.append(cur)
        prev_tok_lower = cur.lower()

    # join and final normalize spaces and punctuation
    out = ' '.join(cleaned)
    out = re.sub(r'\s+,', ',', out)
    out = re.sub(r'\s+', ' ', out).strip()
    # remove trailing commas
    out = re.sub(r',\s*$', '', out)

    return out


# --------------------------
# Prompt templates
# --------------------------
prompts = [
    "{product_name} có không shop?",
    "{product_name} chip gì vậy?",
    "camera con {product_name} thế nào vậy ?",
    "Cho mình hỏi {product_name} được bao nhiêu gb ram vậy?",
    "{product_name} còn hàng không?",
    "{product_name} có màu nào và giá bao nhiêu vậy?",
]

def get_random_prompt(product_name: str) -> str:
    return random.choice(prompts).format(product_name=product_name)

# --------------------------
# Stub search (replace later)
# --------------------------
def run_hybrid_search(prompt: str, k: int):
    """
    Trả về list top-k kết quả (dict với _id).
    Bạn sẽ thay bằng search thực tế.
    """
    df = load_excel('src\hoanghamobile.csv')
    result_list = convert_table_to_rows(df)
    # print(2)
    # print(np.array(result_list))
    result_list = [remove_stopwords_vi(sequence) for sequence in result_list]
    result_list = [clean_text(sequence) for sequence in result_list]
    result_list = [normalize_record(sequence, fix_inch_heu=False) for sequence in result_list]

    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        device="cuda:0",
        # model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
    )

#     # MODELS EMBEDDING 1
    document_embeddings = model.encode(result_list)

    # MODELS EMBEDDING 2
    tokenized_corpus = [doc.split(" ") for doc in result_list]
    bm25 = BM25Okapi(tokenized_corpus)
    print('LOADED MODEL')
    query_embeddings = model.encode([prompt], prompt_name="query")
    print('LOADED MODEL')

    if k > 5 and k % 2 == 0:
        # Compute the (cosine) similarity between the query and document embeddings
        similarity = model.similarity(query_embeddings, document_embeddings)
        top_similarity_idx = np.argsort(-similarity.numpy().ravel())
        top_similarity_id_products = [[f"{int(similarity[0][idx]*100)}%", df.values[idx][0]] for idx in top_similarity_idx]
        
        tokenized_query = prompt.split(" ")
        result_k = list(bm25.get_top_n(tokenized_query, result_list, n=k))
    
        result_bm25 = [result_k[idx].split()[0] for idx in range(k//2)]
        result_qwen = [np.array(top_similarity_id_products)[_,1] for _ in range(k//2)]
        
        result_combined = [{"_id": np.array(top_similarity_id_products)[_,1], "_score": np.array(top_similarity_id_products)[_,0], '_content': result_list[top_similarity_idx[_]] } for _ in range(k)]
        # print(result_combined)
        # result_combined = [{"_id": np.array(result_combined)[_,1], "_score": np.array(top_similarity_id_products)[_,0]} for _ in range(k)]
        
    else:
        similarity = model.similarity(query_embeddings, document_embeddings)
        top_similarity_idx = np.argsort(-similarity.numpy().ravel())
        top_similarity_id_products = [[f"{int(similarity[0][idx]*100)}%", df.values[idx][0]] for idx in top_similarity_idx]
        
        result_qwen = [{"_id": np.array(top_similarity_id_products)[_,1], "_score": np.array(top_similarity_id_products)[_,0], '_content': result_list[top_similarity_idx[_]]} for _ in range(k)]
        result_combined = result_qwen
    
    return result_combined
    

# # --------------------------
# # Benchmark for dataframe
# # --------------------------
# def benchmark_df(df: pd.DataFrame, ks=(1, 5, 10)):
#     results = {f"hit@{k}": 0 for k in ks}
#     total = len(df)

#     for _, row in df.iterrows():
#         ground_truth_id = row["_id"]
#         # nếu bạn muốn lấy tên từ "title", thì thay "product_name" bằng "title"
#         product_name = ' '.join(row["title"].split('-')[:-1])
#         # print(product_name)
#         query = get_random_prompt(str(product_name).lower())
#         # print(f"[TEST] id={ground_truth_id} | query='{query}'")
        
#         for k in ks:
#             retrieved = run_hybrid_search(query, k)
#             id_list = [item["_id"].strip(",") for item in retrieved]
#             if ground_truth_id in id_list:
#                 results[f"hit@{k}"] += 1
#                 # print(f'{results[f"hit@{k}"]} [TRUE] {ground_truth_id} {k} {query} {retrieved}')
#             # else:
#                 # print(f'{results[f"hit@{k}"]} [FALSE] {ground_truth_id} {k, query} {retrieved}')
#             # if any(r["_id"] == ground_truth_id for r in retrieved):
#             #     results[f"hit@{k}"] += 1
        
#         # for k in ks:
#         #     id_list = [item["_id"].strip(",") for item in retrieved]
#         #     if ground_truth_id in id_list:
#         #         print(f'[TRUE] {results[f"hit@{k}"]} {ground_truth_id}, {query}')
#         #     else:
#         #         print('[FALSE]', results[f"hit@{k}"], ground_truth_id, query, retrieved)

#     for k in ks:
#         results[f"hit@{k}"] /= total # pyright: ignore[reportArgumentType]

    # return results


# print(run_hybrid_search('có redmi khong ban?', 10))

# # CSV thực tế
# df = pd.read_csv("src\hoanghamobile.csv")
# scores = run_hybrid_search('có iphone k?', 5)
# print(scores)
# # evals = []

# for i in range(1):
#     scores = benchmark_df(df, ks=(1, 5, 10))
#     evals.append([scores['hit@1'],scores['hit@5'],scores['hit@10']])
#     print(f'round {i}: {scores}')