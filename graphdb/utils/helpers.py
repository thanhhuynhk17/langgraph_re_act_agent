

# def docx_to_md(file_path: str) -> str:
#     from langchain_community.document_loaders import Docx2txtLoader
#     from markdownify import markdownify as md

#     # Step 1: Load the DOCX file
#     loader = Docx2txtLoader(file_path)
#     docs = loader.load()

#     # Step 2: Convert each document content to markdown
#     markdown_texts = [md(doc.page_content) for doc in docs]

#     return "\n".join(markdown_texts)

import unicodedata
# VietnameseToneNormalization.md
# https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md

TONE_NORM_VI = {
    'òa': 'oà', 'Òa': 'Oà', 'ÒA': 'OÀ',\
    'óa': 'oá', 'Óa': 'Oá', 'ÓA': 'OÁ',\
    'ỏa': 'oả', 'Ỏa': 'Oả', 'ỎA': 'OẢ',\
    'õa': 'oã', 'Õa': 'Oã', 'ÕA': 'OÃ',\
    'ọa': 'oạ', 'Ọa': 'Oạ', 'ỌA': 'OẠ',\
    'òe': 'oè', 'Òe': 'Oè', 'ÒE': 'OÈ',\
    'óe': 'oé', 'Óe': 'Oé', 'ÓE': 'OÉ',\
    'ỏe': 'oẻ', 'Ỏe': 'Oẻ', 'ỎE': 'OẺ',\
    'õe': 'oẽ', 'Õe': 'Oẽ', 'ÕE': 'OẼ',\
    'ọe': 'oẹ', 'Ọe': 'Oẹ', 'ỌE': 'OẸ',\
    'ùy': 'uỳ', 'Ùy': 'Uỳ', 'ÙY': 'UỲ',\
    'úy': 'uý', 'Úy': 'Uý', 'ÚY': 'UÝ',\
    'ủy': 'uỷ', 'Ủy': 'Uỷ', 'ỦY': 'UỶ',\
    'ũy': 'uỹ', 'Ũy': 'Uỹ', 'ŨY': 'UỸ',\
    'ụy': 'uỵ', 'Ụy': 'Uỵ', 'ỤY': 'UỴ'
    }

import unicodedata

def normalize_vnese(text):
    for i, j in TONE_NORM_VI.items():
        text = text.replace(i, j)
    # Normalize input text to NFC
    text = unicodedata.normalize("NFC", text)
    # normalize spacing
    text = text.replace('\xa0', ' ')
    return text

from langchain_community.document_loaders import Docx2txtLoader, JSONLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_neo4j import Neo4jVector
import os

def text_to_chunks(file_path: str, file_ext: str, chunk_size=512, chunk_overlap=0):
    print("🔄 Loading documents...")
    print("file_ext", file_ext)
    if file_ext == "docx":
        loader = Docx2txtLoader(file_path)
        documents = loader.load() # 1 item only
        # txt to markdown format
        from markdownify import markdownify
        for doc in documents:
            # normalize Vietnamese text
            doc.page_content = normalize_vnese(doc.page_content)
            # to markdown format
            doc.page_content = markdownify(doc.page_content)

        # extract information prefix
        content = documents[0].page_content
        prefix_txt, commune_items = extract_prefixed_items(content)
        documents[0].page_content = "\n".join(commune_items)
    elif file_ext == "json":
        # Define the metadata extraction function.
        def metadata_func(record: dict, metadata: dict) -> dict:

            metadata["province"] = record.get("province")
            metadata["new"] = record.get("new")

            return metadata

        loader = JSONLoader(
            file_path=file_path,
            jq_schema='.[]',
            content_key="answer",
            metadata_func=metadata_func,
            # text_content=False
        )

        documents = loader.load()
        for d in documents:
            prefix_txt = f"""Thông tin sát nhập xã/phường/đặc khu của tỉnh/thành phố: {d.metadata["province"]}""".strip()
            d.page_content = normalize_vnese(f"{prefix_txt}\n{d.page_content}")
            
    elif file_ext == "csv":
        loader = CSVLoader(file_path,
            csv_args={
                'delimiter': ',',
                'quotechar': '"'
            },
            content_columns=["combined_info"],
            metadata_columns=["_id", "url"]
        )
        documents = loader.load()
        for d in documents:
            d.page_content = normalize_vnese(d.page_content)

    print("✂️ Splitting documents into chunks...")
    splitter = CharacterTextSplitter(separator="\n\n",chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(documents)
    # for d in docs:
    #     d.page_content = f"{prefix_txt}\n\n{d.page_content}"
        
    return docs




SUPPORTED_EXTS = ["docx", "json", "csv"]

async def process_and_embed_to_neo4j(
    embedding_model,
    file_path: str,
    neo4j_url: str,
    username: str,
    password: str,
    database: str = "neo4j",
    chunk_size: int = 512,
    chunk_overlap: int = 0,
    index_name: str = "fulltext_index"
):
    """
    Load .docx files, normalize Vietnamese content, split into chunks,
    and store embeddings in Neo4j with full-text search enabled.

    Args:
        embedding_model: embedding model.
        file_path_pattern (str): Glob pattern for ["docx", "json"] files.
        neo4j_url (str): URL for Neo4j (e.g., "bolt://localhost:7687").
        username (str): Neo4j username.
        password (str): Neo4j password.
        database (str): Neo4j database name (default: "neo4j").
        chunk_size (int): Size of each document chunk.
        chunk_overlap (int): Overlap between chunks.
        index_name (str): Name of the full-text index in Neo4j.
    """
    if embedding_model is None:
        raise ValueError("embedding_model must be provided (e.g., OpenAIEmbeddings, HuggingFaceEmbeddings, etc.)")
    
    file_ext = os.path.splitext(file_path)[1] # get ext
    file_ext = file_ext[1:] # remove the dot
    if file_ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file extension: '{file_ext}'. Supported extensions are: {', '.join(SUPPORTED_EXTS)}.")

    docs = text_to_chunks(file_path, file_ext, chunk_size, chunk_overlap)
    
    
    print(f"💾 Saving embeddings to Neo4j (DB: {database})...")
    batch_size = 64
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        
        Neo4jVector.from_documents(
            documents=batch,
            embedding=embedding_model,
            url=neo4j_url,
            username=username,
            password=password,
            database=database,
            index_name=index_name,
            # search_type="hybrid"
        )

    print(f"✅ Embedded {len(docs)} chunks to Neo4j vector store (index: {index_name})")
    # return vectorstore

import re
from typing import List, Tuple

def extract_prefixed_items(text: str) -> Tuple[str, List[str]]:
    # Cut off everything from "Điều 2" onwards
    text = re.split(r'\nĐiều 2\b', text)[0]

    # Pattern to extract the "prefix" block before indexed items
    prefix_pattern = re.compile(
        r'.*?NGHỊ QUYẾT.*?QUYẾT NGHỊ.*?như sau:',
        re.DOTALL
    )

    # Pattern to extract each indexed item (e.g., 1., 2., ...)
    item_pattern = re.compile(r'\d+\.\s+.*?(?=\n\d+\.|\Z)', re.DOTALL)

    # Find the prefix
    prefix_match = prefix_pattern.search(text)
    if not prefix_match:
        return "", []

    prefix = prefix_match.group().strip()
    # Get everything after the prefix
    remaining_text = text[prefix_match.end():]

    # Find all indexed items
    items = item_pattern.findall(remaining_text)

    # # Append prefix to each item (optional)
    # prefixed_items = [prefix + "\n" + item.strip() for item in items]

    return prefix, items
