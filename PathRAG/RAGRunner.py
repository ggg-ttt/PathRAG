import os
import asyncio
import torch
from .PathRAG import PathRAG, QueryParam
from .utils import EmbeddingFunc
from .llm import (
    hf_model_complete,
    hf_embedding,
    ms_model_complete,
    ms_embedding,
    local_model_complete,
    local_embedding,
    ollama_model_complete,
    ollama_embedding,
    vllm_model_complete,
    vllm_embedding,
)
from minirag.llm.openai import openai_complete_if_cache
from minirag.utils import locate_json_string_body_from_string
from transformers import AutoModel, AutoTokenizer
import modelscope as ms

class RAGRunner:
    def __init__(
        self,
        backend: str,
        working_dir: str,
        llm_model_name: str,
        embedding_model_name: str,
        llm_model_max_token_size: int = 4096,
        llm_model_kwargs: dict = None,
        embedding_dim: int = 384,
        embedding_max_token_size: int = 5000,
        device: str = None,
    ):
        self.backend = backend.lower()
        self.working_dir = working_dir
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.llm_model_max_token_size = llm_model_max_token_size
        self.llm_model_kwargs = llm_model_kwargs or {}
        self.embedding_dim = embedding_dim
        self.embedding_max_token_size = embedding_max_token_size
        # device 参数允许外部显式指定推理设备；若未传入，则优先使用 GPU，找不到 CUDA 再回落 CPU。
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)

        self.rag = None
        self._init_rag()

    def _init_rag(self):
        if self.backend == "hf":
            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            embed_model = AutoModel.from_pretrained(self.embedding_model_name).to(self.device).eval()
            embedding_func = EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=self.embedding_max_token_size,
                func=lambda texts: hf_embedding(texts, tokenizer=tokenizer, embed_model=embed_model),
            )
            llm_func = hf_model_complete

        elif self.backend == "local":
            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            embed_model = AutoModel.from_pretrained(self.embedding_model_name).to(self.device).eval()
            embedding_func = EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=self.embedding_max_token_size,
                func=lambda texts: local_embedding(texts, tokenizer=tokenizer, embed_model=embed_model),
            )
            llm_func = local_model_complete

        elif self.backend == "vllm":
            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            embed_model = AutoModel.from_pretrained(self.embedding_model_name).to(self.device).eval()
            embedding_func = EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=self.embedding_max_token_size,
                func=lambda texts: vllm_embedding(texts, tokenizer=tokenizer, embed_model=embed_model),
            )
            llm_func = vllm_model_complete

        elif self.backend == "openai":
            # 通过本地 vLLM OpenAI 兼容服务调用模型
            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            embed_model = AutoModel.from_pretrained(self.embedding_model_name).to(self.device).eval()
            embedding_func = EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=self.embedding_max_token_size,
                func=lambda texts: hf_embedding(texts, tokenizer=tokenizer, embed_model=embed_model),
            )

            vllm_base_url = os.environ.get("VLLM_SERVER_BASE_URL", "http://0.0.0.0:8001/v1")
            vllm_api_key = os.environ.get("VLLM_API_KEY", "dummy")

            async def vllm_server_complete(
                prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
            ):
                keyword_extraction = kwargs.pop("keyword_extraction", None)
                model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
                result = await openai_complete_if_cache(
                    model=model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    base_url=vllm_base_url,
                    api_key=vllm_api_key,
                    **kwargs,
                )
                if keyword_extraction:
                    return locate_json_string_body_from_string(result)
                return result

            llm_func = vllm_server_complete

        elif self.backend == "ollama":
            embedding_func = EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=self.embedding_max_token_size,
                func=lambda texts: ollama_embedding(texts, embed_model=self.embedding_model_name, host=self.llm_model_kwargs.get("host", "http://localhost:11434")),
            )
            llm_func = ollama_model_complete

        elif self.backend == "ms":
            tokenizer = ms.AutoTokenizer.from_pretrained(self.embedding_model_name, trust_remote_code=True)
            embed_model = ms.AutoModel.from_pretrained(self.embedding_model_name, trust_remote_code=True).to(self.device).eval()
            embedding_func = EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=self.embedding_max_token_size,
                func=lambda texts: ms_embedding(texts, tokenizer=tokenizer, embed_model=embed_model),
            )
            llm_func = ms_model_complete

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.rag = PathRAG(
            working_dir=self.working_dir,
            llm_model_func=llm_func,
            llm_model_name=self.llm_model_name,
            llm_model_max_token_size=self.llm_model_max_token_size,
            llm_model_kwargs=self.llm_model_kwargs,
            embedding_func=embedding_func,
        )

    def insert_text(self, text: str):
        self.rag.insert(text)

    def query(self, question: str, mode: str = "hybrid"):
        #手动设置本次查询的参数
        query_param = QueryParam(
            mode="hybrid",
            top_k=30,
            max_token_for_text_unit=2000,      # 从 4000 减少到 2000
            max_token_for_global_context=2000, # 从 3000 减少到 2000
            max_token_for_local_context=2000 # 从 5000 减少到 2000
    # 总和：2000 + 2000 + 2000 = 6000 tokens（留出 ~2000 tokens 给提示词和查询）
    )
        return self.rag.query(question, param=query_param)