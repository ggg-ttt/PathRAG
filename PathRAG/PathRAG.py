# 导入必要的库和模块
import asyncio  # 用于处理异步操作
import os  # 用于文件系统操作
from tqdm.asyncio import tqdm as tqdm_async  # 用于异步进度条显示
from dataclasses import asdict, dataclass, field  # 用于数据类定义
from datetime import datetime  # 用于获取当前时间
from functools import partial  # 用于函数部分应用
from typing import Type, cast  # 用于类型提示和类型转换


# 从项目内部模块导入所需的功能
from .llm import (
    gpt_4o_mini_complete,  # GPT-4o-mini模型的完成函数
    openai_embedding,  # OpenAI的嵌入向量生成函数
)
from .operate import (
    chunking_by_token_size,  # 根据令牌大小进行文本分块的函数
    extract_entities,  # 从文本中提取实体和关系的函数
    kg_query,  # 知识图谱查询函数
)

from .utils import (
    EmbeddingFunc,  # 嵌入函数的类型定义
    compute_mdhash_id,  # 计算内容的MD哈希ID
    limit_async_func_call,  # 限制异步函数调用频率的装饰器
    convert_response_to_json,  # 将响应转换为JSON格式
    logger,  # 日志记录器
    set_logger,  # 设置日志记录器
)
from .base import (
    BaseGraphStorage,  # 图存储的抽象基类
    BaseKVStorage,  # 键值存储的抽象基类
    BaseVectorStorage,  # 向量存储的抽象基类
    StorageNameSpace,  # 存储命名空间接口
    QueryParam,  # 查询参数数据类
)

from .storage import (
    JsonKVStorage,  # JSON键值存储实现
    NanoVectorDBStorage,  # 轻量级向量数据库存储实现
    NetworkXStorage,  # 基于NetworkX的图存储实现
)




def lazy_external_import(module_name: str, class_name: str):
    """
    懒加载外部模块中的类，基于调用者的包路径
    
    Args:
        module_name: 要导入的模块名称
        class_name: 要导入的类名称
        
    Returns:
        function: 一个函数，调用时会导入指定的类并返回其实例
    """
    # 使用inspect模块获取调用者的帧
    import inspect
    
    # 获取调用该函数的帧
    caller_frame = inspect.currentframe().f_back
    # 获取调用者所在的模块
    module = inspect.getmodule(caller_frame)
    # 获取调用者模块的包名
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        """
        内部函数，负责实际导入模块并创建类实例
        
        Args:
            *args: 传递给类构造函数的位置参数
            **kwargs: 传递给类构造函数的关键字参数
            
        Returns:
            导入的类的实例
        """
        import importlib
        
        # 导入指定的模块
        module = importlib.import_module(module_name, package=package)
        
        # 获取模块中的类
        cls = getattr(module, class_name)
        # 创建并返回类的实例
        return cls(*args, **kwargs)

    return import_class


# 使用懒加载方式定义外部存储实现，避免不必要的模块导入
# Neo4J图数据库存储实现
Neo4JStorage = lazy_external_import(".kg.neo4j_impl", "Neo4JStorage")
# Oracle键值存储实现
OracleKVStorage = lazy_external_import(".kg.oracle_impl", "OracleKVStorage")
# Oracle图存储实现
OracleGraphStorage = lazy_external_import(".kg.oracle_impl", "OracleGraphStorage")
# Oracle向量数据库存储实现
OracleVectorDBStorage = lazy_external_import(".kg.oracle_impl", "OracleVectorDBStorage")
# Milvus向量数据库存储实现
MilvusVectorDBStorge = lazy_external_import(".kg.milvus_impl", "MilvusVectorDBStorge")
# MongoDB键值存储实现
MongoKVStorage = lazy_external_import(".kg.mongo_impl", "MongoKVStorage")
# Chroma向量数据库存储实现
ChromaVectorDBStorage = lazy_external_import(".kg.chroma_impl", "ChromaVectorDBStorage")
# TiDB键值存储实现
TiDBKVStorage = lazy_external_import(".kg.tidb_impl", "TiDBKVStorage")
# TiDB向量数据库存储实现
TiDBVectorDBStorage = lazy_external_import(".kg.tidb_impl", "TiDBVectorDBStorage")
# AGE (Apache AGE) 图存储实现
AGEStorage = lazy_external_import(".kg.age_impl", "AGEStorage")


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    确保始终有可用的事件循环
    
    该函数尝试获取当前事件循环。如果当前事件循环已关闭或不存在，
    它会创建一个新的事件循环并将其设置为当前事件循环。
    
    Returns:
        asyncio.AbstractEventLoop: 当前的或新创建的事件循环
    """
    try:
        # 尝试获取当前事件循环
        current_loop = asyncio.get_event_loop()
        # 检查事件循环是否已关闭
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop
    except RuntimeError:
        # 如果无法获取有效的事件循环，则创建新的
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class PathRAG:
    """
    PathRAG主类 - 基于关系路径的图检索增强生成系统
    
    该类实现了基于知识图谱的检索增强生成，通过提取文档中的实体和关系，
    构建图结构，并在查询时利用实体间的路径关系进行增强检索。
    """
    # 工作目录，用于存储缓存和数据文件
    working_dir: str = field(
        default_factory=lambda: f"./PathRAG_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    # 嵌入向量缓存配置
    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,  # 是否启用嵌入缓存
            "similarity_threshold": 0.95,  # 相似度阈值，超过此阈值则视为相同嵌入
            "use_llm_check": False,  # 是否使用LLM进行额外检查
        }
    )
    # 存储类型配置
    kv_storage: str = field(default="JsonKVStorage")  # 键值存储类型
    vector_storage: str = field(default="NanoVectorDBStorage")  # 向量存储类型
    graph_storage: str = field(default="NetworkXStorage")  # 图存储类型

    # 日志级别配置
    current_log_level = logger.level  # 获取当前日志级别
    log_level: str = field(default=current_log_level)  # 设置日志级别


    # 文档分块配置
    chunk_token_size: int = 1200  # 每个分块的最大令牌数
    chunk_overlap_token_size: int = 100  # 分块之间的重叠令牌数
    tiktoken_model_name: str = "gpt-4o-mini"  # 用于令牌计数的模型名称


    # 实体提取配置
    entity_extract_max_gleaning: int = 1  # 实体提取的最大收集轮数
    entity_summary_to_max_tokens: int = 500  # 实体描述汇总的最大令牌数


    # 节点嵌入算法配置
    node_embedding_algorithm: str = "node2vec"  # 节点嵌入算法
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,  # 嵌入维度
            "num_walks": 10,  # 每个节点的随机游走次数
            "walk_length": 40,  # 随机游走长度
            "window_size": 2,  # 窗口大小
            "iterations": 3,  # 迭代次数
            "random_seed": 3,  # 随机种子
        }
    )


    # 嵌入函数配置
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)  # 嵌入函数
    embedding_batch_num: int = 32  # 嵌入批处理数量
    embedding_func_max_async: int = 16  # 嵌入函数的最大异步并发数


    # 语言模型配置
    llm_model_func: callable = gpt_4o_mini_complete  # LLM完成函数
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  # LLM模型名称
    llm_model_max_token_size: int = 32768  # LLM模型的最大令牌数
    llm_model_max_async: int = 16  # LLM的最大异步并发数
    llm_model_kwargs: dict = field(default_factory=dict)  # 传递给LLM的额外参数


    # 向量数据库存储参数
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)  # 传递给向量存储类的额外参数

    # LLM缓存配置
    enable_llm_cache: bool = True  # 是否启用LLM响应缓存


    # 附加参数和响应处理
    addon_params: dict = field(default_factory=dict)  # 附加参数，可用于扩展功能
    convert_response_to_json_func: callable = convert_response_to_json  # 响应转JSON函数

    def __post_init__(self):
        """
        数据类初始化后的钩子函数，用于初始化组件和设置环境
        """
        # 创建工作目录（如果不存在）
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)

        # 设置日志文件和日志级别，日志写到工作目录下
        log_file = os.path.join(self.working_dir, "PathRAG.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")


        # 获取并设置各种存储类
        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class()[self.kv_storage]  # 获取键值存储类
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.vector_storage  # 获取向量存储类
        ]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[
            self.graph_storage  # 获取图存储类
        ]

        # 初始化LLM响应缓存
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",  # 缓存命名空间
                global_config=asdict(self),  # 全局配置
                embedding_func=None,  # 不需要嵌入函数
            )
            if self.enable_llm_cache  # 仅在启用缓存时创建
            else None
        )
        # 限制嵌入函数的异步调用频率
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )


        # 初始化文档和文本分块存储
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",  # 完整文档存储命名空间
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",  # 文本分块存储命名空间
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",  # 实体关系图存储命名空间
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )


        # 初始化向量数据库存储
        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",  # 实体向量存储命名空间
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},  # 实体名称元数据字段
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",  # 关系向量存储命名空间
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},  # 关系源和目标实体ID元数据字段
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",  # 文本块向量存储命名空间
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        # 限制LLM模型函数的异步调用频率并设置缓存
        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                # 设置LLM响应缓存
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    global_config=asdict(self),
                ),
                **self.llm_model_kwargs,  # 传递额外的LLM参数
            )
        )

    def _get_storage_class(self) -> dict:
        """
        获取所有可用的存储类映射
        
        Returns:
            dict: 存储类名称到存储类的映射
        """
        return {
            # 键值存储类
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            "MongoKVStorage": MongoKVStorage,
            "TiDBKVStorage": TiDBKVStorage,

            # 向量存储类
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "MilvusVectorDBStorge": MilvusVectorDBStorge,
            "ChromaVectorDBStorage": ChromaVectorDBStorage,
            "TiDBVectorDBStorage": TiDBVectorDBStorage,

            # 图存储类
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            "AGEStorage": AGEStorage,
        }

    def insert(self, string_or_strings):
        """
        同步方式插入文档或文档列表
        
        Args:
            string_or_strings: 单个文档字符串或文档字符串列表
            
        Returns:
            异步插入操作的结果
        """
        # 获取事件循环并运行异步插入操作
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        """
        异步方式插入文档或文档列表
        
        Args:
            string_or_strings: 单个文档字符串或文档字符串列表
        """
        update_storage = False  # 标记是否需要更新存储
        try:
            # 确保输入是列表格式
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            # 为每个文档计算MD哈希ID并创建文档字典
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            # 过滤掉已经存在的文档
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            # 如果没有新文档，提前返回
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True  # 标记需要更新存储
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # 初始化文本块存储
            inserting_chunks = {}
            # 使用异步进度条处理每个文档
            for doc_key, doc in tqdm_async(
                new_docs.items(), desc="Chunking documents", unit="doc"
            ):
                # 对文档内容进行分块
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,  # 保留原始分块数据
                        "full_doc_id": doc_key,  # 添加完整文档ID引用
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,  # 重叠大小
                        max_token_size=self.chunk_token_size,  # 最大分块大小
                        tiktoken_model=self.tiktoken_model_name,  # 令牌计数模型
                    )
                }
                inserting_chunks.update(chunks)
            # 过滤掉已经存在的文本块
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            # 如果没有新文本块，提前返回
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            # 将文本块插入向量数据库
            await self.chunks_vdb.upsert(inserting_chunks)

            # 提取实体和关系
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                inserting_chunks,  # 要处理的文本块
                knowledge_graph_inst=self.chunk_entity_relation_graph,  # 知识图谱实例
                entity_vdb=self.entities_vdb,  # 实体向量数据库
                relationships_vdb=self.relationships_vdb,  # 关系向量数据库
                global_config=asdict(self),  # 全局配置
            )
            # 如果没有提取到新的实体和关系，提前返回
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            # 更新知识图谱实例
            self.chunk_entity_relation_graph = maybe_new_kg

            # 将完整文档和文本块插入存储
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            # 无论执行是否成功，如果需要更新存储，则调用完成回调
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        """
        插入完成后的回调函数，确保所有存储都完成索引等后处理操作
        """
        tasks = []
        # 收集所有存储实例的索引完成回调任务
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            # 将存储实例转换为StorageNameSpace并调用其索引完成回调
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        # 并行执行所有回调任务
        await asyncio.gather(*tasks)

    def insert_custom_kg(self, custom_kg: dict):
        """
        同步方式插入自定义知识图谱
        
        Args:
            custom_kg: 包含chunks、entities和relationships的自定义知识图谱数据
            
        Returns:
            异步插入操作的结果
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_custom_kg(custom_kg))

    async def ainsert_custom_kg(self, custom_kg: dict):
        """
        异步方式插入自定义知识图谱
        
        Args:
            custom_kg: 包含chunks、entities和relationships的自定义知识图谱数据
        """
        update_storage = False
        try:
            # 处理文本块数据
            all_chunks_data = {}  # 存储所有文本块
            chunk_to_source_map = {}  # 源ID到文本块ID的映射
            for chunk_data in custom_kg.get("chunks", []):  # 获取所有文本块
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                # 为文本块计算唯一ID
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                # 创建文本块条目并存储
                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True  # 标记需要更新存储

            # 将文本块插入向量数据库和键值存储
            if self.chunks_vdb is not None and all_chunks_data:
                await self.chunks_vdb.upsert(all_chunks_data)
            if self.text_chunks is not None and all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

 
            # 处理实体数据
            all_entities_data = []
            for entity_data in custom_kg.get("entities", []):  # 获取所有实体
                # 格式化实体名称（大写并加引号）
                entity_name = f'"{entity_data["entity_name"].upper()}"'
                entity_type = entity_data.get("entity_type", "UNKNOWN")  # 获取实体类型，默认为UNKNOWN
                description = entity_data.get("description", "No description provided")  # 获取实体描述

                # 获取源信息
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")  # 映射到文本块ID

                # 记录来源未知的警告
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # 准备节点数据
                node_data = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                }

                # 将实体节点插入知识图谱
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                # 添加实体名称到节点数据，用于向量存储
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True


            # 处理关系数据
            all_relationships_data = []
            for relationship_data in custom_kg.get("relationships", []):  # 获取所有关系
                # 格式化源实体和目标实体ID
                src_id = f'"{relationship_data["src_id"].upper()}"'
                tgt_id = f'"{relationship_data["tgt_id"].upper()}"'
                description = relationship_data["description"]  # 关系描述
                keywords = relationship_data["keywords"]  # 关系关键词
                weight = relationship_data.get("weight", 1.0)  # 关系权重，默认为1.0

                # 获取源信息
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # 记录来源未知的警告
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # 确保源实体和目标实体都存在于图中
                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        # 如果实体不存在，创建默认实体节点
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                            },
                        )

                # 将关系边插入知识图谱
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                    },
                )
                # 准备关系数据用于向量存储
                edge_data = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                }
                all_relationships_data.append(edge_data)
                update_storage = True


            # 将实体数据插入向量数据库
            if self.entities_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "content": dp["entity_name"] + dp["description"],  # 组合实体名称和描述作为向量内容
                        "entity_name": dp["entity_name"],  # 存储实体名称作为元数据
                    }
                    for dp in all_entities_data
                }
                await self.entities_vdb.upsert(data_for_vdb)


            # 将关系数据插入向量数据库
            if self.relationships_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                        "src_id": dp["src_id"],  # 源实体ID
                        "tgt_id": dp["tgt_id"],  # 目标实体ID
                        "content": dp["keywords"] + dp["src_id"] + dp["tgt_id"] + dp["description"],  # 组合关键词、实体ID和描述作为向量内容
                    }
                    for dp in all_relationships_data
                }
                await self.relationships_vdb.upsert(data_for_vdb)
        finally:
            # 无论执行是否成功，如果需要更新存储，则调用完成回调
            if update_storage:
                await self._insert_done()
    
    def query(self, query: str, param: QueryParam = QueryParam()):
        """
        同步方式执行查询
        
        Args:
            query: 查询字符串
            param: 查询参数，包含查询模式、上下文长度等配置
            
        Returns:
            查询结果
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))
    
    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        """
        异步方式执行查询
        
        Args:
            query: 查询字符串
            param: 查询参数，包含查询模式、上下文长度等配置
            
        Returns:
            查询结果
            
        Raises:
            ValueError: 当查询模式未知时抛出
        """
        # 检查查询模式是否支持
        if param.mode in ["hybrid", "local"]:  # 支持混合模式和本地模式
            response = await kg_query(
                query,  # 查询文本
                self.chunk_entity_relation_graph,  # 实体关系图
                self.entities_vdb,  # 实体向量数据库
                self.relationships_vdb,  # 关系向量数据库
                self.text_chunks,  # 文本块存储
                param,  # 查询参数
                asdict(self),  # 全局配置，即PathRAG的所有属性，从workingdir那些
                # 设置哈希KV存储用于缓存
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    global_config=asdict(self),
                ),
            )
            print("response all ready")  # 打印响应准备就绪的提示
        else:
            raise ValueError(f"Unknown mode {param.mode}")  # 抛出未知模式异常
        await self._query_done()  # 调用查询完成回调
        return response

        
    async def _query_done(self):
        """
        查询完成后的回调函数，确保相关存储完成后处理操作
        """
        tasks = []
        # 目前只处理LLM响应缓存的回调
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            # 将存储实例转换为StorageNameSpace并调用其索引完成回调
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        # 并行执行所有回调任务
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        """
        同步方式删除指定实体及其相关关系
        
        Args:
            entity_name: 要删除的实体名称
            
        Returns:
            异步删除操作的结果
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        """
        异步方式删除指定实体及其相关关系
        
        Args:
            entity_name: 要删除的实体名称
        """
        # 格式化实体名称（大写并加引号）
        entity_name = f'"{entity_name.upper()}"'

        try:
            # 从实体向量数据库中删除实体
            await self.entities_vdb.delete_entity(entity_name)
            # 从关系向量数据库中删除与该实体相关的关系
            await self.relationships_vdb.delete_relation(entity_name)
            # 从知识图谱中删除实体节点
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            # 记录删除成功日志
            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            # 调用删除完成回调
            await self._delete_by_entity_done()
        except Exception as e:
            # 记录删除失败错误
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        """
        删除实体完成后的回调函数，确保相关存储完成后处理操作
        """
        tasks = []
        # 收集实体、关系和图存储的索引完成回调任务
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            # 将存储实例转换为StorageNameSpace并调用其索引完成回调
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        # 并行执行所有回调任务
        await asyncio.gather(*tasks)
