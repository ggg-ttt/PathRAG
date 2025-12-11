# 导入必要的Python标准库
import asyncio  # 异步IO支持
import json     # JSON数据处理
import re       # 正则表达式支持
from tqdm.asyncio import tqdm as tqdm_async  # 异步进度条
from typing import Union  # 类型注解支持
from collections import Counter, defaultdict  # 计数器和默认字典
import warnings  # 警告处理
import tiktoken  # OpenAI的token计数工具
import time      # 时间操作
import csv       # CSV文件处理

# 导入内部工具函数
from .utils import (
    logger,                 # 日志记录器
    clean_str,              # 字符串清理函数
    compute_mdhash_id,      # 计算MD5哈希ID
    decode_tokens_by_tiktoken,  # 使用tiktoken解码token
    encode_string_by_tiktoken,  # 使用tiktoken编码字符串
    is_float_regex,         # 正则表达式判断是否为浮点数
    list_of_list_to_csv,    # 将列表列表转换为CSV格式
    pack_user_ass_to_openai_messages,  # 打包用户和助手消息为OpenAI格式
    split_string_by_multi_markers,      # 按多个标记分割字符串
    truncate_list_by_token_size,        # 按token大小截断列表
    process_combine_contexts,           # 处理和合并上下文
    compute_args_hash,      # 计算参数哈希
    handle_cache,           # 处理缓存
    save_to_cache,          # 保存到缓存
    CacheData,              # 缓存数据类
)

# 导入基础类和数据结构
from .base import (
    BaseGraphStorage,       # 图存储基类
    BaseKVStorage,          # 键值存储基类
    BaseVectorStorage,      # 向量存储基类
    TextChunkSchema,        # 文本块数据模式
    QueryParam,             # 查询参数类
)

# 导入常量和提示模板
from .prompt import GRAPH_FIELD_SEP, PROMPTS  # 图字段分隔符和提示词模板


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    """
    根据token大小将文本内容分块
    
    参数:
        content: str - 要分块的文本内容
        overlap_token_size: int - 相邻块之间的重叠token数量，默认为128
        max_token_size: int - 每个块的最大token数量，默认为1024
        tiktoken_model: str - 使用的tiktoken模型名称，默认为"gpt-4o"
    
    返回:
        list[dict] - 包含分块信息的字典列表，每个字典包含:
            - tokens: int - 块的token数量
            - content: str - 块的文本内容
            - chunk_order_index: int - 块的顺序索引
    """
    # 使用tiktoken将内容编码为tokens
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    
    # 遍历并创建分块，确保相邻块之间有重叠
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        # 解码指定范围的tokens为文本
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        # 添加块信息到结果列表
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),  # 实际token数量
                "content": chunk_content.strip(),  # 块内容（去除首尾空白）
                "chunk_order_index": index,  # 块的顺序索引
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """
    处理实体或关系的描述摘要，当描述过长时使用LLM进行摘要
    
    参数:
        entity_or_relation_name: str - 实体或关系的名称
        description: str - 需要处理的描述文本
        global_config: dict - 全局配置字典，包含LLM和token相关设置
    
    返回:
        str - 原始描述或摘要后的描述
    """
    # 从全局配置获取必要参数
    use_llm_func: callable = global_config["llm_model_func"]  # LLM模型调用函数
    llm_max_tokens = global_config["llm_model_max_token_size"]  # LLM最大token数
    tiktoken_model_name = global_config["tiktoken_model_name"]  # tiktoken模型名称
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]  # 摘要最大token数
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )  # 语言设置，默认为英语

    # 计算描述的token数量
    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    
    # 如果描述长度小于摘要最大长度，则直接返回原始描述
    if len(tokens) < summary_max_tokens: 
        return description
    
    # 否则使用LLM生成摘要
    prompt_template = PROMPTS["summarize_entity_descriptions"]  # 摘要提示模板
    
    # 确保描述不超过LLM的最大输入长度
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    
    # 准备摘要提示的上下文
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),  # 按分隔符分割描述
        language=language,
    )
    
    # 格式化提示模板
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")  # 记录触发摘要的实体/关系
    
    # 调用LLM生成摘要
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """
    处理单个实体的提取逻辑
    
    参数:
        record_attributes: list[str] - 包含实体属性信息的列表
        chunk_key: str - 文本块的唯一标识符
    
    返回:
        dict - 处理后的实体数据字典，包含实体名称、类型、描述等信息
    """
    """
    处理单个实体提取结果
    
    参数:
        record_attributes: list[str] - 从LLM响应中提取的记录属性列表
        chunk_key: str - 来源文本块的唯一标识符
    
    返回:
        dict or None - 提取的实体信息字典，格式不正确时返回None
    """
    # 验证记录格式是否正确
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
   
    # 提取并清理实体名称（转换为大写）
    entity_name = clean_str(record_attributes[1].upper())
    
    # 检查实体名称是否为空
    if not entity_name.strip():
        return None
    
    # 提取并清理实体类型（转换为大写）
    entity_type = clean_str(record_attributes[2].upper())
    
    # 提取并清理实体描述
    entity_description = clean_str(record_attributes[3])
    
    # 记录实体来源ID
    entity_source_id = chunk_key
    
    # 返回格式化的实体信息字典
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """
    处理单个关系的提取逻辑
    
    参数:
        record_attributes: list[str] - 包含关系属性信息的列表
        chunk_key: str - 文本块的唯一标识符
    
    返回:
        dict - 处理后的关系数据字典，包含源实体、目标实体、关系类型等信息
    """
    """
    处理单个关系提取结果
    
    参数:
        record_attributes: list[str] - 从LLM响应中提取的记录属性列表
        chunk_key: str - 来源文本块的唯一标识符
    
    返回:
        dict or None - 提取的关系信息字典，格式不正确时返回None
    """
    # 验证记录格式是否正确 - 确保记录至少有5个属性且第一个属性是关系标识
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
   
    # 提取并清理关系源实体（转换为大写以确保一致性）
    source = clean_str(record_attributes[1].upper())
    
    # 提取并清理关系目标实体（转换为大写以确保一致性）
    target = clean_str(record_attributes[2].upper())
    
    # 提取并清理关系描述文本
    edge_description = clean_str(record_attributes[3])

    # 提取并清理关系关键词，用于后续搜索和匹配
    edge_keywords = clean_str(record_attributes[4])
    
    # 记录关系来源ID，用于追踪关系的文本块来源
    edge_source_id = chunk_key
    
    # 提取权重值 - 如果最后一个属性是浮点数则使用，否则默认为1.0
    # 权重用于表示关系的重要性或置信度
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    
    # 返回格式化的关系信息字典，包含完整的关系数据
    return dict(
        src_id=source,        # 关系源实体ID
        tgt_id=target,        # 关系目标实体ID
        weight=weight,        # 关系权重值
        description=edge_description,  # 关系详细描述
        keywords=edge_keywords,       # 关系关键词
        source_id=edge_source_id,     # 关系来源文本块ID
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    合并相似节点并将其插入到知识图谱中
    
    参数:
        entity_name: str - 实体名称
        nodes_data: list[dict] - 包含节点数据的列表
        knowledge_graph_inst: BaseGraphStorage - 知识图谱存储实例
        global_config: dict - 全局配置字典
    
    返回:
        str - 处理后的实体节点ID
    """
    """
    合并多个节点数据并更新到知识图谱中
    
    参数:
        entity_name: str - 实体名称
        nodes_data: list[dict] - 包含相同实体的多个节点数据列表
        knowledge_graph_inst: BaseGraphStorage - 知识图谱存储实例
        global_config: dict - 全局配置字典
    
    返回:
        dict - 更新后的节点数据（包含实体名称）
    """
    # 初始化存储已存在节点信息的列表 - 用于后续合并
    already_entity_types = []  # 存储已存在节点的实体类型
    already_source_ids = []    # 存储已存在节点的来源ID
    already_description = []   # 存储已存在节点的描述

    # 检查知识图谱中是否已存在该实体
    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        # 如果存在，收集已有节点的信息用于后续合并
        already_entity_types.append(already_node["entity_type"])
        # 使用分隔符拆分并合并多个来源ID
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    # 合并实体类型 - 选择出现频率最高的类型作为最终类型
    # 使用Counter统计所有类型的出现次数，然后按频率排序选择最常见的
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],  # 按出现次数排序
        reverse=True,        # 降序排列以获取最高频率
    )[0][0]  # 获取最常见的类型
    
    # 合并描述文本 - 去重并排序以确保一致性
    # 合并新旧描述，去重后用字段分隔符连接
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    
    # 合并来源ID - 去重以避免重复引用
    # 合并所有来源ID，去重后用字段分隔符连接
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    
    # 处理合并后的描述 - 可能需要通过LLM进行摘要处理
    # 调用_handle_entity_relation_summary函数处理描述文本（可能进行截断或摘要）
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    
    # 准备节点数据字典 - 整合所有合并后的信息
    node_data = dict(
        entity_type=entity_type,  # 最终确定的实体类型
        description=description,  # 处理后的描述文本
        source_id=source_id,      # 所有关联的来源ID
    )
    
    # 更新或插入节点到知识图谱
    # 如果节点已存在则更新，不存在则插入新节点
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    
    # 添加实体名称到返回数据中，使返回的数据更完整
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    合并相似边并将其插入到知识图谱中
    
    参数:
        src_id: str - 源节点ID
        tgt_id: str - 目标节点ID
        edges_data: list[dict] - 包含边数据的列表
        knowledge_graph_inst: BaseGraphStorage - 知识图谱存储实例
        global_config: dict - 全局配置字典
    
    返回:
        dict - 处理后的边数据
    """
    """
    合并多个边数据并更新到知识图谱中
    
    参数:
        src_id: str - 源实体ID
        tgt_id: str - 目标实体ID
        edges_data: list[dict] - 包含相同关系的多个边数据列表
        knowledge_graph_inst: BaseGraphStorage - 知识图谱存储实例
        global_config: dict - 全局配置字典
    
    返回:
        dict - 更新后的边数据
    """
    # 初始化存储已存在边信息的列表 - 用于后续合并
    already_weights = []      # 存储已存在边的权重
    already_source_ids = []   # 存储已存在边的来源ID
    already_description = []  # 存储已存在边的描述
    already_keywords = []     # 存储已存在边的关键词

    # 检查知识图谱中是否已存在该边
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        # 获取已存在边的数据
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        # 收集已存在边的信息用于后续合并
        already_weights.append(already_edge["weight"])
        # 使用分隔符拆分并收集已存在边的所有来源ID
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        # 收集已存在边的描述文本
        already_description.append(already_edge["description"])
        # 使用分隔符拆分并收集已存在边的所有关键词
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    # 合并权重 - 将所有边的权重求和，反映关系的累计强度或置信度
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    
    # 合并描述文本 - 去重并排序以确保一致性和可读性
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    
    # 合并关键词 - 去重并排序以确保索引和搜索的一致性
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    
    # 合并来源ID - 去重以避免重复引用
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    
    # 确保源节点和目标节点存在 - 防止孤立边的产生
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            # 如果节点不存在，创建一个未知类型的默认节点
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,       # 关联当前边的来源ID
                    "description": description,   # 使用边的描述作为节点描述
                    "entity_type": '"UNKNOWN"',  # 标记为未知实体类型
                },
            )
    
    # 处理合并后的描述 - 可能需要通过LLM进行摘要处理
    # 使用源节点和目标节点的组合作为关系标识进行摘要
    description = await _handle_entity_relation_summary(
        f"({src_id}, {tgt_id})", description, global_config
    )
    
    # 更新或插入边到知识图谱
    # 如果边已存在则更新，不存在则插入新边
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,           # 合并后的累计权重
            description=description,  # 处理后的描述文本
            keywords=keywords,       # 合并后的关键词列表
            source_id=source_id,     # 所有关联的来源ID
        ),
    )

    # 准备返回的边数据 - 不包含权重和来源ID，专注于关系内容
    edge_data = dict(
        src_id=src_id,           # 源实体ID
        tgt_id=tgt_id,           # 目标实体ID
        description=description,  # 处理后的描述文本
        keywords=keywords,       # 合并后的关键词
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    """
    从文本块中提取实体和关系，并构建知识图谱
    
    参数:
        chunks: dict[str, TextChunkSchema] - 文本块字典，键为块ID，值为文本块对象
        knowledge_graph_inst: BaseGraphStorage - 知识图谱存储实例
        entity_vdb: BaseVectorStorage - 实体向量数据库
        relationships_vdb: BaseVectorStorage - 关系向量数据库
        global_config: dict - 全局配置字典
    
    返回:
        Union[BaseGraphStorage, None] - 更新后的知识图谱存储实例，发生错误时返回None
    """
    # 延迟20秒 - 可能是为了避免API限制或速率限制
    time.sleep(10)
    
    # 从全局配置获取必要参数
    use_llm_func: callable = global_config["llm_model_func"]  # LLM模型调用函数，用于文本分析和实体提取
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]  # 最大提取轮数，控制递归提取的深度

    # 将文本块字典转换为列表以便按顺序处理
    # 从字典视图转换为列表，保留键值对
    ordered_chunks = list(chunks.items())
  
    # 获取语言设置和实体类型配置
    # 如果未指定则使用默认值
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    
    # 根据配置选择示例数量
    # 获取用户配置的示例数量，如果未指定则为None
    example_number = global_config["addon_params"].get("example_number", None)
    
    # 根据配置选择适当数量的示例
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        # 如果指定了示例数量且小于可用示例总数，则使用指定数量的示例
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        # 否则使用所有可用示例
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    # 准备示例上下文参数 - 用于格式化示例文本
    # 这些参数将被替换到示例模板中的占位符位置
    # 这样可以确保所有示例都使用统一的分隔符和实体类型定义
    example_context_base = dict(
        # 元组分隔符：用于分隔单个实体或关系的不同属性
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        # 记录分隔符：用于分隔不同的实体或关系记录
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        # 完成分隔符：用于标记实体提取任务的完成
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        # 实体类型列表：转换为逗号分隔的字符串，限定LLM可以提取的实体类型
        entity_types=",".join(entity_types),
        # 语言设置：确保所有提示和输出与用户选择的语言一致
        language=language,
    )
  
    # 格式化示例文本 - 将示例上下文参数应用到示例模板中
    # 这一步骤将所有示例文本中的占位符替换为实际值，生成标准化的示例
    examples = examples.format(**example_context_base)

    # 准备实体提取提示模板
    # 获取预定义的实体提取提示模板，该模板包含了指导LLM如何提取实体和关系的指令
    entity_extract_prompt = PROMPTS["entity_extraction"]
    
    # 准备实体提取的上下文参数 - 用于格式化提取提示词
    # 这些参数将与输入文本一起用于构建完整的提示
    context_base = dict(
        # 元组分隔符：用于分隔实体或关系的属性值
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        # 记录分隔符：用于区分不同的实体或关系记录
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        # 完成分隔符：用于指示LLM何时停止提取
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        # 实体类型列表：限制LLM提取的实体类型范围
        entity_types=",".join(entity_types),
        # 已格式化的示例文本：展示给LLM看的实体和关系提取示例
        examples=examples,
        # 语言设置：确保提示词与用户使用的语言一致
        language=language,
    )

    # 获取继续提取和循环判断的提示模板
    # 用于指导LLM决定是否需要继续从文本中提取更多实体
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]  # 循环判断提示

    # 获取LLM配置参数，用于控制上下文长度
    llm_max_tokens = global_config["llm_model_max_token_size"]  # LLM最大token数
    tiktoken_model_name = global_config["tiktoken_model_name"]  # tiktoken模型名称

    # 辅助函数：截断对话历史，保留最近的对话
    def truncate_history(history, max_tokens, tiktoken_model_name):
        """
        截断对话历史，保留最近的对话
        
        参数:
            history: list - 对话历史列表
            max_tokens: int - 最大token数
            tiktoken_model_name: str - tiktoken模型名称
        
        返回:
            list - 截断后的对话历史
        """
        if not history or max_tokens <= 0:
            return history
        
        total_tokens = 0
        truncated_history = []
        
        # 从后往前遍历，保留最近的对话
        for msg in reversed(history):
            msg_text = msg.get("content", "")
            msg_tokens = encode_string_by_tiktoken(msg_text, model_name=tiktoken_model_name)
            msg_token_count = len(msg_tokens)
            
            if total_tokens + msg_token_count > max_tokens:
                break
            truncated_history.insert(0, msg)
            total_tokens += msg_token_count
        
        if len(truncated_history) < len(history):
            logger.debug(
                f"Truncated history from {len(history)} messages to {len(truncated_history)} messages "
                f"({total_tokens}/{max_tokens} tokens)"
            )
        
        return truncated_history

    # 初始化进度计数器
    already_processed = 0  # 已处理的文本块数量
    already_entities = 0   # 已提取的实体数量（含重复）
    already_relations = 0  # 已提取的关系数量（含重复）

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """
        处理单个文本块内容，提取实体和关系
        
        参数:
            chunk_key_dp: tuple[str, TextChunkSchema] - 包含块ID和块数据的元组
        
        返回:
            dict - 包含提取的节点和边的字典
        """
        # 使用nonlocal关键字修改外部函数的变量 - 用于跟踪全局处理进度
        nonlocal already_processed, already_entities, already_relations
        
        # 解包块ID和块数据
        chunk_key = chunk_key_dp[0]  # 文本块的唯一标识符
        chunk_dp = chunk_key_dp[1]   # 文本块的数据对象
        
        # 获取文本内容
        content = chunk_dp["content"]  # 提取文本块的实际内容
        
        # 计算提示模板（不含输入文本）的token数，用于确定可用的输入文本长度
        # 先构建不含输入文本的提示模板，用于计算模板本身的token数
        template_without_content = entity_extract_prompt.format(
            **context_base, input_text=""
        )
        template_tokens = encode_string_by_tiktoken(
            template_without_content, model_name=tiktoken_model_name
        )
        template_token_count = len(template_tokens)
        
        # 计算可用于输入文本的最大token数（预留一些空间给响应和系统开销）
        # 预留约500 tokens给响应和系统开销
        available_tokens = llm_max_tokens - template_token_count - 500
        
        # 如果可用token数小于0，说明模板本身就已经超过限制
        if available_tokens <= 0:
            logger.warning(
                f"Prompt template itself ({template_token_count} tokens) exceeds "
                f"max token size ({llm_max_tokens}). Consider reducing examples or prompt size."
            )
            # 至少保留100 tokens给输入文本
            available_tokens = max(100, llm_max_tokens - template_token_count)
        
        # 检查并截断输入文本
        content_tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model_name)
        if len(content_tokens) > available_tokens:
            logger.warning(
                f"Content too long ({len(content_tokens)} tokens) for chunk {chunk_key}, "
                f"truncating to {available_tokens} tokens"
            )
            content = decode_tokens_by_tiktoken(
                content_tokens[:available_tokens], model_name=tiktoken_model_name
            )
        
        # 构建实体提取提示 - 需要两次format是因为模板中包含嵌套的占位符
        # 第一次格式化设置基本上下文，第二次格式化插入实际输入文本
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        # 调用LLM进行初始实体和关系提取
        final_result = await use_llm_func(hint_prompt)

        # 记录提示和结果到日志文件
        logger.debug(f"=== Entity Extraction Prompt for chunk {chunk_key} ===\n{hint_prompt}")
        logger.debug(f"=== Entity Extraction Result for chunk {chunk_key} ===\n{final_result}")

        # 打包对话历史，用于后续的多轮提取
        # 保留上下文以实现连贯的多轮对话
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        
        # 多轮提取循环 - 通过多次询问LLM尝试提取更多实体和关系
        for now_glean_index in range(entity_extract_max_gleaning):
            # 在使用history前，限制其长度以确保不超过最大token数
            # 预留空间给当前请求的提示和响应（约1000 tokens）
            max_history_tokens = llm_max_tokens - 1000
            if max_history_tokens > 0:
                history = truncate_history(history, max_history_tokens, tiktoken_model_name)
            
            # 调用LLM继续提取 - 使用continue_prompt提示LLM查找更多实体/关系
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            # 更新对话历史和提取结果
            # 将继续提取的提示和响应添加到对话历史
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            # 累加提取结果
            final_result += glean_result
            
            # 如果达到最大提取轮数，退出循环
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            # 在使用history前，再次限制其长度以确保不超过最大token数
            # 预留空间给当前请求的提示和响应（约1000 tokens）
            max_history_tokens = llm_max_tokens - 1000
            if max_history_tokens > 0:
                history = truncate_history(history, max_history_tokens, tiktoken_model_name)
            
            # 询问是否需要继续提取 - 使用if_loop_prompt判断是否还有更多实体/关系
            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            
            # 清理响应并转换为小写进行比较
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            
            # 如果不需要继续提取，退出循环
            if if_loop_result != "yes":
                break

        # 记录最终累积的提取结果
        logger.debug(f"=== Final Accumulated Entity Extraction Result for chunk {chunk_key} ===\n{final_result}")

        # 分割提取结果，获取所有记录
        # 使用记录分隔符和完成分隔符分割LLM返回的文本，得到单个记录列表
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        # 初始化存储可能的节点和边的数据结构
        # 使用defaultdict自动创建空列表作为默认值
        maybe_nodes = defaultdict(list)  # 键为实体名称，值为实体数据列表
        maybe_edges = defaultdict(list)  # 键为(source, target)元组，值为关系数据列表
        
        # 处理每条记录 - 识别并分类为实体或关系
        for record in records:
            # 提取括号内的内容 - 假设记录格式为 "(属性1,属性2,...)"
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue  # 跳过无效记录格式
            
            # 获取括号内的内容作为记录属性
            record = record.group(1)
            
            # 分割记录属性 - 使用元组分隔符将字符串分割为属性列表
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            
            # 尝试作为实体记录处理
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            
            if if_entities is not None:
                # 如果是有效实体，添加到maybe_nodes字典
                # 使用实体名称作为键，方便后续合并相同实体
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue  # 跳过后续处理，处理下一条记录

            # 如果不是实体，则尝试作为关系记录处理
            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            
            if if_relation is not None:
                # 如果是有效关系，添加到maybe_edges字典
                # 使用(source_id, target_id)元组作为键，方便后续合并相同关系
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        # 更新进度计数器
        already_processed += 1  # 增加已处理块计数
        already_entities += len(maybe_nodes)  # 增加已提取实体计数
        already_relations += len(maybe_edges)  # 增加已提取关系计数
        
        # 获取当前进度指示器
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        
        # 打印进度信息（使用\r回车符实现单行更新）
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",  # 不换行
            flush=True,  # 立即刷新输出
        )
        
        # 返回提取的节点和边数据
        return dict(maybe_nodes), dict(maybe_edges)

    # 初始化结果列表
    results = []
    # 异步处理所有文本块，显示进度条
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),  # 总块数
        desc="Extracting entities from chunks",  # 进度条描述
        unit="chunk",  # 单位
    ):
        # 收集处理结果
        results.append(await result)

    # 合并所有处理结果中的节点和边
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        # 合并节点数据
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        # 合并边数据
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)
    # 记录日志
    logger.info("Inserting entities into storage...")
    # 初始化实体数据列表
    all_entities_data = []
    # 异步合并并插入所有实体，显示进度条
    for result in tqdm_async(
        asyncio.as_completed(
            [
                # 为每个实体创建合并任务
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),  # 总实体数
        desc="Inserting entities",  # 进度条描述
        unit="entity",
    ):
        all_entities_data.append(await result)

    logger.info("Inserting relationships into storage...")
    # 记录日志
    logger.info("Inserting relationships into knowledge graph...")
    # 初始化关系数据列表
    all_relationships_data = []
    # 异步合并并插入所有关系，显示进度条
    for result in tqdm_async(
        asyncio.as_completed(
            [
                # 为每个关系创建合并任务
                _merge_edges_then_upsert(
                    k[0], k[1], v, knowledge_graph_inst, global_config
                )
                for k, v in maybe_edges.items()
            ]
        ),
        total=len(maybe_edges),  # 总关系数
        desc="Inserting relationships",  # 进度条描述
        unit="relationship",
    ):
        # 收集合并后的关系数据
        all_relationships_data.append(await result)

    if not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any entities and relationships, maybe your LLM is not working"
        )
        return None

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    # 记录日志
    logger.info("Inserting entities into vector database...")
    # 准备实体向量数据库插入数据
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": f"{dp['entity_name']} ({dp['entity_type']}): {dp['description']}",
                "entity_name": dp["entity_name"],
                "metadata": {
                    "source_id": dp.get("source_id", ""),
                    "entity_type": dp["entity_type"],
                },
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    # 记录日志
    logger.info("Inserting relationships into vector database...")
    # 准备关系向量数据库插入数据
    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": f"{dp['src_id']} -> {dp['tgt_id']}: {dp['description']} | {dp['keywords']}",
                "metadata": {
                    "src_id": dp["src_id"],
                    "tgt_id": dp["tgt_id"],
                },
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)
    
    # 返回更新后的知识图谱实例
    return knowledge_graph_inst



async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    """
    知识图谱查询主函数，根据用户查询返回相关信息
    
    参数:
        query: str - 查询语句
        knowledge_graph_inst: BaseGraphStorage - 知识图谱存储实例
        entities_vdb: BaseVectorStorage - 实体向量数据库实例
        relationships_vdb: BaseVectorStorage - 关系向量数据库实例
        text_chunks_db: BaseKVStorage[TextChunkSchema] - 文本块键值存储
        query_param: QueryParam - 查询参数
        global_config: dict - 全局配置字典
        hashing_kv: BaseKVStorage - 用于缓存的键值存储，默认为None
    
    返回:
        str - 查询结果
    """
    # 调用LLM模型的函数
    use_model_func = global_config["llm_model_func"]
    # 构建查询缓存键，用于缓存查询结果
    args_hash = compute_args_hash(query_param.mode, query)
    # 检查缓存是否命中
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )
    if cached_response is not None:
        logger.info(f"Query cache hit for: {query}")  # 记录缓存命中
        return cached_response

    # 提取查询关键词
    example_number = global_config["addon_params"].get("example_number", None)  # 获取示例数量设置
    # 根据设置选择lightRAG的高低级关键词提取示例
    if example_number and example_number < len(PROMPTS["keywords_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["keywords_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    # 获取语言设置
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    # 检查查询模式是否有效
    if query_param.mode not in ["hybrid", "local", "global"]:
        logger.warning(
            f"Unknown query mode: {query_param.mode}, using 'hybrid' mode"
        )
        query_param.mode = "hybrid"  # 默认使用混合模式
    elif query_param.mode not in ["hybrid", "local"]:
        logger.error(f"Unknown mode {query_param.mode} in kg_query")
        return PROMPTS["fail_response"]


    # lightRAG的关键词提取提示模板
    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query, examples=examples, language=language)
    # 调用LLM提取关键词
    result = await use_model_func(kw_prompt, keyword_extraction=True)
    logger.info("kw_prompt result:")
    # print("result",result)
    # 解析关键词JSON响应
    try:
       
        match = re.search(r"\{.*\}", result, re.DOTALL)
        # print("match", match)
        if match:
            result = match.group(0)
            if result.startswith("{{") and result.endswith("}}"):
                result = result[1:-1]
            keywords_data = json.loads(result)

            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
        else:
            logger.error("No JSON-like structure found in the result.")
            return PROMPTS["fail_response"]


    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e} {result}")
        return PROMPTS["fail_response"]


    # 检查关键词是否为空
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return PROMPTS["fail_response"]
    if ll_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("low_level_keywords is empty")
        return PROMPTS["fail_response"]
    else:
        ll_keywords = ", ".join(ll_keywords)
    if hl_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("high_level_keywords is empty")
        return PROMPTS["fail_response"]
    else:
        hl_keywords = ", ".join(hl_keywords)


    # 关键词列表，包含低级别和高级别关键词
    keywords = [ll_keywords, hl_keywords]
    # 构建查询上下文，获取与查询相关的知识图谱信息
    context= await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
    )

    

    # 判断是否只需要返回检索到的上下文信息
    if query_param.only_need_context:
        # 如果只需要上下文，则直接返回检索到的上下文内容
        return context
    
    # 检查上下文是否为空
    if context is None:
        # 如果上下文为空，返回预设的失败响应提示
        return PROMPTS["fail_response"]
    
    # 获取RAG响应的系统提示模板
    sys_prompt_temp = PROMPTS["rag_response"]
    
    # 格式化系统提示，将检索到的上下文和响应类型填入模板
    # context_data: 检索到的相关文本上下文
    # response_type: 期望的响应类型（可能是"总结"、"问答"等）
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    
    # 判断是否只需要返回格式化后的提示内容
    if query_param.only_need_prompt:
        # 如果只需要提示，则直接返回格式化后的系统提示
        return sys_prompt
    
    # 调用模型函数生成回答
    # use_model_func: 异步模型调用函数，可能是各种LLM的封装
    # query: 用户的查询问题
    # system_prompt: 包含上下文的系统提示
    # stream: 是否启用流式输出
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    
    # 处理模型响应，清理不需要的内容
    # 只有当响应是字符串类型且长度大于系统提示长度时才进行清理
    if isinstance(response, str) and len(response) > len(sys_prompt):
        # 移除系统提示、用户、模型标记等不需要的内容
        response = (
            response.replace(sys_prompt, "")  # 移除系统提示
            .replace("user", "")  # 移除user标记
            .replace("model", "")  # 移除model标记
            .replace(query, "")  # 移除原始查询
            .replace("<system>", "")  # 移除<system>标签
            .replace("</system>", "")  # 移除</system>标签
            .strip()  # 去除首尾空白字符
        )


    # 将查询结果保存到缓存中，以便后续相同查询快速响应
    # hashing_kv: 键值存储实例，用于缓存管理
    # CacheData: 缓存数据结构，包含查询相关的所有信息
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,  # 查询参数的哈希值，作为缓存键
            content=response,  # 模型生成的回答内容
            prompt=query,  # 用户原始查询
            quantized=quantized,  # 向量量化状态
            min_val=min_val,  # 向量最小值
            max_val=max_val,  # 向量最大值
            mode=query_param.mode,  # 查询模式（如hybrid、local等）
        ),
    )
    
    # 返回模型生成的最终回答
    return response


async def _build_query_context(
    query: list,  # 查询关键词列表 [ll_keywords, hl_keywords]
    knowledge_graph_inst: BaseGraphStorage,  # 知识图谱存储实例
    entities_vdb: BaseVectorStorage,  # 实体向量数据库
    relationships_vdb: BaseVectorStorage,  # 关系向量数据库
    text_chunks_db: BaseKVStorage[TextChunkSchema],  # 文本块存储
    query_param: QueryParam,  # 查询参数
):
    """
    构建查询上下文，根据查询模式和关键词获取相关的实体、关系和文本信息
    
    支持三种查询模式：
    1. hybrid: 混合使用高级别和低级别信息
    2. local: 仅使用低级别信息
    3. global: 仅使用高级别信息
    """
    # 初始化各类型上下文变量
    ll_entities_context, ll_relations_context, ll_text_units_context = "", "", ""  # 低级别上下文（实体、关系、文本单元）
    hl_entities_context, hl_relations_context, hl_text_units_context = "", "", ""  # 高级别上下文（实体、关系、文本单元）

    # 提取低级别和高级别关键词
    ll_kewwords, hl_keywrds = query[0], query[1]
    mode = query_param.mode  # 获取当前查询模式
    
    # 根据模式处理不同级别的上下文
    if mode in ["hybrid","local","global"]:
        # 检查关键词是否都为空
        if ll_kewwords == "" and hl_keywrds == "":
            warnings.warn(
                "低级别和全局级别上下文为空。返回空的实体/关系/源文本"
            )
        else:
            # 处理低级别关键词
            if ll_kewwords == "":
                ll_entities_context, ll_relations_context, ll_text_units_context = (
                    "",
                    "",
                    "",
                )
                warnings.warn(
                    "低级别上下文为空。返回空的低级别实体/关系/源文本"
                )
                mode = "global"  # 切换到全局模式
            else:
                # 获取低级别节点数据（实体、关系、文本单元）
                (
                    ll_entities_context,
                    ll_relations_context,
                    ll_text_units_context,
                ) = await _get_node_data(
                    ll_kewwords,
                    knowledge_graph_inst,
                    entities_vdb,
                    text_chunks_db,
                    query_param,
                )
            # 处理高级别关键词
            if hl_keywrds == "":
                hl_entities_context, hl_relations_context, hl_text_units_context = (
                    "",
                    "",
                    "",
                )
                warnings.warn(
                    "高级别上下文为空。返回空的高级别实体/关系/源文本"
                )
                mode = "local"  # 切换到本地模式
            else:
                # 获取高级别边数据（关系相关实体、关系详情、文本单元）
                (
                    hl_entities_context,
                    hl_relations_context,
                    hl_text_units_context,
                ) = await _get_edge_data(
                    hl_keywrds,
                    knowledge_graph_inst,
                    relationships_vdb,
                    text_chunks_db,
                    query_param,
                )

    # 根据不同的查询模式构建上下文
    if mode == "hybrid":
        # 混合模式：合并高级别和低级别上下文
        entities_context, relations_context, text_units_context = combine_contexts(
            [hl_entities_context, hl_relations_context],  # 高级别信息
            [ll_entities_context, ll_relations_context],  # 低级别信息
            [hl_text_units_context, ll_text_units_context],  # 文本单元信息
        )
        # 构建混合模式的上下文输出
        context=f"""
-----global-information-----  # 全局信息部分
-----high-level entity information-----  # 高级别实体信息
```csv
{hl_entities_context}  # 高级别实体CSV数据
```
-----high-level relationship information-----  # 高级别关系信息
```csv
{hl_relations_context}  # 高级别关系CSV数据
```
-----Sources-----  # 来源文本信息
```csv
{text_units_context}  # 合并后的文本单元CSV数据
```
-----local-information-----  # 本地信息部分
-----low-level entity information-----  # 低级别实体信息
```csv
{ll_entities_context}  # 低级别实体CSV数据
```
-----low-level relationship information-----  # 低级别关系信息
```csv
{ll_relations_context}  # 低级别关系CSV数据
```
"""
    elif mode == "local":
        # 本地模式：仅包含低级别上下文
        context=f"""
-----local-information-----  # 本地信息部分
-----low-level entity information-----  # 低级别实体信息
```csv
{ll_entities_context}  # 低级别实体CSV数据
```
-----Sources-----  # 来源文本信息
```csv
{ll_text_units_context}  # 低级别文本单元CSV数据
```
-----low-level relationship information-----  # 低级别关系信息
```csv
{ll_relations_context}  # 低级别关系CSV数据
```
"""
    elif mode == "global":
        # 全局模式：仅包含高级别上下文
        context=f"""
-----global-information-----  # 全局信息部分
-----high-level entity information-----  # 高级别实体信息
```csv
{hl_entities_context}  # 高级别实体CSV数据
```
-----Sources-----  # 来源文本信息
```csv
{hl_text_units_context}  # 高级别文本单元CSV数据
```
-----low-level relationship information-----  # 高级别关系信息
```csv
{hl_relations_context}  # 高级别关系CSV数据
```
"""

    # print(context)  # 调试输出
    return context  # 返回构建好的上下文

async def _get_node_data(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    """
    获取与查询相关的节点数据，包括实体信息、关系和文本单元
    
    参数:
        query: 查询字符串
        knowledge_graph_inst: 知识图谱存储实例
        entities_vdb: 实体向量数据库
        text_chunks_db: 文本块键值存储
        query_param: 查询参数对象
    
    返回:
        包含实体、关系和文本单元上下文的元组
    """
    # 在实体向量数据库中查询与关键词相关的实体，底层实现不用管了
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    # 如果没有查询结果，返回空的上下文
    if not len(results):
        return "", "", ""

    # 并行获取每个实体的节点详细信息
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    # 检查是否有缺失的节点，如果有则记录警告
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    # 并行获取每个实体的度（连接数）信息
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    # 合并节点数据、实体名称和度信息，过滤掉空节点
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  
    # 查找与这些实体最相关的文本单元
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )

    # 发现给定的实体间的高得分的路径，并用自然语言描述这些路径
    use_relations= await _find_most_related_edges_from_entities3(
        node_datas, query_param, knowledge_graph_inst
    )

    # 记录本地查询使用的实体、关系和文本单元数量
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )

    # 构建实体信息的CSV格式
    entites_section_list = [["id", "entity", "type", "description", "rank"]]  # CSV表头
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],  # 实体名称
                n.get("entity_type", "UNKNOWN"),  # 实体类型，默认UNKNOWN
                n.get("description", "UNKNOWN"),  # 实体描述，默认UNKNOWN
                n["rank"],  # 实体的度（重要性排名）
            ]
        )
    # 将列表转换为CSV字符串
    entities_context = list_of_list_to_csv(entites_section_list)

    # 构建关系信息的CSV格式
    relations_section_list=[["id","context"]]  # CSV表头
    for i,e in enumerate(use_relations):
        relations_section_list.append([i,e])
    # 将列表转换为CSV字符串
    relations_context=list_of_list_to_csv(relations_section_list)

    # 构建文本单元的CSV格式
    text_units_section_list = [["id", "content"]]  # CSV表头
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])  # 添加每个文本块的内容
    # 将列表转换为CSV字符串
    text_units_context = list_of_list_to_csv(text_units_section_list)
    
    # 返回三种上下文信息
    return entities_context,relations_context,text_units_context


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    """
    查找与实体最相关的文本单元
    由于抄的lightRAG所以文本块和节点关系是分开存储到，需要通过sourceid去找对应的文本块
    参数:
        node_datas: list[dict] - 实体节点数据列表
        query_param: QueryParam - 查询参数
        text_chunks_db: BaseKVStorage[TextChunkSchema] - 文本块存储
        knowledge_graph_inst: BaseGraphStorage - 知识图谱存储实例
        
    返回:
        list[TextChunkSchema] - 相关文本单元列表
    """
    # 从每个节点数据中提取源文本ID列表，按分隔符分割
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    
    # 并行获取每个节点的边（关系）
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    
    # 收集所有一跳邻居节点
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    # 转换为列表以便批量处理
    all_one_hop_nodes = list(all_one_hop_nodes)
    
    # 并行获取所有一跳邻居节点的数据
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # 创建一跳邻居节点的文本单元查找表
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  
    }

    # 初始化文本单元查找表
    all_text_units_lookup = {}
    
    # 遍历每个实体的文本单元和边
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        # 处理每个文本单元ID
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                # 如果文本单元尚未添加，则获取其数据并初始化
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),  # 获取文本块数据
                    "order": index,  # 记录实体在原始列表中的顺序
                    "relation_counts": 0,  # 统计与其他实体的关联次数
                }

            # 计算文本单元与其他实体的关联度
            if this_edges:
                for e in this_edges:
                    # 如果边的目标节点在一跳邻居中，且当前文本单元也属于该邻居，则增加关联计数
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        all_text_units_lookup[c_id]["relation_counts"] += 1

    # 过滤有效文本单元
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    # 如果没有找到有效文本单元，记录警告并返回空列表
    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    # 按实体顺序和关联计数排序（先按实体顺序，同一实体内按关联计数降序）
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    # 按token大小截断文本单元列表，确保不超过最大token限制
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],  # 以文本内容计算token
        max_token_size=query_param.max_token_for_text_unit,  # 使用查询参数中的最大token限制
    )

    # 提取文本单元数据并返回
    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units

async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    """
    根据关键词获取边数据（关系数据）
    
    参数:
        keywords: str - 查询关键词
        knowledge_graph_inst: BaseGraphStorage - 知识图谱存储实例
        relationships_vdb: BaseVectorStorage - 关系向量数据库实例
        text_chunks_db: BaseKVStorage[TextChunkSchema] - 文本块存储实例
        query_param: QueryParam - 查询参数
    
    返回:
        tuple[str, str, str] - (实体上下文, 关系上下文, 文本单元上下文)
    """
    # 在关系向量数据库中查询与关键词相关的关系
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    # 如果没有找到相关关系，返回空字符串
    if not len(results):
        return "", "", ""

    # 并发获取所有关系的详细信息
    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )

    # 检查是否有边数据缺失
    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    
    # 并发获取所有边的度数（连接数）
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
    )
    
    # 合并边数据、查询结果和度数信息，并过滤掉无效数据
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    
    # 按排名和权重降序排序，优先选择重要关系
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    
    # 根据最大token数截断边数据列表，确保不超过上下文长度限制
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    # 获取与关系相关的实体信息
    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )
    
    # 获取与关系相关的文本单元信息
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    
    # 记录查询使用的实体、关系和文本单元数量
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )

    # 准备关系信息的CSV格式数据
    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    # 准备实体信息的CSV格式数据
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    # 准备文本单元信息的CSV格式数据
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    
    # 返回实体、关系和文本单元的上下文信息
    return entities_context, relations_context, text_units_context


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                chunk_data = await text_chunks_db.get_by_id(c_id)

                if chunk_data is not None and "content" in chunk_data:
                    all_text_units_lookup[c_id] = {
                        "data": chunk_data,
                        "order": index,
                    }

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])


    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


def combine_contexts(entities, relationships, sources):
    """
    合并高级别和低级别上下文信息
    
    参数:
        entities: list - 包含高级别和低级别实体上下文的列表
        relationships: list - 包含高级别和低级别关系上下文的列表
        sources: list - 包含高级别和低级别文本源上下文的列表
    
    返回:
        tuple[str, str, str] - 合并后的实体、关系和文本源上下文
    """
    # 提取高级别(hl)和低级别(ll)实体上下文，entities[0]为高级别，entities[1]为低级别
    hl_entities, ll_entities = entities[0], entities[1]
    
    # 提取高级别(hl)和低级别(ll)关系上下文，relationships[0]为高级别，relationships[1]为低级别
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    
    # 提取高级别(hl)和低级别(ll)文本源上下文，sources[0]为高级别，sources[1]为低级别
    hl_sources, ll_sources = sources[0], sources[1]

    # 调用工具函数合并实体上下文，将高级别和低级别实体信息进行组合
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # 调用工具函数合并关系上下文，将高级别和低级别关系信息进行组合
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    # 调用工具函数合并文本源上下文，将高级别和低级别文本信息进行组合
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    # 返回合并后的三种上下文信息：实体上下文、关系上下文和文本源上下文
    return combined_entities, combined_relationships, combined_sources


import networkx as nx
from collections import defaultdict
async def find_paths_and_edges_with_stats(graph, target_nodes):
    """
    PathRAG自创的查找图中节点间的路径和边，并计算统计信息
    
    参数:
        graph: 图结构对象，包含节点和边信息
        target_nodes: 目标节点列表，需要查找路径的节点集合
    
    返回:
        tuple: 包含路径统计信息的多个数据结构
            - result: 路径结果字典
            - path_stats: 路径统计信息
            - one_hop_paths: 单跳路径列表
            - two_hop_paths: 两跳路径列表
            - three_hop_paths: 三跳路径列表
    """
    """
    在图中查找目标节点之间的路径和边，并统计路径信息
    
    参数:
        graph: 图数据结构，用于路径查找
        target_nodes: 目标节点列表，需要在这些节点间查找路径
    
    返回:
        dict: 包含路径和边信息的字典
        dict: 路径统计信息
        list: 1跳路径列表
        list: 2跳路径列表
        list: 3跳路径列表
    """

    # 初始化结果字典，默认值为包含paths列表和edges集合的字典
    # 使用(node1, node2)作为键，存储从node1到node2的所有路径和边
    result = defaultdict(lambda: {"paths": [], "edges": set()})
    # 初始化路径统计字典，记录不同跳数路径的数量
    path_stats = {"1-hop": 0, "2-hop": 0, "3-hop": 0}   
    # 初始化不同跳数路径的列表，用于分别存储
    one_hop_paths = []  # 1跳路径（直接相连的节点）
    two_hop_paths = []  # 2跳路径（通过一个中间节点相连）
    three_hop_paths = []  # 3跳路径（通过两个中间节点相连）

    async def dfs(current, target, path, depth):
        """
        深度优先搜索子函数，用于查找从起始节点到目标节点的路径
        
        参数:
            current: 当前节点
            target: 目标节点
            path: 当前路径列表
            depth: 当前搜索深度
        """
        # 深度限制：不搜索超过3跳的路径
        if depth > 3: 
            return
        
        # 找到目标节点，记录路径和边信息
        if current == target: 
            # 将找到的路径添加到结果字典中
            result[(path[0], target)]["paths"].append(list(path))
            
            # 提取路径中的所有边（按顺序排序以避免重复）
            for u, v in zip(path[:-1], path[1:]):
                # 使用排序后的节点对作为边的唯一标识，避免(u,v)和(v,u)被视为不同边
                result[(path[0], target)]["edges"].add(tuple(sorted((u, v))))
            
            # 根据路径长度更新统计信息和对应列表
            if depth == 1:  # 直接相连的节点（1跳）
                path_stats["1-hop"] += 1
                one_hop_paths.append(list(path))
            elif depth == 2:  # 通过一个中间节点（2跳）
                path_stats["2-hop"] += 1
                two_hop_paths.append(list(path))
            elif depth == 3:  # 通过两个中间节点（3跳）
                path_stats["3-hop"] += 1
                three_hop_paths.append(list(path))
            return
        
        # 获取当前节点的所有邻居
        neighbors = graph.neighbors(current) 
        
        # 递归搜索每个未在当前路径中的邻居
        for neighbor in neighbors:
            if neighbor not in path:  # 避免循环
                # 递归调用DFS，将当前邻居加入路径，深度+1
                await dfs(neighbor, target, path + [neighbor], depth + 1)

    # 遍历所有目标节点对，查找它们之间的路径
    for node1 in target_nodes:
        for node2 in target_nodes:
            if node1 != node2:  # 避免查找节点到自身的路径
                # 从node1开始DFS搜索到node2的路径，初始路径包含node1，初始深度为0
                await dfs(node1, node2, [node1], 0)

    # 将结果中的边集合转换为列表，便于后续处理
    for key in result:
        result[key]["edges"] = list(result[key]["edges"])

    # 返回结果：
    # 1. 转换为普通字典的路径和边信息
    # 2. 路径统计信息
    # 3-5. 不同跳数的路径列表
    return dict(result), path_stats , one_hop_paths, two_hop_paths, three_hop_paths
def bfs_weighted_paths(G, path, source, target, threshold, alpha):
    """
    使用带权重的广度优先搜索对已找到的路径计算边权重。
    
    参数:
        G: 图结构对象
        path: 当前路径列表
        source: 源节点
        target: 目标节点
        threshold: 权重阈值，超过此值的边不会被加入路径
        alpha: 权重衰减因子，用于计算路径的累积权重
    
    返回:
        list: 找到的路径列表，每条路径包含节点和相应的权重
    """
    # 初始化结果列表，用于存储找到的路径
    results = [] 
    # 使用defaultdict存储边权重，默认值为0
    edge_weights = defaultdict(float)  
    # 源节点
    node = source
    # 初始化节点跟随关系字典，用于存储每个节点可以到达的下一节点集合
    follow_dict = {}

    # 遍历所有输入路径，构建节点间的跟随关系
    for p in path:
        # 遍历路径中的每一对相邻节点
        for i in range(len(p) - 1):  
            current = p[i]  # 当前节点
            next_num = p[i + 1]  # 下一节点

            # 将下一节点添加到当前节点的跟随集合中
            if current in follow_dict:
                follow_dict[current].add(next_num)  # 如果当前节点已存在，添加到现有集合
            else:
                follow_dict[current] = {next_num}  # 如果当前节点不存在，创建新集合

    # 开始广度优先搜索，遍历源节点的所有邻居
    for neighbor in follow_dict[node]:
        # 计算从源节点到邻居节点的初始权重（均匀分配）
        edge_weights[(node, neighbor)] += 1/len(follow_dict[node])

        # 检查是否到达目标节点
        if neighbor == target:
            results.append(([node, neighbor]))  # 添加长度为2的路径
            continue
        
        # 如果边的权重超过阈值，继续搜索
        if edge_weights[(node, neighbor)] > threshold:
            # 遍历邻居节点的邻居（第三跳）
            for second_neighbor in follow_dict[neighbor]:
                # 计算权重，考虑衰减因子alpha和邻居节点的出度
                weight = edge_weights[(node, neighbor)] * alpha / len(follow_dict[neighbor])
                edge_weights[(neighbor, second_neighbor)] += weight

                # 检查是否到达目标节点
                if second_neighbor == target:
                    results.append(([node, neighbor, second_neighbor]))  # 添加长度为3的路径
                    continue

                # 如果边的权重超过阈值，继续搜索
                if edge_weights[(neighbor, second_neighbor)] > threshold:    
                    # 遍历第三跳节点的邻居（第四跳）
                    for third_neighbor in follow_dict[second_neighbor]:
                        # 计算权重，继续衰减
                        weight = edge_weights[(neighbor, second_neighbor)] * alpha / len(follow_dict[second_neighbor]) 
                        edge_weights[(second_neighbor, third_neighbor)] += weight

                        # 检查是否到达目标节点
                        if third_neighbor == target :
                            results.append(([node, neighbor, second_neighbor, third_neighbor]))  # 添加长度为4的路径
                            continue
    # 初始化路径权重列表
    path_weights = []
    # 遍历每个原始路径，计算其平均权重
    for p in path:
        path_weight = 0  # 初始化路径总权重
        # 遍历路径中的每条边，累加边权重
        for i in range(len(p) - 1):
            edge = (p[i], p[i + 1])  # 构建边的元组表示
            path_weight += edge_weights.get(edge, 0)  # 获取边权重，如果不存在则为0
        # 计算路径的平均权重（总权重除以边数）
        path_weights.append(path_weight/(len(p)-1))

    # 将路径和对应的平均权重组合成元组列表
    combined = [(p, w) for p, w in zip(path, path_weights)]

    # 返回路径-权重组合列表
    return combined
async def _find_most_related_edges_from_entities3(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):  
    """
    发现给定的实体间的高得分的路径，并用自然语言描述这些路径
    
    参数:
        node_datas: 实体节点数据列表
        query_param: 查询参数，包含token大小限制等配置
        knowledge_graph_inst: 知识图谱存储实例
    
    返回:
        list: 按相关性排序的关系列表
    """

    G = nx.Graph()
    edges = await knowledge_graph_inst.edges()
    nodes = await knowledge_graph_inst.nodes()

    for u, v in edges:
        G.add_edge(u, v) 
    G.add_nodes_from(nodes)
    # 提取要查询的源实体名称
    source_nodes = [dp["entity_name"] for dp in node_datas]  # 从节点数据中提取实体名称
    
    # 查找源实体之间的路径和边，统计不同跳数的路径
    result, path_stats, one_hop_paths, two_hop_paths, three_hop_paths = await find_paths_and_edges_with_stats(G, source_nodes)


    # 设置BFS权重路径算法的参数
    threshold = 0.3  # 边权重阈值，超过该值才继续扩展路径
    alpha = 0.8  # 权重衰减系数，每跳权重乘以该系数
    all_results = []  # 存储所有实体对的路径权重结果
    
    # 遍历所有实体对，计算每对实体间的路径权重
    for node1 in source_nodes:  # 遍历每个源实体作为起点
        for node2 in source_nodes:  # 遍历每个源实体作为终点
            if node1 != node2:  # 跳过自身到自身的情况
                if (node1, node2) in result:  # 如果这对实体之间存在路径
                    # 创建子图用于分析
                    sub_G = nx.Graph()
                    paths = result[(node1,node2)]['paths']  # 获取这对实体间的所有路径
                    edges = result[(node1,node2)]['edges']  # 获取这对实体间涉及的所有边
                    sub_G.add_edges_from(edges)  # 将边添加到子图
                    # 使用带权重的BFS计算路径平均权重
                    results = bfs_weighted_paths(G, paths, node1, node2, threshold, alpha)
                    all_results += results  # 合并结果
    # 按路径权重降序排序（含有重复的路径）
    all_results = sorted(all_results, key=lambda x: x[1], reverse=True)
    
    # 边去重:因为以上把不同路径、不同遍历方向产生的同一条无向边都收集了
    seen = set()  # 记录已保留的无向边（用排序后的端点表示，避免(A,B)/(B,A)重复）
    result_edge = []  # 去掉同一无向边的重复后得到的路径，保持 all_results 的降序顺序
    for edge, weight in all_results:
        sorted_edge = tuple(sorted(edge))  # 将边端点排序，构造无向唯一标识
        if sorted_edge not in seen:
            seen.add(sorted_edge)  # 标记该无向边已处理
            result_edge.append((edge, weight))  # 仍按权重顺序保存原路径及其权重

    
    
    length_1 = int(len(one_hop_paths)/2)  
    length_2 = int(len(two_hop_paths)/2)  
    length_3 = int(len(three_hop_paths)/2) 
    
    results = []  # 3种长度的路径自取前一半拼在一起的原始路径集合，用来统计候选数量，不含权重、也没排序。
    # 添加一跳路径（如果存在）
    if one_hop_paths != []:
        results = one_hop_paths[0:length_1]
    # 添加两跳路径（如果存在）
    if two_hop_paths != []:
        results = results + two_hop_paths[0:length_2]
    # 添加三跳路径（如果存在）
    if three_hop_paths != []:
        results = results + three_hop_paths[0:length_3]

    # 依据前面截取的路径数量，确定最终输出的上限
    length = len(results)  # 当前挑选出来的路径数量
    total_edges = 15  # 默认最多展示15条（路径/边）
    if length < total_edges:  # 如果候选路径不足15条，则只取现有数量
        total_edges = length
    
    sort_result = []  # 存储最终保留的(路径, 平均权重)，从 result_edge 里取前 total_edges 条（默认 15 条）后的结果，继承 result_edge 的降序顺序

    if result_edge:
        # `result_edge` 已按权重降序排列，取前 total_edges 条即可
        if len(result_edge) > total_edges:
            sort_result = result_edge[0:total_edges]
        else:
            sort_result = result_edge
    
    # 仅提取路径节点序列，后续用于构造自然语言描述
    final_result = []
    for edge, weight in sort_result:
        final_result.append(edge)

    # 构建关系描述列表
    relationship = []  # 存储格式化的关系描述

    for path in final_result:
        if len(path) == 4:
            s_name,b1_name,b2_name,t_name = path[0],path[1],path[2],path[3]
            edge0 = await knowledge_graph_inst.get_edge(path[0], path[1]) or await knowledge_graph_inst.get_edge(path[1], path[0])
            edge1 = await knowledge_graph_inst.get_edge(path[1],path[2]) or await knowledge_graph_inst.get_edge(path[2], path[1])
            edge2 = await knowledge_graph_inst.get_edge(path[2],path[3]) or await knowledge_graph_inst.get_edge(path[3], path[2])
            if edge0==None or edge1==None or edge2==None:
                print(path,"边丢失")
                if edge0==None:
                    print("edge0丢失")
                if edge1==None:
                    print("edge1丢失")
                if edge2==None:
                    print("edge2丢失")
                continue
            e1 = "through edge ("+edge0["keywords"]+") to connect to "+s_name+" and "+b1_name+"."
            e2 = "through edge ("+edge1["keywords"]+") to connect to "+b1_name+" and "+b2_name+"."
            e3 = "through edge ("+edge2["keywords"]+") to connect to "+b2_name+" and "+t_name+"."
            s = await knowledge_graph_inst.get_node(s_name)
            s = "The entity "+s_name+" is a "+s["entity_type"]+" with the description("+s["description"]+")"
            b1 = await knowledge_graph_inst.get_node(b1_name)
            b1 = "The entity "+b1_name+" is a "+b1["entity_type"]+" with the description("+b1["description"]+")"
            b2 = await knowledge_graph_inst.get_node(b2_name)
            b2 = "The entity "+b2_name+" is a "+b2["entity_type"]+" with the description("+b2["description"]+")"
            t = await knowledge_graph_inst.get_node(t_name)
            t = "The entity "+t_name+" is a "+t["entity_type"]+" with the description("+t["description"]+")"
            relationship.append([s+e1+b1+"and"+b1+e2+b2+"and"+b2+e3+t])
        elif len(path) == 3:
            s_name,b_name,t_name = path[0],path[1],path[2]
            edge0 = await knowledge_graph_inst.get_edge(path[0], path[1]) or await knowledge_graph_inst.get_edge(path[1], path[0])
            edge1 = await knowledge_graph_inst.get_edge(path[1],path[2]) or await knowledge_graph_inst.get_edge(path[2], path[1])
            if edge0==None or edge1==None:
                print(path,"边丢失")
                continue
            e1 = "through edge("+edge0["keywords"]+") to connect to "+s_name+" and "+b_name+"."
            e2 = "through edge("+edge1["keywords"]+") to connect to "+b_name+" and "+t_name+"."
            s = await knowledge_graph_inst.get_node(s_name)
            s = "The entity "+s_name+" is a "+s["entity_type"]+" with the description("+s["description"]+")"
            b = await knowledge_graph_inst.get_node(b_name)
            b = "The entity "+b_name+" is a "+b["entity_type"]+" with the description("+b["description"]+")"
            t = await knowledge_graph_inst.get_node(t_name)
            t = "The entity "+t_name+" is a "+t["entity_type"]+" with the description("+t["description"]+")"
            relationship.append([s+e1+b+"and"+b+e2+t])
        elif len(path) == 2:  # 处理长度为2的路径，即直接相连的实体
            s_name,t_name = path[0],path[1]  # 解包源实体和目标实体名称
            # 尝试获取边数据，先尝试正向边，再尝试反向边
            edge0 = await knowledge_graph_inst.get_edge(path[0], path[1]) or await knowledge_graph_inst.get_edge(path[1], path[0])
            if edge0==None:
                print(path,"边丢失")  # 边不存在时打印警告
                continue
            # 构建关系描述文本
            e = "through edge("+edge0["keywords"]+") to connect to "+s_name+" and "+t_name+"."
            # 获取源实体信息
            s = await knowledge_graph_inst.get_node(s_name)
            s = "The entity "+s_name+" is a "+s["entity_type"]+" with the description("+s["description"]+")"
            # 获取目标实体信息
            t = await knowledge_graph_inst.get_node(t_name)
            t = "The entity "+t_name+" is a "+t["entity_type"]+" with the description("+t["description"]+")"
            # 将构建的关系信息添加到结果列表
            relationship.append([s+e+t])

    # 按token大小截断关系列表，确保不超过最大token限制
    relationship = truncate_list_by_token_size(
          relationship, 
          key=lambda x: x[0],  # 使用列表中的第一个元素（关系描述）计算token数
          max_token_size=query_param.max_token_for_local_context,  # 本地上下文的最大token限制
    )

    # 反转关系列表顺序
    reversed_relationship = relationship[::-1]
    # 返回反转后的关系列表
    return reversed_relationship
