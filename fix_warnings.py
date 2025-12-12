#!/usr/bin/env python3
"""
修复PathRAG中的警告问题
1. attention_mask 警告
2. pad_token_id 警告
"""

import os

def fix_embedding_attention_mask():
    """修复embedding函数中的attention_mask警告"""
    file_path = "/data/gty/workspace/PathRAG/PathRAG/llm.py"

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复hf_embedding函数
    old_hf_embedding = '''async def hf_embedding(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    device = next(embed_model.parameters()).device
    input_ids = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(device)
    with torch.no_grad():
        outputs = embed_model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()'''

    new_hf_embedding = '''async def hf_embedding(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    device = next(embed_model.parameters()).device
    # 添加attention_mask以避免警告
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    with torch.no_grad():
        outputs = embed_model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()'''

    # 修复ms_embedding函数
    old_ms_embedding = '''async def ms_embedding(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    device = next(embed_model.parameters()).device
    input_ids = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(device)
    with torch.no_grad():
        outputs = embed_model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()'''

    new_ms_embedding = '''async def ms_embedding(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    device = next(embed_model.parameters()).device
    # 添加attention_mask以避免警告
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    with torch.no_grad():
        outputs = embed_model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()'''

    # 修复local_embedding函数
    old_local_embedding = '''async def local_embedding(texts: list[str], tokenizer=None, embed_model=None) -> np.ndarray:
    if tokenizer is None or embed_model is None:
        raise ValueError("Tokenizer and model must be provided")
    device = next(embed_model.parameters()).device
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)
    with torch.no_grad():
        outputs = embed_model(encoded)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()'''

    new_local_embedding = '''async def local_embedding(texts: list[str], tokenizer=None, embed_model=None) -> np.ndarray:
    if tokenizer is None or embed_model is None:
        raise ValueError("Tokenizer and model must be provided")
    device = next(embed_model.parameters()).device
    # 添加attention_mask以避免警告
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    with torch.no_grad():
        outputs = embed_model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()'''

    # 应用修复
    content = content.replace(old_hf_embedding, new_hf_embedding)
    content = content.replace(old_ms_embedding, new_ms_embedding)
    content = content.replace(old_local_embedding, new_local_embedding)

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ 修复了embedding函数中的attention_mask警告")


def fix_pad_token_id_warning():
    """修复pad_token_id警告"""
    file_path = "/data/gty/workspace/PathRAG/PathRAG/llm.py"

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 查找并修复pad_token_id相关的代码
    # 这个警告通常出现在模型生成时，需要在调用生成函数前设置pad_token_id

    # 首先查找所有模型生成函数
    fixes = [
        # 修复local_model_complete函数
        ('''
    output = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )''', '''
    # 设置pad_token_id以避免警告
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    output = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )'''),

        # 修复vllm_model相关（vLLM通常不需要这个设置）
        # 但如果有其他使用transformers直接调用的地方，也需要修复
    ]

    for old_code, new_code in fixes:
        if old_code in content:
            content = content.replace(old_code, new_code)
            print("✅ 修复了pad_token_id警告")

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    print("开始修复PathRAG警告问题...")
    print("=" * 60)

    # 修复attention_mask警告
    fix_embedding_attention_mask()

    # 修复pad_token_id警告
    fix_pad_token_id_warning()

    print("=" * 60)
    print("✅ 所有警告修复完成！")
    print("\n说明：")
    print("1. attention_mask警告已通过在embedding函数中添加attention_mask参数解决")
    print("2. pad_token_id警告已通过在生成前设置tokenizer.pad_token解决")


if __name__ == "__main__":
    main()