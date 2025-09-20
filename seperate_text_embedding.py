from datasets import Dataset
from transformers import AutoTokenizer, LongformerModel, BertModel, BertTokenizer
import torch
import pandas as pd
import os
import multiprocessing as mp
import json


def text_embedding(text, max_length=4096, overlap=512):
    """
    对长文本进行分块处理，并使用 Longformer 模型进行编码。

    Args:
        text (str): 待编码的长文本。
        max_length (int): 每个块的最大长度。
        overlap (int): 块之间的重叠长度。

    Returns:
        torch.Tensor: 整个长文本的单一向量表示 (shape: [1, hidden_size])。
    """
    # 假设你已经定义了 tokenizer 和 model
    # tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    # model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

    # 选择 Longformer 模型
    model_name = 'allenai/longformer-base-4096'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LongformerModel.from_pretrained(model_name)

    # 将文本分词
    tokens = tokenizer(text, truncation=False, return_tensors="pt")
    input_ids = tokens['input_ids'].squeeze(0)
    
    # 获取原始序列的长度
    seq_len = input_ids.size(0)

    # 存储分块后的 input_ids 和 attention_mask
    chunks_with_padding = []
    attention_masks = []
    
    for i in range(0, seq_len, max_length - overlap):
        # 提取当前块
        chunk = input_ids[i:i + max_length]
        
        # 确定当前块的有效长度（非填充部分的长度）
        valid_len = chunk.size(0)
        
        # 如果块太短（通常是最后一个块），则进行填充
        if valid_len < max_length:
            padding_len = max_length - valid_len
            padding_chunk = torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)
            chunk = torch.cat([chunk, padding_chunk], dim=0)
            
            # 创建 attention_mask
            mask = torch.cat([torch.ones(valid_len, dtype=torch.long), 
                              torch.zeros(padding_len, dtype=torch.long)], dim=0)
        else:
            # 如果块的长度正好等于 max_length，则所有部分都有效
            mask = torch.ones(max_length, dtype=torch.long)
            
        chunks_with_padding.append(chunk)
        attention_masks.append(mask)

    # 将列表转换为张量
    chunks = torch.stack(chunks_with_padding)
    masks = torch.stack(attention_masks)
    
    all_embeddings = []
    model.eval()  # 切换到评估模式

    with torch.no_grad():
        for i in range(chunks.size(0)):
            chunk = chunks[i].unsqueeze(0)
            mask = masks[i].unsqueeze(0)
            
            # 设置 global_attention_mask，通常只有 [CLS] token 被设置为 1
            # 注意：这里的 global_attention_mask 的形状必须与 input_ids 相同
            global_attention_mask = torch.zeros_like(mask)
            global_attention_mask[:, 0] = 1 # [CLS] token
            
            outputs = model(
                input_ids=chunk,
                attention_mask=mask,
                global_attention_mask=global_attention_mask
            )
            
            # 获取 [CLS] token 的输出作为该块的表示
            chunk_embedding = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(chunk_embedding)
    
    # 聚合所有块的编码向量，这里使用平均池化
    stack_embedding = torch.stack(all_embeddings, dim=1)
    final_embedding = torch.mean(stack_embedding, dim=1)
    
    return final_embedding


def get_seperate_embedding(folder, max_length=4096, overlap=512):
    folder_path = "/Users/lipeizhe/fintech/bloomberg_dataset/"

    if not folder.startswith('.'):
        os.makedirs(f"/Users/lipeizhe/fintech/embeddings/{folder}", exist_ok=True)
        
        print(f"Embedding for {folder}")

        file_list = os.listdir(os.path.join(folder_path, folder))
        file_list.sort()

        for file in file_list:
            if file.endswith('.txt') and not file.endswith('test.txt') and not file.endswith('timestamp.txt'):
                ec_name = file.split('.')[0]
                # if ec_name != 'AMD_21_07_28_0500':
                #     continue
                pt_file = f"/Users/lipeizhe/fintech/embeddings/{folder}/{ec_name}.pt"

                if os.path.exists(pt_file):
                    continue
                with open(os.path.join(folder_path, folder, file), 'r') as f:
                    text = f.read()
                    try:
                        pres_part = text.split('Presentation\n')[1].split('Questions And Answers\n')[0]
                    except IndexError:
                        print(f"Error in {ec_name}, no presentation part found.")
                        continue

                    try:
                        qna_part = text.split('Questions And Answers\n')[1]
                    except IndexError:
                        print(f"Error in {ec_name}, no Q&A part found.")
                        continue

                    if len(qna_part) < 50:
                        print(f"Error in {ec_name}, Q&A part is too short.")
                        continue

                    pres_embedding = text_embedding(pres_part, max_length, overlap)
                    qna_embedding = text_embedding(qna_part, max_length, overlap)

                    text_all_embedding = torch.cat([pres_embedding, qna_embedding], dim=0)
                    torch.save(text_all_embedding, pt_file)


def get_factor_embedding(pool_name):
    model_name = 'bert-base-uncased' 
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    with open(f"./{pool_name}.json", "r") as f:
        pool_dict = json.load(f)

    factor_list = []
    for role, factors in pool_dict.items():
        factor_list.extend(factors)

    encoded_inputs = tokenizer(
        factor_list,
        padding=True,          # 自动填充到批次中最长的句子
        truncation=True,       # 自动截断过长的句子
        return_tensors="pt"    # 返回 PyTorch tensors
    )

    with torch.no_grad():
        outputs = model(**encoded_inputs)
        factor_embedding = outputs.last_hidden_state.mean(dim=1)

    torch.save(factor_embedding, f"./{pool_name}_embedding.pt")


if __name__ == '__main__':
    pool_name = 'multi_factor_pool_0_2'
    get_factor_embedding(pool_name)
    # folder_path = "/Users/lipeizhe/fintech/bloomberg_dataset/"

    # pool = mp.Pool(4)
    # res = [pool.apply_async(get_seperate_embedding, args=(folder, )) for folder in os.listdir(folder_path)]

    # for r in res:
    #     r.get()

    # pool.close()
    # pool.join()     

    # for folder in ['ORCL', 'T', 'IBM', 'NFLX']:
    #     get_seperate_embedding(folder)