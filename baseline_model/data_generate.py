import os
import json
import numpy as np
import torch
import torch.nn as nn

from data_structure import FactorRAG, MeetingStore


factor_pool_path = "./multi_factor_pool_0_2.json"
factor_embedding_path = "./multi_factor_pool_0_2_embedding.pt"


def pad_to_k(tensor_list, k=15):
    """
    把一批以 (1, d1, d2, ...) 形式的张量拼成 (k, d1, d2, ...)
    - 如果数量不足 k，补零
    - 如果数量超过 k，截断
    - 如果 list 为空，返回全零 (k, d1, d2, ...)
    """
    n = len(tensor_list)

    if n == 0:
        raise ValueError("tensor_list 为空时无法确定维度，请至少传一个 tensor 来确定形状")

    # 确定目标维度：去掉第一个 1
    shape_tail = tensor_list[0].shape[1:]   # (d1, d2, ...)

    # 拼接已有的
    x = torch.cat(tensor_list, dim=0)  # (n, d1, d2, ...)

    # 截断或补零
    if n > k:
        return x[:k]

    if n < k:
        pad_shape = (k - n,) + shape_tail
        pad = torch.zeros(pad_shape)
        return torch.cat([x, pad], dim=0)

    return x


def data_generate():

    factor_rag = FactorRAG(factor_pool_path, factor_embedding_path)
    data_store = MeetingStore()

    # 读取分组后的会议
    with open('meeting_groups_by_trading_day.json', 'r') as f:
        meeting_groups = json.load(f)

    for batch_idx, group in enumerate(meeting_groups):
        batch_ec_factor_jsons = []
        batch_ec_names = []

        for record in group:
            date = record['date']
            time = record['time']
            ec_name = record['ec_name']
            ebds_addr = record['ebds_addr']

            text_emb = torch.load(ebds_addr)
            pres_text_emb = text_emb[0].unsqueeze(0)
            qnda_text_emb = text_emb[1].unsqueeze(0)

            label = record['1_day_return']
            company = ec_name.split('_')[0]

            ec_factor_json_path = f"./bloomberg_factor_match/{company}/{ec_name}.json"
            if not os.path.exists(ec_factor_json_path):
                continue
            with open(ec_factor_json_path, 'r') as f:
                ec_factor_json = json.load(f)

            pres_factor = []
            qnda_factor = []
            pres_fac_hist = []
            qnda_fac_hist = []
            pres_fac_label = []
            qnda_fac_label = []
            for key, value in ec_factor_json.items():
                if key == 'Pres':
                    for fac in value:
                        if fac not in pres_factor and len(pres_factor) < 15:
                            factor_emb = factor_rag.get_factor_embedding(fac)
                            if factor_emb is None:
                                continue
                            pres_factor.append(factor_emb)

                            pres_emb, pres_label = factor_rag.get_topk_aggregated_embedding(fac, mode='pres', top_k=20)
                            if pres_emb is not None:
                                pres_fac_hist.append(pres_emb)
                                pres_fac_label.append(pres_label)
                else:
                    for fac in value:
                        if fac not in qnda_factor and len(qnda_factor) < 15:
                            factor_emb = factor_rag.get_factor_embedding(fac)
                            if factor_emb is None:
                                continue
                            qnda_factor.append(factor_emb)

                            qnda_emb, qnda_label = factor_rag.get_topk_aggregated_embedding(fac, mode='qnda', top_k=20)
                            if pres_emb is not None:
                                qnda_fac_hist.append(qnda_emb)
                                qnda_fac_label.append(qnda_label)

            if len(pres_factor) == 0:
                pres_factor.append(torch.zeros((1, 768)))
            if len(qnda_factor) == 0:
                qnda_factor.append(torch.zeros((1, 768)))
            if len(pres_fac_hist) == 0:
                pres_fac_hist.append(torch.zeros((1, 20, 768)))
            if len(qnda_fac_hist) == 0:
                qnda_fac_hist.append(torch.zeros((1, 20, 768)))
            if len(pres_fac_label) == 0:
                pres_fac_label.append(torch.zeros((1, 20, 1)))
            if len(qnda_fac_label) == 0:
                qnda_fac_label.append(torch.zeros((1, 20, 1)))

            pres_factor = pad_to_k(pres_factor, k=15)
            qnda_factor = pad_to_k(qnda_factor, k=15)
            pres_fac_hist = pad_to_k(pres_fac_hist, k=15)
            qnda_fac_hist = pad_to_k(qnda_fac_hist, k=15)
            pres_fac_label = pad_to_k(pres_fac_label, k=15)
            qnda_fac_label = pad_to_k(qnda_fac_label, k=15)

            if len(pres_fac_label.shape) == 2:
                print(1)

            data_store.update(ec_name, pres_text_emb, qnda_text_emb, label,
                              pres_factor, qnda_factor, pres_fac_hist, qnda_fac_hist,
                              pres_fac_label, qnda_fac_label)
            
            batch_ec_factor_jsons.append(ec_factor_json)
            batch_ec_names.append(ec_name)
            
        for ec_factor_json, ec_name in zip(batch_ec_factor_jsons, batch_ec_names):
            factor_rag.update_record(ec_factor_json, ec_name)

    data_store.save_pickle('meeting_data_store.pkl')
            

if __name__ == '__main__':
    data_generate()

            




