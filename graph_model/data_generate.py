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
        raise ValueError("tensor_list 为空时无法确定维度，请至少传一个元素来确定形状")
    
    if isinstance(tensor_list[0], int):
        if n < k:
            pad = [0] * (k - n)
            return tensor_list + pad
        elif n > k:
            return tensor_list[:k]
        else:
            return tensor_list
    else:
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


def data_generate(num_fac=15, num_hist=20):

    factor_rag = FactorRAG(factor_pool_path, factor_embedding_path)
    data_store = MeetingStore()

    # 读取分组后的会议
    with open('conference_info_grouped.json', 'r') as f:
        meeting_groups = json.load(f)

    for batch_idx, group in enumerate(meeting_groups):
        batch_ec_factor_jsons = []
        batch_ec_names = []

        for record in group:
            date = record['date']
            time = record['time']
            ec_name = record['ec_name']
            ebds_addr = record['ebds_addr']

            if not os.path.exists(ebds_addr):
                continue
            text_emb = torch.load(ebds_addr, weights_only=True)
            pres_text_emb = text_emb[0].unsqueeze(0)
            qnda_text_emb = text_emb[1].unsqueeze(0)

            ret_close = record['ret_close']
            ret_sign = record['ret_sign']
            ret_close_20d = record['ret_close_20d']
            ret_sign_20d = record['ret_sign_20d']
            ret_vol_20d = record['ret_vol_20d']
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
            pres_len = []
            qnda_len = []
            for key, value in ec_factor_json.items():
                if key == 'Pres':
                    for fac in value:
                        if fac not in pres_factor and len(pres_factor) < num_fac:
                            factor_emb = factor_rag.get_factor_embedding(fac)
                            if factor_emb is None:
                                continue
                            pres_factor.append(factor_emb)

                            pres_emb, pres_label, valid_len = factor_rag.get_topk_aggregated_embedding(fac, mode='pres', top_k=num_hist)
                            if pres_emb is not None:
                                pres_fac_hist.append(pres_emb)
                                pres_fac_label.append(pres_label)
                                pres_len.append(valid_len)
                else:
                    for fac in value:
                        if fac not in qnda_factor and len(qnda_factor) < num_fac:
                            factor_emb = factor_rag.get_factor_embedding(fac)
                            if factor_emb is None:
                                continue
                            qnda_factor.append(factor_emb)

                            qnda_emb, qnda_label, valid_len = factor_rag.get_topk_aggregated_embedding(fac, mode='qnda', top_k=num_hist)
                            if qnda_emb is not None:
                                qnda_fac_hist.append(qnda_emb)
                                qnda_fac_label.append(qnda_label)
                                qnda_len.append(valid_len)

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
            if len(pres_len) == 0:
                pres_len.append(0)
            if len(qnda_len) == 0:
                qnda_len.append(0)

            pres_factor = pad_to_k(pres_factor, k=num_fac)
            qnda_factor = pad_to_k(qnda_factor, k=num_fac)
            pres_fac_hist = pad_to_k(pres_fac_hist, k=num_fac)
            qnda_fac_hist = pad_to_k(qnda_fac_hist, k=num_fac)
            pres_fac_label = pad_to_k(pres_fac_label, k=num_fac)
            qnda_fac_label = pad_to_k(qnda_fac_label, k=num_fac)
            pres_len = pad_to_k(pres_len, k=num_fac)
            qnda_len = pad_to_k(qnda_len, k=num_fac)

            data_store.update(ec_name, pres_text_emb, qnda_text_emb,
                              ret_close, ret_sign, ret_close_20d, ret_sign_20d, ret_vol_20d,
                              pres_factor, qnda_factor, pres_fac_hist, qnda_fac_hist,
                              pres_fac_label, qnda_fac_label, pres_len, qnda_len)
            
            batch_ec_factor_jsons.append(ec_factor_json)
            batch_ec_names.append(ec_name)
            
        for ec_factor_json, ec_name in zip(batch_ec_factor_jsons, batch_ec_names):
            factor_rag.update_record(ec_factor_json, ec_name)

    data_store.save_pickle(f'meeting_data_store_{num_fac}_{num_hist}.pkl')
            

if __name__ == '__main__':
    data_generate(num_fac=15, num_hist=20)

            




