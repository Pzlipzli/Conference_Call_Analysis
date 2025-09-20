import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from collections import defaultdict
from datetime import datetime, timedelta
from rapidfuzz import fuzz
import pickle
from typing import List, Optional
import copy


EMBEDDING_DIM = 768
with open('conference_info.json', 'r') as f:
    date_list = json.load(f)

def load_ec_embedding(ec_name_list, date_list, mode='pres'):
    ec_embedding_list = []

    if mode == 'pres':
        idx = 0
    elif mode == 'qnda':
        idx = 1
    else:
        raise ValueError("Mode should be either 'pres' or 'qnda'.")

    for ec_name in ec_name_list:
        for record in date_list:
            if record['ec_name'] == ec_name:
                ebds_addr = record['ebds_addr']
                ec_embedding = torch.load(ebds_addr)[idx].unsqueeze(0)
                ec_embedding_list.append(ec_embedding)
                break

    if len(ec_embedding_list) == 0:
        return torch.zeros((1, EMBEDDING_DIM))
    
    ec_embedding_tensor = torch.cat(ec_embedding_list, dim=0)
    return ec_embedding_tensor


# s1 = "Operational efficiencies are improving."
# s2 = "Operating efficiencies are improving."
# if fuzzy_match(s1, s2):
#     print(f"'{s1}' 和 '{s2}' 匹配成功！")

class FactorRAG:
    def __init__(self, factor_pool_path, factor_embedding_path):
        with open(factor_pool_path, 'r') as f:
            self.factors = json.load(f)['textual']
            self.factor_embeddings = torch.load(factor_embedding_path)

            # 改为保存列表
            self.factor_pres_ec_emb_list = {idx: [] for idx in range(len(self.factors))}
            self.factor_qnda_ec_emb_list = {idx: [] for idx in range(len(self.factors))}        
                                       
            self.factor_record_pres = {idx: [] for idx in range(len(self.factors))}
            self.factor_record_qnda = {idx: [] for idx in range(len(self.factors))}

            self.pres_label_list = {idx: [] for idx in range(len(self.factors))}
            self.qnda_label_list = {idx: [] for idx in range(len(self.factors))}

    def __getitem__(self, idx):
        pres_ec = self.factor_record_pres[idx]
        qnda_ec = self.factor_record_qnda[idx]
        return pres_ec, qnda_ec
    
    def __len__(self):
        return len(self.factors)
    
    @staticmethod
    def fuzzy_match(s1, str_list, threshold=85):
        """
        使用 rapidfuzz 库进行近似匹配。
        """
        # 使用 token_set_ratio 算法，它对词序和无关词不敏感
        for s2 in str_list:
            score = fuzz.token_set_ratio(s1.lower(), s2.lower())

            if score >= threshold:
                return s2
        return None

    def get_index(self, factor):
        if factor in self.factors:
            return self.factors.index(factor)
        else:
            fuzz_result = self.fuzzy_match(factor, self.factors)
            if fuzz_result is not None:
                return self.factors.index(fuzz_result)
            
            return None
        
    def get_history_record(self, factor):
        idx = self.get_index(factor)
        if idx is None:
            return None, None

        return self.__getitem__(idx)
    
    def get_history_data(self, factor):
        idx = self.get_index(factor)
        if idx is None:
            return None, None, None

        pres_ec = self.factor_pres_ec_emb_list[idx]
        qnda_ec = self.factor_qnda_ec_emb_list[idx]
        pres_label = self.pres_label_list[idx]
        qnda_label = self.qnda_label_list[idx]

        return pres_ec, qnda_ec, pres_label, qnda_label
    
    def update_ec_emb(self, idx, ec_name, mode):
        """
        将 ec_embedding 添加到对应的列表，而不是做平均
        """
        for record in date_list:
            if record['ec_name'] == ec_name:
                ebds_addr = record['ebds_addr']
                ec_embedding = torch.load(ebds_addr)
                label_return = record['1_day_return']
                break
        else:
            # 找不到对应 embedding
            return
        
        if mode == 'pres':
            # 假设 ec_embedding[0] 对应 pres
            self.factor_pres_ec_emb_list[idx].append(ec_embedding[0].unsqueeze(0))
            self.pres_label_list[idx].append(label_return)

        elif mode == 'qnda':
            # 假设 ec_embedding[1] 对应 qnda
            self.factor_qnda_ec_emb_list[idx].append(ec_embedding[1].unsqueeze(0))
            self.qnda_label_list[idx].append(label_return)

        else:
            raise ValueError("Mode should be either 'pres' or 'qnda'.")

    # def get_ec_embedding(self, factor, mode):
    #     idx = self.get_index(factor)
    #     if idx is None:
    #         return None

    #     if mode == 'pres':
    #         return self.factor_pres_ec_emb[idx]
    #     elif mode == 'qnda':
    #         return self.factor_qnda_ec_emb[idx]
    
    def get_factor_embedding(self, factor):
        idx = self.get_index(factor)
        if idx is None:
            return None
        
        return self.factor_embeddings[idx].unsqueeze(0)        
    
    def get_topk_aggregated_embedding(self, factor, mode='pres', top_k=20):
        """
        factor: factor 文本
        mode: 'pres' 或 'qnda'
        top_k: 聚合前选取的 top_k embeddings
        输出: shape = (top_k, embedding_dim)
        """
        idx = self.get_index(factor)
        if idx is None:
            # 返回None
            return None

        if mode == 'pres':
            emb_list = self.factor_pres_ec_emb_list[idx]
            label_list = self.pres_label_list[idx]
        elif mode == 'qnda':
            emb_list = self.factor_qnda_ec_emb_list[idx]
            label_list = self.qnda_label_list[idx]
        else:
            raise ValueError("Mode should be either 'pres' or 'qnda'.")

        embedding_dim = EMBEDDING_DIM  # 假设你全局定义了 EMBEDDING_DIM
        output = torch.zeros((top_k, embedding_dim), dtype=torch.float)
        label_return = torch.zeros((top_k, 1), dtype=torch.float)

        if len(emb_list) == 0:
            # 没有历史 embedding，直接返回全0
            return output.unsqueeze(0), label_return.unsqueeze(0)

        # 取最近 top_k 个
        selected_embs = emb_list[-top_k:]
        stacked = torch.cat(selected_embs, dim=0)  # shape: (len(selected_embs), dim)

        # 取最近 top_k 个 label
        selected_labels = label_list[-top_k:]
        stacked_label = torch.tensor(selected_labels, dtype=torch.float).unsqueeze(1) # shape: (len(selected_embs), 1)

        # 填充到 output 的前 len(selected_embs) 行
        output[:stacked.size(0), :] = stacked
        label_return[:stacked_label.size(0), :] = stacked_label

        return output.unsqueeze(0), label_return.unsqueeze(0)


    def update_record(self, ec_factor_json, ec_name):
        pres_list = []
        qnda_list = []

        for key, value in ec_factor_json.items():
            if key == 'Pres':
                for fac in value:
                    if fac not in pres_list:
                        idx = self.get_index(fac)
                        if idx is None:
                            continue
                        self.update_ec_emb(idx, ec_name, 'pres')
                        self.factor_record_pres[idx].append(ec_name)
                        pres_list.append(fac)
            else:
                for fac in value:    
                    if fac not in qnda_list:
                        idx = self.get_index(fac)
                        if idx is None:
                            continue
                        self.update_ec_emb(idx, ec_name, 'qnda')
                        self.factor_record_qnda[idx].append(ec_name)
                        qnda_list.append(fac)
        return

    def restart_record(self):
        self.factor_record_pres = {idx: [] for idx in range(len(self.factors))}
        self.factor_record_qnda = {idx: [] for idx in range(len(self.factors))}
        
        return


class MeetingStore:
    def __init__(self, store=None):
        # 主数据结构：dict
        if store is None:
            self.store = {'ec_name': [],
                        'pres_emb': None,
                        'qnda_emb': None,
                        'label': None,
                        'pres_factors': None,
                        'qnda_factors': None,
                        'pres_fac_hist': None,
                        'qnda_fac_hist': None,
                        'pres_fac_hist_label': None,
                        'qnda_fac_hist_label': None
                        }
        else:
            self.store = store
        
    def update_ec_name(self, ec_name: str):
        if ec_name not in self.store['ec_name']:
            self.store['ec_name'].append(ec_name)
        return
    
    def update_pres_emb(self, pres: torch.Tensor):
        if self.store['pres_emb'] is None:
            self.store['pres_emb'] = pres
        else:
            self.store['pres_emb'] = torch.cat([self.store['pres_emb'], pres], dim=0)
        return
    
    def update_qnda_emb(self, qnda: torch.Tensor):
        if self.store['qnda_emb'] is None:
            self.store['qnda_emb'] = qnda
        else:
            self.store['qnda_emb'] = torch.cat([self.store['qnda_emb'], qnda], dim=0)
        return
    
    def update_label(self, label: float):
        label_tensor = torch.tensor([label], dtype=torch.float32)
        if self.store['label'] is None:
            self.store['label'] = label_tensor
        else:
            self.store['label'] = torch.cat([self.store['label'], label_tensor], dim=0)
        return
    
    def update_pres_factors(self, pres_factors: torch.Tensor):
        pres_factors = pres_factors.unsqueeze(0)
        if self.store['pres_factors'] is None:
            self.store['pres_factors'] = pres_factors
        else:
            self.store['pres_factors'] = torch.cat([self.store['pres_factors'], pres_factors], dim=0)
        return
    
    def update_qnda_factors(self, qnda_factors: torch.Tensor):
        qnda_factors = qnda_factors.unsqueeze(0)
        if self.store['qnda_factors'] is None:
            self.store['qnda_factors'] = qnda_factors
        else:
            self.store['qnda_factors'] = torch.cat([self.store['qnda_factors'], qnda_factors], dim=0)
        return
    
    def update_pres_fac_hist(self, pres_fac_pres_hist: torch.Tensor):
        pres_fac_pres_hist = pres_fac_pres_hist.unsqueeze(0)
        if self.store['pres_fac_hist'] is None:
            self.store['pres_fac_hist'] = pres_fac_pres_hist
        else:
            self.store['pres_fac_hist'] = torch.cat([self.store['pres_fac_hist'], pres_fac_pres_hist], dim=0)
        return

    def update_qnda_fac_hist(self, qnda_fac_qnda_hist: torch.Tensor):
        qnda_fac_qnda_hist = qnda_fac_qnda_hist.unsqueeze(0)
        if self.store['qnda_fac_hist'] is None:
            self.store['qnda_fac_hist'] = qnda_fac_qnda_hist
        else:
            self.store['qnda_fac_hist'] = torch.cat([self.store['qnda_fac_hist'], qnda_fac_qnda_hist], dim=0)
        return
    
    def update_pres_fac_hist_label(self, pres_fac_pres_hist_label: torch.Tensor):
        pres_fac_pres_hist_label = pres_fac_pres_hist_label.unsqueeze(0)
        if self.store['pres_fac_hist_label'] is None:
            self.store['pres_fac_hist_label'] = pres_fac_pres_hist_label
        else:
            self.store['pres_fac_hist_label'] = torch.cat([self.store['pres_fac_hist_label'], pres_fac_pres_hist_label], dim=0)
        return
    
    def update_qnda_fac_hist_label(self, qnda_fac_qnda_hist_label: torch.Tensor):
        qnda_fac_qnda_hist_label = qnda_fac_qnda_hist_label.unsqueeze(0)
        if self.store['qnda_fac_hist_label'] is None:
            self.store['qnda_fac_hist_label'] = qnda_fac_qnda_hist_label
        else:
            self.store['qnda_fac_hist_label'] = torch.cat([self.store['qnda_fac_hist_label'], qnda_fac_qnda_hist_label], dim=0)
        return
    
    def update(self, ec_name: str, pres: torch.Tensor, qnda:torch.Tensor, label: float,
            pres_factors: torch.Tensor, qnda_factors:torch.Tensor, pres_fac_hist: torch.Tensor, qnda_fac_hist: torch.Tensor,
            pres_fac_hist_label: torch.Tensor, qnda_fac_hist_label: torch.Tensor):
        self.update_ec_name(ec_name)
        self.update_pres_emb(pres)
        self.update_qnda_emb(qnda)
        self.update_label(label)
        self.update_pres_factors(pres_factors)
        self.update_qnda_factors(qnda_factors)
        self.update_pres_fac_hist(pres_fac_hist)
        self.update_qnda_fac_hist(qnda_fac_hist)
        self.update_pres_fac_hist_label(pres_fac_hist_label)
        self.update_qnda_fac_hist_label(qnda_fac_hist_label)

        return

    def get_item(self, item, start, end):
        if item not in self.store:
            raise ValueError(f"Item '{item}' not found in store.")
        
        data = self.store[item]
        if data is None:
            return None
        
        if item == 'ec_name':
            return data[start:end]
        else:
            return data[start:end].clone()
    
    def get_idx(self, ec_name: str):
        if ec_name not in self.store['ec_name']:
            raise ValueError(f"EC name '{ec_name}' not found in store.")
        
        return self.store['ec_name'].index(ec_name)
    
    def copy_dataset(self, start, end):
        new_store = {}
        for key, value in self.store.items():
            if value is None:
                new_store[key] = None
                continue

            # list（ec_name）直接切片即可
            if isinstance(value, list):
                new_store[key] = value[start:end]
            # tensor：切片 + clone，避免 view
            elif isinstance(value, torch.Tensor):
                subset = value[start:end]
                new_store[key] = subset.clone()
            else:
                # 兜底：复制引用或 copy.copy
                new_store[key] = copy.copy(value)

        return MeetingStore(new_store)
    
    def update_device(self, device: str):
        for key, value in self.store.items():
            if value is not None and key != 'ec_name':
                self.store[key] = value.to(device)
        return
    
    def save_pickle(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.store, f)
    
    def load_pickle(self, path: str):
        with open(path, "rb") as f:
            self.store = pickle.load(f)
