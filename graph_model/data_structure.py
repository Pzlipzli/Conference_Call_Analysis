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

            # --- 保持你的 embedding 列表 ---
            self.factor_pres_ec_emb_list = {idx: [] for idx in range(len(self.factors))}
            self.factor_qnda_ec_emb_list = {idx: [] for idx in range(len(self.factors))}

            self.factor_record_pres = {idx: [] for idx in range(len(self.factors))}
            self.factor_record_qnda = {idx: [] for idx in range(len(self.factors))}

            # --- 这里：把 label 从单路改为四路（字典形式便于扩展） ---
            def _empty_label_dict():
                return {'ret_close': [], 'ret_sign': [], 'ret_close_20d': [], 'ret_sign_20d': [], 'ret_vol_20d': []}
            self.pres_label_list = {idx: _empty_label_dict() for idx in range(len(self.factors))}
            self.qnda_label_list = {idx: _empty_label_dict() for idx in range(len(self.factors))}

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
        """
        返回：pres/qnda 的 embedding 列表 以及 四个标签的字典（各自是 list）
        """
        idx = self.get_index(factor)
        if idx is None:
            return None, None, None

        pres_ec = self.factor_pres_ec_emb_list[idx]
        qnda_ec = self.factor_qnda_ec_emb_list[idx]
        # --- 现在返回 dict ---
        pres_label = self.pres_label_list[idx]
        qnda_label = self.qnda_label_list[idx]

        return pres_ec, qnda_ec, pres_label, qnda_label

    def update_ec_emb(self, idx, ec_name, mode):
        """
        将 ec_embedding 添加到对应的列表，并分别记录四类 label
        需要保证 date_list 的 record 里有 ret_close, ret_sign, hl_ratio, low_crash 四个键
        """
        for record in date_list:
            if record['ec_name'] == ec_name:
                ebds_addr = record['ebds_addr']
                ec_embedding = torch.load(ebds_addr)
                # --- 四个标签 ---
                lab_ret_close = float(record.get('ret_close', 0.0))
                lab_ret_sign  = float(record.get('ret_sign', 0.0))   # {-1,1} 或 {0,1} 均可先原样存
                lab_ret_close_20d  = float(record.get('ret_close_20d', 0.0))
                lab_ret_sign_20d = float(record.get('ret_sign_20d', 0.0))  # {0,1}
                lab_ret_vol_20d = float(record.get('ret_vol_20d', 0.0))
                break
        else:
            return  # 没找到就跳过

        if mode == 'pres':
            self.factor_pres_ec_emb_list[idx].append(ec_embedding[0].unsqueeze(0))
            self.pres_label_list[idx]['ret_close'].append(lab_ret_close)
            self.pres_label_list[idx]['ret_sign'].append(lab_ret_sign)
            self.pres_label_list[idx]['ret_close_20d'].append(lab_ret_close_20d)
            self.pres_label_list[idx]['ret_sign_20d'].append(lab_ret_sign_20d)
            self.pres_label_list[idx]['ret_vol_20d'].append(lab_ret_vol_20d)

        elif mode == 'qnda':
            self.factor_qnda_ec_emb_list[idx].append(ec_embedding[1].unsqueeze(0))
            self.qnda_label_list[idx]['ret_close'].append(lab_ret_close)
            self.qnda_label_list[idx]['ret_sign'].append(lab_ret_sign)
            self.qnda_label_list[idx]['ret_close_20d'].append(lab_ret_close_20d)
            self.qnda_label_list[idx]['ret_sign_20d'].append(lab_ret_sign_20d)
            self.pres_label_list[idx]['ret_vol_20d'].append(lab_ret_vol_20d)
        else:
            raise ValueError("Mode should be either 'pres' or 'qnda'.")
    
    def get_factor_embedding(self, factor):
        idx = self.get_index(factor)
        if idx is None:
            return None
        
        return self.factor_embeddings[idx].unsqueeze(0)        
    
    def get_topk_aggregated_embedding(self, factor, mode='pres', top_k=20):
        """
        输出:
        - embeddings:  (1, top_k, EMBEDDING_DIM)
        - ret_close:   (1, top_k, 1), torch.float
        - hist_len:    int，实际返回的有效历史条数 m（m<=top_k）
        仅返回历史因子的收益率序列（ret_close），其余标签不再返回。
        """
        idx = self.get_index(factor)
        if idx is None:
            return None

        if mode == 'pres':
            emb_list = self.factor_pres_ec_emb_list[idx]
            label_src = self.pres_label_list[idx]  # 可能是 dict 或 list
        elif mode == 'qnda':
            emb_list = self.factor_qnda_ec_emb_list[idx]
            label_src = self.qnda_label_list[idx]
        else:
            raise ValueError("Mode should be either 'pres' or 'qnda'.")

        embedding_dim = EMBEDDING_DIM
        emb_out = torch.zeros((top_k, embedding_dim), dtype=torch.float)
        ret_close_out = torch.zeros((top_k, 1), dtype=torch.float)

        if len(emb_list) == 0:
            # 无历史
            return emb_out.unsqueeze(0), ret_close_out.unsqueeze(0), 0

        # 取最近 top_k 个
        selected_embs = emb_list[-top_k:]
        stacked = torch.cat(selected_embs, dim=0)  # (m, d)
        m = stacked.size(0)
        emb_out[:m, :] = stacked

        # 兼容 label 两种格式
        if isinstance(label_src, dict):
            ret_close_list = label_src.get('ret_close', [])
        else:
            ret_close_list = label_src

        selected_ret = ret_close_list[-top_k:]
        if len(selected_ret) > 0:
            t = torch.as_tensor(selected_ret, dtype=torch.float).unsqueeze(1)  # (m,1)
            ret_close_out[:t.size(0), :] = t

        return emb_out.unsqueeze(0), ret_close_out.unsqueeze(0), m

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
        if store is None:
            self.store = {
                'ec_name': [],
                'pres_emb': None,
                'qnda_emb': None,
                # --- 用四个独立字段替代 label ---
                'ret_close': None,
                'ret_sign': None,
                'ret_close_20d': None,
                'ret_sign_20d': None,
                'ret_vol_20d': None,

                'pres_factors': None,
                'qnda_factors': None,
                'pres_fac_hist': None,
                'qnda_fac_hist': None,
                'pres_fac_hist_label': None,
                'qnda_fac_hist_label': None,
                'pres_fac_hist_len': None,
                'qnda_fac_hist_len': None,
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
    
    def _append_scalar_series(self, key: str, value: float):
        t = torch.tensor([value], dtype=torch.float32)
        if self.store[key] is None:
            self.store[key] = t
        else:
            self.store[key] = torch.cat([self.store[key], t], dim=0)

    # --- 新：一次性更新四个标签 ---
    def update_labels(self, ret_close: float, ret_sign: float, ret_close_20d: float, ret_sign_20d: float, ret_vol_20d: float):
        self._append_scalar_series('ret_close', ret_close)
        self._append_scalar_series('ret_sign', ret_sign)
        self._append_scalar_series('ret_close_20d', ret_close_20d)
        self._append_scalar_series('ret_sign_20d', ret_sign_20d)
        self._append_scalar_series('ret_vol_20d', ret_vol_20d)
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
    
    def update_pres_fac_hist_label(self, pres_fac_pres_hist_label: torch.Tensor,
                                    pres_fac_hist_len: list):
        pres_fac_pres_hist_label = pres_fac_pres_hist_label.unsqueeze(0)
        if self.store['pres_fac_hist_label'] is None:
            self.store['pres_fac_hist_label'] = pres_fac_pres_hist_label
        else:
            self.store['pres_fac_hist_label'] = torch.cat([self.store['pres_fac_hist_label'], pres_fac_pres_hist_label], dim=0)

        pres_fac_hist_len_tensor = torch.tensor(pres_fac_hist_len, dtype=torch.long).unsqueeze(0)
        if self.store['pres_fac_hist_len'] is None:
            self.store['pres_fac_hist_len'] = pres_fac_hist_len_tensor
        else:
            self.store['pres_fac_hist_len'] = torch.cat([self.store['pres_fac_hist_len'], pres_fac_hist_len_tensor], dim=0)
        return
    
    def update_qnda_fac_hist_label(self, qnda_fac_qnda_hist_label: torch.Tensor,
                                    qnda_fac_hist_len: list):
        qnda_fac_qnda_hist_label = qnda_fac_qnda_hist_label.unsqueeze(0)
        if self.store['qnda_fac_hist_label'] is None:
            self.store['qnda_fac_hist_label'] = qnda_fac_qnda_hist_label
        else:
            self.store['qnda_fac_hist_label'] = torch.cat([self.store['qnda_fac_hist_label'], qnda_fac_qnda_hist_label], dim=0)

        qnda_fac_hist_len_tensor = torch.tensor(qnda_fac_hist_len, dtype=torch.long).unsqueeze(0)
        if self.store['qnda_fac_hist_len'] is None:
            self.store['qnda_fac_hist_len'] = qnda_fac_hist_len_tensor
        else:
            self.store['qnda_fac_hist_len'] = torch.cat([self.store['qnda_fac_hist_len'], qnda_fac_hist_len_tensor], dim=0)
        return
    
    def update(self, ec_name: str, pres: torch.Tensor, qnda: torch.Tensor,
               ret_close: float, ret_sign: float, ret_close_20d: float, ret_sign_20d: float, ret_vol_20d: float,
               pres_factors: torch.Tensor, qnda_factors: torch.Tensor,
               pres_fac_hist: torch.Tensor, qnda_fac_hist: torch.Tensor,
               pres_fac_hist_label: torch.Tensor, qnda_fac_hist_label: torch.Tensor,
               pres_fac_hist_len: list, qnda_fac_hist_len: list):

        self.update_ec_name(ec_name)
        self.update_pres_emb(pres)
        self.update_qnda_emb(qnda)

        # --- 新：四个标签 ---
        self.update_labels(ret_close, ret_sign, ret_close_20d, ret_sign_20d, ret_vol_20d)

        self.update_pres_factors(pres_factors)
        self.update_qnda_factors(qnda_factors)
        self.update_pres_fac_hist(pres_fac_hist)
        self.update_qnda_fac_hist(qnda_fac_hist)
        self.update_pres_fac_hist_label(pres_fac_hist_label, pres_fac_hist_len)
        self.update_qnda_fac_hist_label(qnda_fac_hist_label, qnda_fac_hist_len)
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
