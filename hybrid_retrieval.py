import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import math
from loguru import logger


class RerankerClient:
    def __init__(
        self,
        api_key: str = None,
        model: str = "gte-rerank",
        top_n: int = 3
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self.top_n = top_n
        
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = None
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        top_n = top_n or self.top_n
        
        try:
            from dashscope import TextReRank
            
            docs_for_rerank = [
                doc.get("content", "")[:2000] for doc in documents
            ]
            
            response = TextReRank.call(
                model=self.model,
                query=query,
                documents=docs_for_rerank,
                top_n=top_n
            )
            
            if response.status_code == 200 and hasattr(response, 'output'):
                results = []
                for item in response.output.results:
                    original_idx = item.index - 1
                    if 0 <= original_idx < len(documents):
                        doc = documents[original_idx]
                        doc_copy = doc.copy()
                        doc_copy["rerank_score"] = item.relevance_score
                        doc_copy["rerank_rank"] = item.index
                        results.append(doc_copy)
                
                return results
            else:
                return documents[:top_n]
                
        except Exception as e:
            logger.error(f"Rerank失败: {e}")
            return documents[:top_n]


class LightweightReranker:
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        query_terms = set(query.lower().split())
        
        scored_docs = []
        for doc in documents:
            content = doc.get("content", "").lower()
            metadata = doc.get("metadata", {})
            
            score = 0
            
            for term in query_terms:
                if term in content:
                    score += 1
                if metadata.get("section_path", "").lower().count(term) > 0:
                    score += 2
            
            h1 = metadata.get("h1", "").lower()
            h2 = metadata.get("h2", "").lower()
            h3 = metadata.get("h3", "").lower()
            for term in query_terms:
                if term in h1 or term in h2 or term in h3:
                    score += 3
            
            original_score = doc.get("score", 0)
            combined_score = original_score * 0.3 + score * 0.7
            
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = combined_score
            scored_docs.append(doc_copy)
        
        scored_docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return scored_docs[:top_n]


class KeywordExtractor:
    def __init__(self, stopwords: List[str] = None):
        self.stopwords = stopwords or self._default_stopwords()
        self.keyword_doc_freq: Dict[str, int] = {}
        self.total_docs = 0
    
    def _default_stopwords(self) -> List[str]:
        return [
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "他",
            "她", "它", "们", "为", "与", "但", "或", "以", "对", "及",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "and", "but", "if", "or",
            "because", "until", "while", "about", "against", "this",
            "that", "these", "those", "am", "it", "its", "their"
        ]
    
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]
        return tokens
    
    def build_index(self, documents: List[Dict[str, Any]]):
        self.keyword_doc_freq.clear()
        self.total_docs = len(documents)
        
        for doc in documents:
            content = doc.get("content", "")
            tokens = set(self.tokenize(content))
            for token in tokens:
                self.keyword_doc_freq[token] = self.keyword_doc_freq.get(token, 0) + 1
        
        logger.info(f"关键词索引构建完成，共 {len(self.keyword_doc_freq)} 个关键词")


class BM25:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        keyword_extractor: KeywordExtractor = None
    ):
        self.k1 = k1
        self.b = b
        self.keyword_extractor = keyword_extractor or KeywordExtractor()
        self.doc_lengths: List[int] = []
        self.doc_avg_length = 0
        self.documents: List[Dict[str, Any]] = []
    
    def _calculate_idf(self, term: str) -> float:
        df = self.keyword_extractor.keyword_doc_freq.get(term, 0)
        if df == 0:
            return 0
        N = self.keyword_extractor.total_docs
        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def build_index(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.doc_lengths = []
        
        for doc in documents:
            content = doc.get("content", "")
            tokens = self.keyword_extractor.tokenize(content)
            self.doc_lengths.append(len(tokens))
        
        self.doc_avg_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self.keyword_extractor.build_index(documents)
        
        logger.info(f"BM25索引构建完成，文档数: {len(documents)}, 平均长度: {self.doc_avg_length:.2f}")
    
    def score(self, query: str, doc_idx: int) -> float:
        if doc_idx >= len(self.documents):
            return 0
        
        doc = self.documents[doc_idx]
        content = doc.get("content", "")
        tokens = self.keyword_extractor.tokenize(content)
        doc_length = self.doc_lengths[doc_idx]
        
        query_tokens = self.keyword_extractor.tokenize(query)
        
        score = 0
        for term in query_tokens:
            tf = tokens.count(term)
            if tf == 0:
                continue
            
            idf = self._calculate_idf(term)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.doc_avg_length)
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.documents:
            return []
        
        scores = []
        for idx in range(len(self.documents)):
            score = self.score(query, idx)
            if score > 0:
                scores.append((idx, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:top_k]:
            results.append({
                "index": idx,
                "score": score,
                "content": self.documents[idx].get("content", ""),
                "metadata": self.documents[idx].get("metadata", {})
            })
        
        return results


class VectorRetrieval:
    def __init__(
        self,
        embed_model: str = "text-embedding-v3",
        api_key: str = None,
        dimension: int = 1536
    ):
        self.embed_model = embed_model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.dimension = dimension
        self.faiss_index = None
        self.documents: List[Dict[str, Any]] = []
        
        try:
            import faiss
            self.faiss = faiss
            self._init_index()
            logger.info("Faiss索引初始化成功")
        except ImportError as e:
            logger.warning(f"Faiss未安装: {e}")
            self.faiss = None
    
    def _init_index(self):
        if self.faiss:
            self.faiss_index = self.faiss.IndexFlatL2(self.dimension)
    
    def _get_embedding(self, text: str) -> List[float]:
        import time
        
        for attempt in range(3):
            try:
                from dashscope import TextEmbedding
                rsp = TextEmbedding.call(
                    model=TextEmbedding.Models.text_embedding_v2,
                    input=text,
                    text_type="document"
                )
                if rsp.status_code == 200:
                    return rsp.output['embeddings'][0]['embedding']
            except Exception as e:
                logger.warning(f"获取embedding失败 (尝试 {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(1)
        
        return [0.0] * self.dimension
    
    def build_index(self, documents: List[Dict[str, Any]]):
        if not self.faiss or not self.faiss_index:
            logger.warning("Faiss未初始化，无法构建索引")
            return
        
        self.documents = documents
        
        embeddings = []
        for doc in documents:
            content = doc.get("content", "")
            embedding = self._get_embedding(content)
            embeddings.append(embedding)
        
        if embeddings:
            import numpy as np
            embeddings_array = np.array(embeddings, dtype=np.float32)
            logger.info(f"embeddings_array shape: {embeddings_array.shape}")
            self.faiss_index.add(embeddings_array)
        
        logger.info(f"向量索引构建完成，文档数: {len(documents)}, 维度: {self.dimension}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.faiss_index or not self.documents:
            logger.warning(f"VectorRetrieval.search 提前返回: faiss_index={self.faiss_index is not None}, documents={len(self.documents) if self.documents else 0}")
            return []
        
        try:
            import numpy as np
            query_embedding = self._get_embedding(query)
            query_array = np.array([query_embedding], dtype=np.float32)
            logger.info(f"查询embedding形状: {query_array.shape}, faiss_index类型: {type(self.faiss_index)}")
            distances, indices = self.faiss_index.search(
                query_array, 
                min(top_k, len(self.documents))
            )
            logger.info(f"搜索完成: distances形状={distances.shape if hasattr(distances, 'shape') else 'N/A'}, indices形状={indices.shape if hasattr(indices, 'shape') else 'N/A'}")
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.documents):
                    results.append({
                        "index": int(idx),
                        "score": float(1.0 / (1.0 + distance)),
                        "content": self.documents[idx].get("content", ""),
                        "metadata": self.documents[idx].get("metadata", {})
                    })
            
            return results
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []


class HybridRetrieval:
    def __init__(
        self,
        embed_model: str = "text-embedding-v3",
        api_key: str = None,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        rrf_k: int = 60,
        use_rerank: bool = True,
        rerank_top_k: int = 3,
        rerank_model: str = "gte-rerank"
    ):
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self.use_rerank = use_rerank
        self.rerank_top_k = rerank_top_k
        self.rerank_model = rerank_model
        
        self.bm25 = BM25()
        self.vector_retrieval = VectorRetrieval(
            embed_model=embed_model,
            api_key=api_key
        )
        
        self.reranker = None
        if use_rerank:
            self._init_reranker(api_key)
        
        self.documents: List[Dict[str, Any]] = []
    
    def _init_reranker(self, api_key: str = None):
        try:
            from dashscope import TextReRank
            self.reranker = RerankerClient(
                api_key=api_key,
                model=self.rerank_model
            )
            logger.info(f"Reranker初始化成功: {self.rerank_model}")
        except ImportError:
            logger.warning("dashscope不支持TextReRank，使用轻量级rerank")
            self.reranker = LightweightReranker()
        except Exception as e:
            logger.warning(f"Reranker初始化失败: {e}，使用轻量级rerank")
            self.reranker = LightweightReranker()
    
    def _rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        if not self.reranker or not documents:
            return documents[:top_k] if documents else []
        
        try:
            return self.reranker.rerank(query, documents, top_k)
        except Exception as e:
            logger.warning(f"Rerank失败: {e}，使用原始排序")
            return documents[:top_k]
    
    def build_index(self, documents: List[Dict[str, Any]]):
        logger.info(f"HybridRetrieval.build_index 被调用，文档数: {len(documents)}")
        self.documents = documents
        
        self.bm25.build_index(documents)
        self.vector_retrieval.build_index(documents)
        
        logger.info(f"混合检索索引构建完成，文档数: {len(documents)}")
    
    def _rrf_fusion(
        self,
        keyword_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        rrf_scores: Dict[int, float] = defaultdict(float)
        
        for rank, result in enumerate(keyword_results):
            doc_idx = result.get("index", -1)
            if doc_idx >= 0:
                rrf_scores[doc_idx] += 1.0 / (self.rrf_k + rank + 1)
        
        for rank, result in enumerate(vector_results):
            doc_idx = result.get("index", -1)
            if doc_idx >= 0:
                rrf_scores[doc_idx] += 1.0 / (self.rrf_k + rank + 1)
        
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_idx, score in sorted_docs:
            if 0 <= doc_idx < len(self.documents):
                keyword_score = next(
                    (r["score"] for r in keyword_results if r["index"] == doc_idx),
                    0
                )
                vector_score = next(
                    (r["score"] for r in vector_results if r["index"] == doc_idx),
                    0
                )
                
                final_score = (
                    self.keyword_weight * keyword_score + 
                    self.vector_weight * vector_score
                )
                
                results.append({
                    "index": doc_idx,
                    "score": final_score,
                    "keyword_score": keyword_score,
                    "vector_score": vector_score,
                    "rrf_score": score,
                    "content": self.documents[doc_idx].get("content", ""),
                    "metadata": self.documents[doc_idx].get("metadata", {})
                })
        
        return results
    
    def _weighted_fusion(
        self,
        keyword_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        all_docs: Dict[int, Dict[str, Any]] = {}
        
        for result in keyword_results:
            doc_idx = result.get("index", -1)
            if doc_idx >= 0:
                if doc_idx not in all_docs:
                    all_docs[doc_idx] = {
                        "index": doc_idx,
                        "keyword_score": 0,
                        "vector_score": 0,
                        "content": result.get("content", ""),
                        "metadata": result.get("metadata", {})
                    }
                all_docs[doc_idx]["keyword_score"] = result.get("score", 0)
        
        for result in vector_results:
            doc_idx = result.get("index", -1)
            if doc_idx >= 0:
                if doc_idx not in all_docs:
                    all_docs[doc_idx] = {
                        "index": doc_idx,
                        "keyword_score": 0,
                        "vector_score": 0,
                        "content": result.get("content", ""),
                        "metadata": result.get("metadata", {})
                    }
                all_docs[doc_idx]["vector_score"] = result.get("score", 0)
        
        max_keyword = max((d["keyword_score"] for d in all_docs.values()), default=1)
        max_vector = max((d["vector_score"] for d in all_docs.values()), default=1)
        
        for doc in all_docs.values():
            normalized_keyword = doc["keyword_score"] / max_keyword if max_keyword > 0 else 0
            normalized_vector = doc["vector_score"] / max_vector if max_vector > 0 else 0
            
            doc["score"] = (
                self.keyword_weight * normalized_keyword +
                self.vector_weight * normalized_vector
            )
        
        sorted_docs = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
        
        return sorted_docs[:top_k]
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: str = "weighted",
        use_rerank: bool = None
    ) -> List[Dict[str, Any]]:
        logger.info(f"HybridRetrieval.search 被调用: query={query[:50]}..., top_k={top_k}")
        keyword_results = self.bm25.search(query, top_k=top_k * 2)
        logger.info(f"BM25搜索完成，结果数: {len(keyword_results)}")
        vector_results = self.vector_retrieval.search(query, top_k=top_k * 2)
        logger.info(f"向量搜索完成，结果数: {len(vector_results)}")
        
        if fusion_method == "rrf":
            results = self._rrf_fusion(keyword_results, vector_results)[:top_k]
        else:
            results = self._weighted_fusion(keyword_results, vector_results, top_k * 2)
        
        use_rerank = use_rerank if use_rerank is not None else self.use_rerank
        if use_rerank and self.reranker:
            results = self._rerank(query, results, top_k)
        
        logger.info(f"混合检索完成，最终结果数: {len(results)}")
        return results
    
    def search_only_keyword(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.bm25.search(query, top_k=top_k)
    
    def search_only_vector(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.vector_retrieval.search(query, top_k=top_k)
    
    def set_weights(self, keyword_weight: float, vector_weight: float):
        total = keyword_weight + vector_weight
        self.keyword_weight = keyword_weight / total
        self.vector_weight = vector_weight / total
        logger.info(f"权重已更新: 关键词={self.keyword_weight:.2f}, 向量={self.vector_weight:.2f}")
    
    def _separate_headings_and_content(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        headings = []
        content = []
        
        for idx, doc in enumerate(self.documents):
            doc_type = doc.get("metadata", {}).get("content_type", doc.get("type", ""))
            doc_copy = doc.copy()
            doc_copy["index"] = idx
            
            if doc_type == "heading" or doc.get("type") == "heading":
                headings.append(doc_copy)
            else:
                content.append(doc_copy)
        
        return headings, content
    
    def _get_section_documents(
        self,
        section_path: str
    ) -> List[Dict[str, Any]]:
        section_docs = []
        
        for idx, doc in enumerate(self.documents):
            doc_section_path = doc.get("metadata", {}).get("section_path", "")
            if doc_section_path == section_path or section_path in doc_section_path:
                doc_copy = doc.copy()
                doc_copy["index"] = idx
                section_docs.append(doc_copy)
        
        return section_docs
    
    def search_with_routing(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: str = "weighted",
        use_rerank: bool = None,
        use_routing: bool = True,
        routing_top_k: int = 3
    ) -> List[Dict[str, Any]]:
        if not use_routing:
            return self.search(query, top_k, fusion_method, use_rerank)
        
        headings, content = self._separate_headings_and_content()
        
        if not headings:
            logger.warning("没有找到标题文档，使用普通检索")
            return self.search(query, top_k, fusion_method, use_rerank)
        
        temp_docs_backup = self.documents
        self.documents = headings
        heading_results = self.search(query, top_k=routing_top_k, fusion_method=fusion_method, use_rerank=False)
        self.documents = temp_docs_backup
        
        if not heading_results:
            logger.warning("标题检索无结果，使用普通检索")
            return self.search(query, top_k, fusion_method, use_rerank)
        
        section_paths = []
        for result in heading_results:
            section_path = result.get("metadata", {}).get("section_path", "")
            if section_path and section_path not in section_paths:
                section_paths.append(section_path)
        
        if not section_paths:
            return self.search(query, top_k, fusion_method, use_rerank)
        
        all_section_results = []
        
        for section_path in section_paths:
            section_docs = self._get_section_documents(section_path)
            
            if not section_docs:
                continue
            
            temp_docs_backup = self.documents
            self.documents = section_docs
            
            section_results = self.search(
                query, 
                top_k=top_k, 
                fusion_method=fusion_method, 
                use_rerank=False
            )
            
            self.documents = temp_docs_backup
            
            for result in section_results:
                result["matched_section"] = section_path
            
            all_section_results.extend(section_results)
        
        if not all_section_results:
            return self.search(query, top_k, fusion_method, use_rerank)
        
        all_section_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        final_results = all_section_results[:top_k]
        
        use_rerank = use_rerank if use_rerank is not None else self.use_rerank
        if use_rerank and self.reranker:
            final_results = self._rerank(query, final_results, top_k)
        
        return final_results
    
    def _count_parent_occurrences(
        self,
        results: List[Dict[str, Any]],
        node_id_to_parent: Dict[str, str]
    ) -> Dict[str, int]:
        parent_counts: Dict[str, int] = defaultdict(int)
        
        for result in results:
            node_id = result.get("metadata", {}).get("node_id", "")
            if not node_id:
                continue
            
            parent_id = node_id_to_parent.get(node_id)
            if parent_id:
                parent_counts[parent_id] += 1
        
        return parent_counts
    
    def _get_parent_documents(
        self,
        parent_ids: List[str],
        node_id_to_doc: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        parent_docs = []
        
        for parent_id in parent_ids:
            doc = node_id_to_doc.get(parent_id)
            if doc:
                doc_copy = doc.copy()
                doc_copy["is_merged_parent"] = True
                doc_copy["original_node_ids"] = [
                    nid for nid, pid in node_id_to_doc.items() 
                    if pid == parent_id
                ]
                parent_docs.append(doc_copy)
        
        return parent_docs
    
    def search_with_auto_merge(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: str = "weighted",
        use_rerank: bool = None,
        merge_threshold: float = 0.5,
        use_routing: bool = False
    ) -> List[Dict[str, Any]]:
        if use_routing:
            results = self.search_with_routing(
                query, 
                top_k=top_k * 3, 
                fusion_method=fusion_method, 
                use_rerank=False
            )
        else:
            results = self.search(
                query, 
                top_k=top_k * 3, 
                fusion_method=fusion_method, 
                use_rerank=False
            )
        
        if not results:
            return results
        
        node_id_to_parent: Dict[str, str] = {}
        node_id_to_doc: Dict[str, Dict[str, Any]] = {}
        
        for idx, result in enumerate(results):
            metadata = result.get("metadata", {})
            node_id = metadata.get("node_id", "")
            parent_id = metadata.get("parent_id", "")
            
            if node_id:
                node_id_to_parent[node_id] = parent_id
                node_id_to_doc[node_id] = {
                    "content": result.get("content", ""),
                    "metadata": metadata,
                    "index": result.get("index", idx),
                    "score": result.get("score", 0)
                }
        
        if not node_id_to_parent:
            use_rerank = use_rerank if use_rerank is not None else self.use_rerank
            if use_rerank and self.reranker:
                results = self._rerank(query, results, top_k)
            return results[:top_k]
        
        parent_counts = self._count_parent_occurrences(results, node_id_to_parent)
        
        threshold_count = max(2, int(len(results) * merge_threshold))
        
        mergeable_parent_ids = [
            pid for pid, count in parent_counts.items() 
            if count >= threshold_count
        ]
        
        if not mergeable_parent_ids:
            use_rerank = use_rerank if use_rerank is not None else self.use_rerank
            if use_rerank and self.reranker:
                results = self._rerank(query, results, top_k)
            return results[:top_k]
        
        parent_docs = self._get_parent_documents(mergeable_parent_ids, node_id_to_doc)
        
        non_merged_results = [
            r for r in results 
            if r.get("metadata", {}).get("parent_id") not in mergeable_parent_ids
        ]
        
        merged_results = []
        for parent_doc in parent_docs:
            merged_results.append({
                "content": parent_doc.get("content", ""),
                "metadata": parent_doc.get("metadata", {}),
                "score": parent_doc.get("score", 0) * 1.2,
                "is_merged_parent": True,
                "original_node_count": len(parent_doc.get("original_node_ids", []))
            })
        
        final_results = merged_results + non_merged_results
        
        final_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        final_results = final_results[:top_k]
        
        use_rerank = use_rerank if use_rerank is not None else self.use_rerank
        if use_rerank and self.reranker:
            final_results = self._rerank(query, final_results, top_k)
        
        return final_results
    
    def search_with_hierarchy(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: str = "weighted",
        use_rerank: bool = None,
        merge_threshold: float = 0.5,
        use_routing: bool = True
    ) -> List[Dict[str, Any]]:
        return self.search_with_auto_merge(
            query=query,
            top_k=top_k,
            fusion_method=fusion_method,
            use_rerank=use_rerank,
            merge_threshold=merge_threshold,
            use_routing=use_routing
        )
    
    def recursive_search(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: str = "weighted",
        depth: int = 2,
        use_rerank: bool = None
    ) -> List[Dict[str, Any]]:
        if depth <= 0:
            return self.search(query, top_k, fusion_method, use_rerank)
        
        headings, content = self._separate_headings_and_content()
        
        if not headings:
            return self.search(query, top_k, fusion_method, use_rerank)
        
        temp_docs_backup = self.documents
        self.documents = headings
        heading_results = self.search(query, top_k=depth, fusion_method=fusion_method, use_rerank=False)
        self.documents = temp_docs_backup
        
        all_results = []
        
        for heading_result in heading_results:
            section_path = heading_result.get("metadata", {}).get("section_path", "")
            section_title = heading_result.get("metadata", {}).get("h1", "")
            
            if not section_path:
                continue
            
            section_docs = self._get_section_documents(section_path)
            
            if not section_docs:
                continue
            
            temp_docs_backup = self.documents
            self.documents = section_docs
            
            section_results = self.search(
                query,
                top_k=top_k,
                fusion_method=fusion_method,
                use_rerank=False
            )
            
            self.documents = temp_docs_backup
            
            for result in section_results:
                result["routing_section"] = section_path
                result["routing_title"] = section_title
                result["heading_score"] = heading_result.get("score", 0)
            
            all_results.extend(section_results)
        
        if not all_results:
            return self.search(query, top_k, fusion_method, use_rerank)
        
        all_results.sort(key=lambda x: x.get("score", 0) * 0.7 + x.get("heading_score", 0) * 0.3, reverse=True)
        final_results = all_results[:top_k]
        
        use_rerank = use_rerank if use_rerank is not None else self.use_rerank
        if use_rerank and self.reranker:
            final_results = self._rerank(query, final_results, top_k)
        
        return final_results
