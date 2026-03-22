import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger


class DashScopeEmbedding:
    def __init__(self, model_name: str = "text-embedding-v3", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")

    def _embed_text(self, text: str) -> List[float]:
        from dashscope import TextEmbedding
        response = TextEmbedding.call(
            model=self.model_name,
            text=text,
            api_key=self.api_key
        )
        if response.status_code == 200:
            return response.output['embeddings'][0]['embedding']
        raise Exception(f"Embedding failed: {response.message}")

    def get_text_embedding(self, text: str) -> List[float]:
        return self._embed_text(text)

    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_text(self, text: str) -> List[float]:
        return self._embed_text(text)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.get_text_embedding_batch(texts)


class HierarchicalNodeParser:
    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400,
        language: str = "zh"
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.language = language
    
    def _split_by_punctuation(self, text: str) -> List[str]:
        if not text:
            return []
        
        punctuation = r'[。！？；：\.\!\?\;\,]'
        
        parts = re.split(punctuation, text)
        
        sentences = []
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
        
        return sentences
    
    def _merge_into_chunks(self, sentences: List[str], max_size: int) -> List[str]:
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) <= max_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(sentence) <= max_size:
                    current_chunk = sentence
                else:
                    current_chunk = sentence[:max_size]
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def parse(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []

        sentences = self._split_by_punctuation(text)

        parent_chunks = []
        current_parent = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_parent) + len(sentence) <= self.parent_chunk_size:
                current_parent += sentence
            else:
                if current_parent:
                    parent_chunks.append(current_parent.strip())
                current_parent = sentence

        if current_parent.strip():
            parent_chunks.append(current_parent.strip())

        all_nodes = []

        for idx, parent_content in enumerate(parent_chunks):
            child_chunks = self._split_into_child_chunks(parent_content)

            parent_node = {
                "content": parent_content,
                "chunk_size": len(parent_content),
                "level": 0,
                "chunk_index": idx,
                "total_parent_chunks": len(parent_chunks),
                "child_count": len(child_chunks)
            }
            all_nodes.append(parent_node)

            for child_idx, child_content in enumerate(child_chunks):
                all_nodes.append({
                    "content": child_content,
                    "chunk_size": len(child_content),
                    "level": 1,
                    "chunk_index": child_idx,
                    "total_parent_chunks": len(parent_chunks),
                    "parent_chunk_index": idx
                })

        return all_nodes

    def _split_into_child_chunks(self, text: str) -> List[str]:
        if not text:
            return []

        sentences = self._split_by_punctuation(text)

        child_chunks = []
        current_child = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_child) + len(sentence) <= self.child_chunk_size:
                current_child += sentence
            else:
                if current_child:
                    child_chunks.append(current_child.strip())
                current_child = sentence

        if current_child.strip():
            child_chunks.append(current_child.strip())

        return child_chunks
    
    def parse_with_metadata(
        self,
        text: str,
        section_title: str = "",
        section_path: str = "",
        parent_id: str = None,
        doc_id: str = None
    ) -> List[Dict[str, Any]]:
        nodes = self.parse(text)
        
        for idx, node in enumerate(nodes):
            node["section_title"] = section_title
            node["section_path"] = section_path
            node["parent_id"] = parent_id
            node["doc_id"] = doc_id
            
            if node["level"] == 0:
                node["node_id"] = f"node_{doc_id}_parent_{idx}"
                node["is_parent"] = True
                node["child_ids"] = []
            else:
                node["node_id"] = f"node_{doc_id}_child_{node['parent_chunk_index']}_{idx}"
                node["is_parent"] = False
                node["parent_node_id"] = f"node_{doc_id}_parent_{node['parent_chunk_index']}"
        
        for node in nodes:
            if node["level"] == 0:
                parent_idx = node["parent_chunk_index"]
                child_ids = [
                    n["node_id"] for n in nodes
                    if n["level"] == 1 and n["parent_chunk_index"] == parent_idx
                ]
                node["child_ids"] = child_ids
        
        return nodes


class ChunkEmbedder:
    def __init__(
        self,
        api_key: str,
        embed_model: str = "text-embedding-v3",
        dimension: int = 1024
    ):
        self.api_key = api_key
        self.embed_model = embed_model
        self.dimension = dimension
        
        os.environ["DASHSCOPE_API_KEY"] = api_key
        
        self._init_embedding()
    
    def _init_embedding(self):
        try:
            self.embeddings = DashScopeEmbedding(
                model_name=self.embed_model,
                api_key=self.api_key
            )

            logger.info(f"Embedding模型初始化成功: {self.embed_model}")
        except Exception as e:
            logger.warning(f"Embedding初始化失败: {e}")
            self.embeddings = None
    
    def get_embedding(self, text: str) -> List[float]:
        if self.embeddings is None:
            logger.warning("Embedding未初始化，返回零向量")
            return [0.0] * self.dimension
        
        try:
            embedding = self.embeddings.get_text_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"获取embedding失败: {e}")
            return [0.0] * self.dimension
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.embeddings is None:
            logger.warning("Embedding未初始化，返回零向量")
            return [[0.0] * self.dimension for _ in texts]
        
        try:
            embeddings = self.embeddings.get_text_embedding_batch(texts)
            return embeddings
        except Exception as e:
            logger.error(f"批量获取embedding失败: {e}")
            return [[0.0] * self.dimension for _ in texts]


class FaissStore:
    def __init__(
        self,
        dimension: int = 1024,
        index_name: str = "default",
        index_path: str = None
    ):
        self.dimension = dimension
        self.index_name = index_name
        self.index_path = index_path
        
        self.faiss_index = None
        self.documents: List[Dict[str, Any]] = []
        self.metadata: List[Dict[str, Any]] = []
        
        self._init_faiss()
    
    def _init_faiss(self):
        try:
            import faiss
            
            self.faiss = faiss
            
            if self.index_path and Path(self.index_path).exists():
                try:
                    self.faiss_index = faiss.read_index(self.index_path)
                    self._load_metadata()
                    logger.info(f"从 {self.index_path} 加载Faiss索引")
                    return
                except Exception as e:
                    logger.warning(f"加载索引失败: {e}，创建新索引")
            
            self.faiss_index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"Faiss索引初始化成功，维度: {self.dimension}")
            
        except ImportError:
            logger.error("faiss未安装，请运行: pip install faiss-cpu")
            self.faiss = None
        except Exception as e:
            logger.error(f"Faiss初始化失败: {e}")
            self.faiss = None
    
    def _load_metadata(self):
        metadata_path = self._get_metadata_path()
        if metadata_path and Path(metadata_path).exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    self.metadata = data.get("metadata", [])
                logger.info(f"加载了 {len(self.documents)} 条文档元数据")
            except Exception as e:
                logger.warning(f"加载元数据失败: {e}")
                self.documents = []
                self.metadata = []
    
    def _get_metadata_path(self) -> Optional[str]:
        if self.index_path:
            return str(Path(self.index_path).with_suffix('.meta.json'))
        return None
    
    def _save_metadata(self):
        metadata_path = self._get_metadata_path()
        if not metadata_path:
            return
        
        try:
            Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump({
                    "documents": self.documents,
                    "metadata": self.metadata
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
    
    def save_index(self, path: str = None):
        save_path = path or self.index_path
        if not save_path or not self.faiss_index:
            logger.warning("未指定保存路径或索引未初始化")
            return
        
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.faiss.write_index(self.faiss_index, save_path)
            self._save_metadata()
            logger.info(f"索引已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
    
    def add(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]] = None
    ):
        if not self.faiss_index:
            logger.error("Faiss索引未初始化")
            return
        
        if not texts:
            return
        
        try:
            embedder = ChunkEmbedder(
                api_key=os.getenv("DASHSCOPE_API_KEY", ""),
                dimension=self.dimension
            )
            
            embeddings = embedder.get_embeddings(texts)
            
            self.faiss_index.add(embeddings)
            
            self.documents.extend(texts)
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{}] * len(texts))
            
            logger.info(f"添加了 {len(texts)} 个文档到索引")
            
        except Exception as e:
            logger.error(f"添加文档到索引失败: {e}")
    
    def add_with_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]] = None
    ):
        if not self.faiss_index:
            logger.error("Faiss索引未初始化")
            return
        
        if not texts or not embeddings:
            return
        
        if len(texts) != len(embeddings):
            logger.error("文本数量与向量数量不匹配")
            return
        
        try:
            import numpy as np
            
            embeddings_array = np.array(embeddings, dtype='float32')
            
            self.faiss_index.add(embeddings_array)
            
            self.documents.extend(texts)
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{}] * len(texts))
            
            logger.info(f"添加了 {len(texts)} 个文档到索引")
            
        except ImportError:
            logger.error("请安装numpy: pip install numpy")
        except Exception as e:
            logger.error(f"添加文档到索引失败: {e}")
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            logger.warning("索引为空")
            return []
        
        try:
            embedder = ChunkEmbedder(
                api_key=os.getenv("DASHSCOPE_API_KEY", ""),
                dimension=self.dimension
            )
            
            query_embedding = embedder.get_embedding(query)
            
            import numpy as np
            query_vector = np.array([query_embedding], dtype='float32')
            
            distances, indices = self.faiss_index.search(
                query_vector, 
                min(top_k, self.faiss_index.ntotal)
            )
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.documents):
                    results.append({
                        "content": self.documents[idx],
                        "metadata": self.metadata[idx] if idx < len(self.metadata) else {},
                        "score": float(1.0 / (1.0 + distance)),
                        "distance": float(distance),
                        "index": int(idx)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            return []
        
        try:
            import numpy as np
            
            query_vector = np.array([query_embedding], dtype='float32')
            
            distances, indices = self.faiss_index.search(
                query_vector,
                min(top_k, self.faiss_index.ntotal)
            )
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.documents):
                    results.append({
                        "content": self.documents[idx],
                        "metadata": self.metadata[idx] if idx < len(self.metadata) else {},
                        "score": float(1.0 / (1.0 + distance)),
                        "distance": float(distance),
                        "index": int(idx)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def clear(self):
        if self.faiss_index:
            self.faiss_index.reset()
        self.documents = []
        self.metadata = []
        logger.info("索引已清空")
    
    def __len__(self):
        if self.faiss_index:
            return self.faiss_index.ntotal
        return 0


class ChunkEmbeddingPipeline:
    def __init__(
        self,
        api_key: str,
        embed_model: str = "text-embedding-v3",
        dimension: int = 1024,
        index_name: str = "chunk_embeddings",
        index_path: str = None,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400
    ):
        self.api_key = api_key
        self.embed_model = embed_model
        self.dimension = dimension
        self.index_name = index_name
        self.index_path = index_path
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        
        self.embedder = ChunkEmbedder(
            api_key=api_key,
            embed_model=embed_model,
            dimension=dimension
        )
        
        self.faiss_store = FaissStore(
            dimension=dimension,
            index_name=index_name,
            index_path=index_path
        )
        
        self.node_parser = HierarchicalNodeParser(
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size
        )
        
        logger.info(f"ChunkEmbeddingPipeline初始化完成: {index_name}")
    
    def load_from_json(self, content_file: str) -> List[Dict[str, Any]]:
        with open(content_file, "r", encoding="utf-8") as f:
            content_data = json.load(f)
        
        content_items = content_data.get("content_list", [])
        
        chunks = []
        for i, item in enumerate(content_items):
            item_type = item.get("type", "")
            
            content = item.get("content", "")
            heading_context = item.get("heading_context", {})
            
            h1 = heading_context.get("h1", "")
            h2 = heading_context.get("h2", "")
            h3 = heading_context.get("h3", "")
            
            section_path = ""
            if h1:
                section_path = h1
                if h2:
                    section_path += f" > {h2}"
                    if h3:
                        section_path += f" > {h3}"
            
            chunks.append({
                "content": content,
                "type": item_type,
                "metadata": {
                    "h1": h1,
                    "h2": h2,
                    "h3": h3,
                    "section_path": section_path,
                    "item_index": i,
                    "content_type": item_type,
                    "doc_id": item.get("doc_id", f"doc_{i}")
                }
            })
        
        logger.info(f"从 {content_file} 加载了 {len(chunks)} 个文本块")
        return chunks
    
    def build_index_with_hierarchy(
        self,
        content_file: str = None,
        chunks: List[Dict[str, Any]] = None,
        save_index: bool = True
    ):
        if content_file:
            chunks = self.load_from_json(content_file)
        
        if not chunks:
            logger.warning("没有可处理的文本块")
            return
        
        all_nodes = []
        
        sections = {}
        heading_docs = {}
        
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            item_type = chunk.get("type", "")
            section_path = metadata.get("section_path", "")
            doc_id = metadata.get("doc_id", f"doc_{i}")
            
            if item_type == "heading":
                heading_docs[section_path] = {
                    "content": chunk.get("content", ""),
                    "metadata": metadata,
                    "doc_id": doc_id
                }
                continue
            
            if section_path not in sections:
                sections[section_path] = []
            sections[section_path].append({
                "content": chunk.get("content", ""),
                "metadata": metadata,
                "doc_id": doc_id
            })
        
        for section_path, section_chunks in sections.items():
            section_title = section_path.split(" > ")[-1] if " > " in section_path else section_path
            
            heading_info = heading_docs.get(section_path, {})
            heading_content = heading_info.get("content", section_title)
            heading_doc_id = heading_info.get("doc_id", f"doc_{section_title}")
            
            heading_node = {
                "content": heading_content,
                "type": "heading",
                "section_title": section_title,
                "section_path": section_path,
                "node_id": f"node_{heading_doc_id}_heading",
                "level": 0,
                "is_parent": True,
                "child_ids": [],
                "doc_id": heading_doc_id
            }
            all_nodes.append(heading_node)
            
            combined_text = "\n\n".join([
                c.get("content", "") for c in section_chunks
            ])
            
            if combined_text.strip():
                text_nodes = self.node_parser.parse_with_metadata(
                    text=combined_text,
                    section_title=section_title,
                    section_path=section_path,
                    parent_id=heading_node["node_id"],
                    doc_id=heading_doc_id
                )
                
                parent_nodes = [n for n in text_nodes if n.get("is_parent", False)]
                for parent_node in parent_nodes:
                    child_ids = [n["node_id"] for n in text_nodes if not n["is_parent"] and n.get("parent_node_id") == parent_node["node_id"]]
                    parent_node["child_ids"] = child_ids
                
                heading_node["child_ids"] = [n["node_id"] for n in text_nodes if n.get("is_parent", False)]
                
                all_nodes.extend(text_nodes)
        
        logger.info(f"递归节点构建完成，共 {len(all_nodes)} 个节点")
        
        texts = [node.get("content", "") for node in all_nodes]
        
        metadata = []
        for node in all_nodes:
            meta = {
                "node_id": node.get("node_id"),
                "section_title": node.get("section_title", ""),
                "section_path": node.get("section_path", ""),
                "parent_id": node.get("parent_id"),
                "chunk_level": node.get("level", 0),
                "chunk_size": node.get("chunk_size", 0),
                "is_parent": node.get("is_parent", False),
                "is_heading": node.get("type") == "heading",
                "doc_id": node.get("doc_id", "")
            }
            if "child_ids" in node:
                meta["child_ids"] = node["child_ids"]
            metadata.append(meta)
        
        embeddings = self.embedder.get_embeddings(texts)
        
        self.faiss_store.add_with_embeddings(texts, embeddings, metadata)
        
        if save_index:
            self.faiss_store.save_index()
        
        logger.info(f"层级索引构建完成，共 {len(texts)} 个文档")
    
    def build_index(
        self,
        content_file: str = None,
        chunks: List[Dict[str, Any]] = None,
        save_index: bool = True
    ):
        if content_file:
            chunks = self.load_from_json(content_file)
        
        if not chunks:
            logger.warning("没有可处理的文本块")
            return
        
        texts = [chunk.get("content", "") for chunk in chunks]
        metadata = [chunk.get("metadata", {}) for chunk in chunks]
        
        embeddings = self.embedder.get_embeddings(texts)
        
        self.faiss_store.add_with_embeddings(texts, embeddings, metadata)
        
        if save_index:
            self.faiss_store.save_index()
        
        logger.info(f"索引构建完成，共 {len(texts)} 个文档")
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        return self.faiss_store.search(query, top_k)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.search(query, top_k)
    
    def clear(self, clear_storage: bool = True):
        self.faiss_store.clear()
        logger.info("索引已清空")
    
    def __len__(self):
        return len(self.faiss_store)
