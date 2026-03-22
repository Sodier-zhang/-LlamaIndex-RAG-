import os
import json
from typing import Optional, List, Dict, Any, Callable, Union
from pathlib import Path
from loguru import logger

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
    StorageContext,
)
import faiss
from datetime import datetime

from hybrid_retrieval import HybridRetrieval


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


class DashScopeLLM:
    def __init__(self, model: str = "qwen-turbo", api_key: str = None, temperature: float = 0.7):
        self.model = model
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.temperature = temperature

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        from dashscope import Generation
        from llama_index.core.llms import ChatMessage

        llm_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            llm_messages.append({"role": role, "content": content})

        response = Generation.call(
            model=self.model,
            messages=llm_messages,
            temperature=self.temperature,
            result_format="message"
        )

        if response.status_code == 200:
            return {
                "message": ChatMessage(
                    role="assistant",
                    content=response.output.choices[0].message.content
                ),
                "raw": response.raw
            }
        raise Exception(f"LLM call failed: {response.message}")

    def stream_chat(self, messages: List[Dict[str, str]]):
        from dashscope import Generation

        llm_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            llm_messages.append({"role": role, "content": content})

        return Generation.call(
            model=self.model,
            messages=llm_messages,
            temperature=self.temperature,
            result_format="message",
            stream=True
        )


class LlamaIndexRAG:
    def __init__(
        self,
        api_key: str,
        embed_model: str = "text-embedding-v3",
        llm_model: str = "qwen-turbo",
        index_name: str = "llamaindex_rag_docs",
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        faiss_index_path: str = None,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        use_rerank: bool = True,
        use_routing: bool = True,
        use_auto_merge: bool = True
    ):
        self.api_key = api_key
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.faiss_index_path = faiss_index_path
        self.dimension = 1536
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.use_rerank = use_rerank
        self.use_routing = use_routing
        self.use_auto_merge = use_auto_merge

        os.environ["DASHSCOPE_API_KEY"] = api_key

        self.embeddings = DashScopeEmbedding(
            model_name=embed_model,
            api_key=api_key
        )

        self._init_hybrid_retrieval()
        self._init_vector_store()
    
    def _init_hybrid_retrieval(self):
        try:
            self.hybrid_retrieval = HybridRetrieval(
                embed_model=self.embed_model,
                api_key=self.api_key,
                keyword_weight=self.keyword_weight,
                vector_weight=self.vector_weight,
                use_rerank=self.use_rerank
            )
            logger.info("混合检索初始化成功")
        except Exception as e:
            logger.warning(f"混合检索初始化失败: {e}")
            self.hybrid_retrieval = None

    def _init_vector_store(self):
        self.vector_store = None
        logger.info("向量存储使用LlamaIndex默认内存存储")
    
    def load_documents(self, content_file: str) -> List[Document]:
        with open(content_file, "r", encoding="utf-8") as f:
            content_data = json.load(f)
        
        content_items = content_data.get("content_list", [])
        
        documents = []
        
        for i, item in enumerate(content_items):
            if item.get("type") in ["text", "heading"]:
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
                
                doc = Document(
                    text=content,
                    metadata={
                        "h1": h1,
                        "h2": h2,
                        "h3": h3,
                        "section_path": section_path,
                        "item_index": i,
                        "content_type": item.get("type", "text")
                    },
                    id_=f"doc_{i}"
                )
                documents.append(doc)
        
        logger.info(f"加载了 {len(documents)} 个文档")
        return documents

    def build_index(self, documents: List[Document]):
        from llama_index.core import Settings
        from llama_index.core.embeddings import BaseEmbedding
        import typing

        logger.info("开始构建向量索引...")

        try:
            logger.info("使用DashScope embedding模型")

            class LocalEmbedding(BaseEmbedding):
                model_name: str = "dashscope-v2"

                def _get_text_embedding(self, text: str) -> typing.List[float]:
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
                    
                    return [0.0] * 1536

                async def _aget_text_embedding(self, text: str) -> typing.List[float]:
                    return self._get_text_embedding(text)

                def _get_query_embedding(self, text: str) -> typing.List[float]:
                    return self._get_text_embedding(text)

                async def _aget_query_embedding(self, text: str) -> typing.List[float]:
                    return self._get_text_embedding(text)

            embed_model = LocalEmbedding()
            Settings.embed_model = embed_model

            logger.info("开始构建向量索引...")
            self.index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True
            )
            logger.info("向量索引构建完成")

            if self.hybrid_retrieval:
                hybrid_docs = [{"content": d.text, "metadata": d.metadata} for d in documents]
                self.hybrid_retrieval.build_index(hybrid_docs)
                logger.info("混合检索索引构建成功")

            logger.info("索引构建完成")
        except Exception as e:
            logger.error(f"索引构建失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def build_hierarchy_index(
        self,
        content_file: str,
        chunk_sizes: List[int] = None
    ):
        from chunk_embedding import ChunkEmbeddingPipeline
        
        pipeline = ChunkEmbeddingPipeline(
            api_key=self.api_key,
            embed_model=self.embed_model,
            dimension=self.dimension,
            index_name=self.index_name,
            chunk_sizes=chunk_sizes
        )
        
        pipeline.build_index_with_hierarchy(
            content_file=content_file,
            save_index=True
        )
        
        self.hierarchy_pipeline = pipeline
        logger.info("层级索引构建完成")
    
    def query(
        self,
        query_str: str,
        similarity_top_k: int = 5,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        if not hasattr(self, 'index'):
            raise ValueError("索引未构建，请先调用build_index方法")
        
        query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            streaming=False
        )
        
        response = query_engine.query(query_str)
        
        result = {
            "answer": str(response),
            "source_nodes": []
        }
        
        if include_sources and hasattr(response, 'source_nodes'):
            sources = []
            for node in response.source_nodes:
                source_info = {
                    "content": node.node.get_content()[:200] + "..." if len(node.node.get_content()) > 200 else node.node.get_content(),
                    "score": node.score if hasattr(node, 'score') else 0,
                    "metadata": node.node.metadata if hasattr(node.node, 'metadata') else {}
                }
                sources.append(source_info)
            
            result["source_nodes"] = sources
        
        return result
    
    def stream_query(
        self,
        query_str: str,
        similarity_top_k: int = 5,
        callback: Optional[Callable] = None
    ) -> str:
        if not hasattr(self, 'index'):
            raise ValueError("索引未构建，请先调用build_index方法")
        
        query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            streaming=True
        )
        
        full_response = ""
        
        if callback:
            for response in query_engine.stream_query(query_str):
                content = str(response)
                full_response += content
                callback(content)
        else:
            for response in query_engine.stream_query(query_str):
                full_response += str(response)
        
        return full_response
    
    def get_retriever(self, similarity_top_k: int = 5):
        if not hasattr(self, 'index'):
            raise ValueError("索引未构建，请先调用build_index方法")

        retriever = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        )

        return retriever
    
    def retrieve(
        self,
        query_str: str,
        similarity_top_k: int = 5,
        use_routing: bool = None,
        use_auto_merge: bool = None,
        merge_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        use_routing = use_routing if use_routing is not None else self.use_routing
        use_auto_merge = use_auto_merge if use_auto_merge is not None else self.use_auto_merge
        
        logger.info(f"retrieve 被调用: query={query_str[:50]}..., hybrid_retrieval={self.hybrid_retrieval is not None}, use_routing={use_routing}, use_auto_merge={use_auto_merge}")
        
        if self.hybrid_retrieval and (use_routing or use_auto_merge):
            if use_auto_merge:
                results = self.hybrid_retrieval.search_with_hierarchy(
                    query=query_str,
                    top_k=similarity_top_k,
                    use_routing=use_routing,
                    merge_threshold=merge_threshold
                )
            else:
                results = self.hybrid_retrieval.search_with_routing(
                    query=query_str,
                    top_k=similarity_top_k,
                    use_routing=use_routing
                )
            return results
        
        if self.hybrid_retrieval:
            logger.info("使用 hybrid_retrieval.search")
            return self.hybrid_retrieval.search(query_str, top_k=similarity_top_k)
        
        retriever = self.get_retriever(similarity_top_k=similarity_top_k)
        nodes = retriever.retrieve(query_str)
        
        results = []
        for node in nodes:
            results.append({
                "content": node.node.get_content(),
                "score": node.score if hasattr(node, 'score') else 0,
                "metadata": node.node.metadata if hasattr(node.node, 'metadata') else {}
            })
        
        return results


class LLMProvider:
    def __init__(self, api_key: str, model: str = "qwen-turbo"):
        self.api_key = api_key
        self.model = model
        os.environ["DASHSCOPE_API_KEY"] = api_key

        self.llm = DashScopeLLM(model=model, api_key=api_key)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        stream: bool = False
    ) -> Union[Dict, Callable]:
        if stream:
            return self._stream_generate(messages, temperature)
        else:
            return self._generate(messages, temperature)

    def _generate(self, messages: List[Dict[str, str]], temperature: float) -> Dict:
        response = self.llm.chat(messages)

        return {
            "content": str(response["message"].content),
            "usage": response.get("raw", {}).get("usage", {})
        }

    def _stream_generate(self, messages: List[Dict[str, str]], temperature: float):
        return self.llm.stream_chat(messages)


class QueryProcessor:
    def __init__(self, api_key: str, model: str = "qwen-turbo"):
        self.api_key = api_key
        self.model = model
        os.environ["DASHSCOPE_API_KEY"] = api_key
    
    def rewrite_query(self, query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        from dashscope import Generation
        
        if conversation_history:
            history_text = "\n".join([
                f"用户: {msg.get('question', '')}\n助手: {msg.get('answer', '')}"
                for msg in conversation_history[-3:]
            ])
            prompt = f"""你是一个专业的学术论文问答助手。请将以下对话中的用户问题改写为一个适合检索学术论文的搜索查询。

对话历史：
{history_text}

当前问题：{query}

要求：
1. 如果问题是"这篇论文说了什么"、"总结一下"等，改为"请总结该论文的核心创新点、主要贡献和实验方案"
2. 如果问题是"实验怎么做"，改为"请描述该论文的实验设计、方法论和评估指标"
3. 如果问题是"结果是什么"，改为"请提取该论文的主要实验结果、关键数据和分析结论"
4. 如果问题是关于特定概念或术语，需要保留原问题中的专业术语
5. 去除口语化表达，输出简洁、精确的学术检索查询
6. 不要添加解释，直接输出改写后的查询语句

改写后的查询："""
        else:
            prompt = f"""你是一个专业的学术论文问答助手。请将以下用户问题改写为适合检索学术论文的查询语句。

原始问题：{query}

要求：
1. 如果问题是"这篇论文说了什么"、"总结一下"等，改为"请总结该论文的核心创新点、主要贡献和实验方案"
2. 如果问题是"实验怎么做"，改为"请描述该论文的实验设计、方法论和评估指标"
3. 如果问题是"结果是什么"，改为"请提取该论文的主要实验结果、关键数据和分析结论"
4. 如果问题是关于特定概念或术语，需要保留原问题中的专业术语
5. 去除口语化表达，输出简洁、精确的学术检索查询
6. 不要添加解释，直接输出改写后的查询语句

改写后的查询："""
        
        try:
            response = Generation.call(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                result_format="message"
            )
            
            if response.status_code == 200:
                rewritten = response.output.choices[0].message.content.strip()
                logger.info(f"查询改写: {query} -> {rewritten}")
                return rewritten
            else:
                logger.warning(f"查询改写失败: {response.message}")
                return query
        except Exception as e:
            logger.error(f"查询改写异常: {e}")
            return query
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        if len(documents) <= top_k:
            return documents
        
        from dashscope import Generation
        
        doc_texts = "\n".join([
            f"[文档 {i+1}]\n{doc.get('content', '')[:500]}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""请根据查询与文档的相关性，对以下文档进行排序。

查询：{query}

{doc_texts}

要求：
1. 只返回文档编号，按相关性从高到低排序
2. 格式：1,2,3,4,...（用逗号分隔）
3. 不要返回其他内容

排序结果："""
        
        try:
            response = Generation.call(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                result_format="message"
            )
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content.strip()
                ranks = [int(r.strip()) - 1 for r in content.split(",") if r.strip().isdigit()]
                
                reranked = []
                for rank in ranks:
                    if 0 <= rank < len(documents):
                        doc = documents[rank]
                        doc["rerank_score"] = len(ranks) - ranks.index(rank)
                        reranked.append(doc)
                
                for i, doc in enumerate(documents):
                    if doc not in reranked:
                        doc["rerank_score"] = 0
                        reranked.append(doc)
                
                logger.info(f"重排序完成，返回 {len(reranked[:top_k])} 条结果")
                return reranked[:top_k]
            else:
                logger.warning(f"重排序失败: {response.message}")
                return documents[:top_k]
        except Exception as e:
            logger.error(f"重排序异常: {e}")
            return documents[:top_k]


class ConversationManager:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []
    
    def add_interaction(self, question: str, answer: str, sources: List[Dict] = None):
        entry = {
            "question": question,
            "answer": answer,
            "sources": sources or []
        }
        self.history.append(entry)
        
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history(self, last_n: int = None) -> List[Dict[str, str]]:
        if last_n is None:
            return self.history
        return self.history[-last_n:]
    
    def clear(self):
        self.history = []


class ShortTermMemory:
    def __init__(self, max_length: int = 10):
        self.max_length = max_length
        self.messages: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        
        if len(self.messages) > self.max_length:
            self.messages = self.messages[-self.max_length:]
    
    def get_messages(self, last_n: int = None) -> List[Dict[str, str]]:
        if last_n is None:
            return self.messages.copy()
        return self.messages[-last_n:].copy()
    
    def get_context_for_prompt(self, last_n: int = 5) -> str:
        recent = self.messages[-last_n:] if last_n > 0 else self.messages
        context_parts = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            context_parts.append(f"{role}: {msg['content']}")
        return "\n".join(context_parts)
    
    def clear(self):
        self.messages = []
    
    def __len__(self):
        return len(self.messages)


class LongTermMemory:
    def __init__(
        self,
        storage_dir: str = "./memory_storage",
        index_name: str = "rag_conversation_history"
    ):
        self.storage_dir = Path(storage_dir)
        self.index_name = index_name
        self.faiss_index = None
        self.conversations: List[Dict[str, Any]] = []
        self.conversation_file = self.storage_dir / f"{index_name}_conversations.json"
        
        os.makedirs(self.storage_dir, exist_ok=True)
        self._init_faiss()
        self._load_conversations()
    
    def _init_faiss(self):
        try:
            dimension = 1536
            self.faiss_index = faiss.IndexFlatL2(dimension)
            logger.info(f"长期记忆Faiss索引初始化成功，维度: {dimension}")
        except Exception as e:
            logger.warning(f"长期记忆Faiss索引初始化失败: {e}")
            self.faiss_index = None
    
    def _load_conversations(self):
        if self.conversation_file.exists():
            try:
                with open(self.conversation_file, "r", encoding="utf-8") as f:
                    self.conversations = json.load(f)
                logger.info(f"加载了 {len(self.conversations)} 条历史对话记录")
            except Exception as e:
                logger.warning(f"加载历史对话失败: {e}")
                self.conversations = []
    
    def _save_conversations(self):
        try:
            with open(self.conversation_file, "w", encoding="utf-8") as f:
                json.dump(self.conversations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存对话历史失败: {e}")
    
    def _get_embedding(self, text: str) -> List[float]:
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
            logger.warning(f"获取embedding失败: {e}")
        return [0.0] * 1536
    
    def save_interaction(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: List[Dict] = None,
        metadata: Dict[str, Any] = None
    ):
        interaction = {
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "sources": sources or [],
            "metadata": metadata or {}
        }
        
        self.conversations.append(interaction)
        
        if self.faiss_index:
            combined_text = f"问: {question} 答: {answer}"
            embedding = self._get_embedding(combined_text)
            import numpy as np
            embedding_array = np.array([embedding], dtype=np.float32)
            self.faiss_index.add(embedding_array)
        
        self._save_conversations()
        logger.info(f"长期记忆保存成功: session_id={session_id}")
    
    def get_session_history(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        session_conversations = [
            c for c in self.conversations 
            if c.get("session_id") == session_id
        ]
        return session_conversations[-limit:]
    
    def search_related(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.faiss_index or not self.conversations:
            return []
        
        try:
            query_embedding = self._get_embedding(query)
            import numpy as np
            query_array = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.faiss_index.search(query_array, min(limit, len(self.conversations)))
            
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.conversations):
                    conv = self.conversations[idx]
                    results.append({
                        "session_id": conv.get("session_id", ""),
                        "question": conv.get("question", ""),
                        "answer": conv.get("answer", ""),
                        "timestamp": conv.get("timestamp", ""),
                        "score": float(distances[0][list(indices[0]).index(idx)])
                    })
            return results
        except Exception as e:
            logger.error(f"搜索长期记忆失败: {e}")
            return []
    
    def clear_session(self, session_id: str):
        self.conversations = [
            c for c in self.conversations 
            if c.get("session_id") != session_id
        ]
        self._save_conversations()
        
        if self.faiss_index:
            self._rebuild_index()
        
        logger.info(f"清除会话记忆成功: session_id={session_id}")
    
    def _rebuild_index(self):
        if not self.faiss_index or not self.conversations:
            return
        
        try:
            self.faiss_index.reset()
            embeddings = []
            for conv in self.conversations:
                combined_text = f"问: {conv.get('question', '')} 答: {conv.get('answer', '')}"
                embedding = self._get_embedding(combined_text)
                embeddings.append(embedding)
            
            if embeddings:
                self.faiss_index.add(embeddings)
        except Exception as e:
            logger.error(f"重建Faiss索引失败: {e}")


class HierarchicalMemorySystem:
    def __init__(
        self,
        short_term_max: int = 10,
        storage_dir: str = "./memory_storage",
        session_id: str = "default"
    ):
        self.session_id = session_id
        self.short_term = ShortTermMemory(max_length=short_term_max)
        self.long_term = LongTermMemory(storage_dir=storage_dir)
        self.summary = ""
    
    def add_interaction(
        self,
        question: str,
        answer: str,
        sources: List[Dict] = None,
        metadata: Dict[str, Any] = None
    ):
        self.short_term.add_message("user", question, metadata)
        self.short_term.add_message("assistant", answer, {"sources": sources})
        
        self.long_term.save_interaction(
            session_id=self.session_id,
            question=question,
            answer=answer,
            sources=sources,
            metadata=metadata
        )
    
    def get_context_for_query(
        self,
        use_long_term: bool = True,
        short_term_count: int = 5,
        long_term_count: int = 3
    ) -> str:
        context_parts = []
        
        if self.summary:
            context_parts.append(f"会话摘要:\n{self.summary}")
        
        short_context = self.short_term.get_context_for_prompt(short_term_count)
        if short_context:
            context_parts.append(f"最近对话:\n{short_context}")
        
        if use_long_term:
            long_term_history = self.long_term.get_session_history(
                self.session_id,
                limit=long_term_count
            )
            if long_term_history:
                history_parts = []
                for item in long_term_history:
                    history_parts.append(
                        f"问: {item['question']}\n答: {item['answer'][:200]}..."
                    )
                context_parts.append(f"历史对话:\n" + "\n".join(history_parts))
        
        return "\n\n".join(context_parts)
    
    def generate_summary(self, api_key: str, model: str = "qwen-turbo") -> str:
        from dashscope import Generation
        
        messages = self.short_term.get_messages()
        if not messages:
            return ""
        
        conversation_text = "\n".join([
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
            for m in messages
        ])
        
        prompt = f"""请总结以下对话的要点，保留关键信息（不超过200字）:

{conversation_text}

摘要:"""
        
        try:
            response = Generation.call(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                result_format="message"
            )
            
            if response.status_code == 200:
                self.summary = response.output.choices[0].message.content.strip()
                return self.summary
        except Exception as e:
            logger.error(f"生成摘要失败: {e}")
        
        return ""
    
    def search_related_history(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        return self.long_term.search_related(query, limit=limit)
    
    def clear(self, clear_long_term: bool = False):
        self.short_term.clear()
        self.summary = ""
        if clear_long_term:
            self.long_term.clear_session(self.session_id)
