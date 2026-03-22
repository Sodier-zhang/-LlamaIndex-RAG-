import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger


class EvaluationDataset:
    def __init__(self, data_file: str = None):
        self.data_file = data_file
        self.questions: List[Dict[str, Any]] = []
        
        if data_file and Path(data_file).exists():
            self.load(data_file)
    
    def add_question(
        self,
        question: str,
        ground_truth: List[str],
        metadata: Dict[str, Any] = None
    ):
        self.questions.append({
            "question": question,
            "ground_truth": ground_truth,
            "metadata": metadata or {}
        })
    
    def load(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.questions = data.get("questions", [])
    
    def save(self, file_path: str = None):
        save_path = file_path or self.data_file
        if not save_path:
            raise ValueError("未指定保存路径")
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"questions": self.questions}, f, ensure_ascii=False, indent=2)
    
    def __len__(self):
        return len(self.questions)


class RAGEvaluator:
    def __init__(
        self,
        rag_instance,
        eval_dataset: EvaluationDataset = None
    ):
        self.rag_instance = rag_instance
        self.eval_dataset = eval_dataset or EvaluationDataset()
        self.results: List[Dict[str, Any]] = []
    
    def _compute_hit_rate(self, retrieved_docs: List[str], ground_truth: List[str]) -> bool:
        for doc in retrieved_docs:
            doc_lower = doc.lower()
            for truth in ground_truth:
                if truth.lower() in doc_lower:
                    return True
        return False
    
    def _compute_mrr(self, retrieved_docs: List[str], ground_truth: List[str]) -> float:
        for idx, doc in enumerate(retrieved_docs):
            doc_lower = doc.lower()
            for truth in ground_truth:
                if truth.lower() in doc_lower:
                    return 1.0 / (idx + 1)
        return 0.0
    
    def evaluate_single(
        self,
        question: str,
        ground_truth: List[str],
        top_k: int = 5
    ) -> Dict[str, Any]:
        if not self.rag_instance:
            raise ValueError("RAG实例未初始化")
        
        try:
            search_results = self.rag_instance.retrieve(question, similarity_top_k=top_k)
            
            retrieved_contents = [
                result.get("content", "") for result in search_results
            ]
            
            hit = self._compute_hit_rate(retrieved_contents, ground_truth)
            mrr = self._compute_mrr(retrieved_contents, ground_truth)
            
            return {
                "question": question,
                "ground_truth": ground_truth,
                "retrieved": retrieved_contents,
                "hit": hit,
                "mrr": mrr,
                "hit_at_k": hit,
                "mrr_at_k": mrr,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"评估单个问题失败: {e}")
            return {
                "question": question,
                "ground_truth": ground_truth,
                "error": str(e),
                "hit": False,
                "mrr": 0.0
            }
    
    def evaluate_all(
        self,
        top_k: int = 5,
        save_results: str = None
    ) -> Dict[str, Any]:
        if not self.eval_dataset.questions:
            return {
                "hit_rate": 0.0,
                "mrr": 0.0,
                "total": 0,
                "hits": 0,
                "details": []
            }
        
        results = []
        hits = 0
        mrr_sum = 0.0
        
        for q in self.eval_dataset.questions:
            question = q.get("question", "")
            ground_truth = q.get("ground_truth", [])
            
            result = self.evaluate_single(question, ground_truth, top_k=top_k)
            results.append(result)
            
            if result.get("hit"):
                hits += 1
            mrr_sum += result.get("mrr", 0.0)
        
        total = len(results)
        hit_rate = hits / total if total > 0 else 0.0
        mrr = mrr_sum / total if total > 0 else 0.0
        
        eval_results = {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "total": total,
            "hits": hits,
            "misses": total - hits,
            "top_k": top_k,
            "details": results,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results = results
        
        if save_results:
            with open(save_results, "w", encoding="utf-8") as f:
                json.dump(eval_results, f, ensure_ascii=False, indent=2)
            logger.info(f"评估结果已保存到: {save_results}")
        
        return eval_results
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {
                "hit_rate": 0.0,
                "mrr": 0.0,
                "total": 0
            }
        
        hits = sum(1 for r in self.results if r.get("hit"))
        mrr_sum = sum(r.get("mrr", 0.0) for r in self.results)
        total = len(self.results)
        
        return {
            "hit_rate": hits / total if total > 0 else 0.0,
            "mrr": mrr_sum / total if total > 0 else 0.0,
            "total": total,
            "hits": hits,
            "misses": total - hits
        }


class EvalPipeline:
    def __init__(
        self,
        api_key: str,
        embed_model: str = "text-embedding-v3",
        llm_model: str = "qwen-turbo"
    ):
        self.api_key = api_key
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.eval_datasets: Dict[str, EvaluationDataset] = {}
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def create_dataset(
        self,
        name: str,
        data_file: str = None
    ) -> EvaluationDataset:
        dataset = EvaluationDataset(data_file=data_file)
        self.eval_datasets[name] = dataset
        return dataset
    
    def add_evaluation_sample(
        self,
        dataset_name: str,
        question: str,
        ground_truth: List[str],
        metadata: Dict[str, Any] = None
    ):
        if dataset_name not in self.eval_datasets:
            self.create_dataset(dataset_name)
        
        self.eval_datasets[dataset_name].add_question(
            question=question,
            ground_truth=ground_truth,
            metadata=metadata
        )
    
    def run_evaluation(
        self,
        dataset_name: str,
        rag_instance,
        top_k: int = 5,
        save_results: str = None
    ) -> Dict[str, Any]:
        if dataset_name not in self.eval_datasets:
            raise ValueError(f"数据集 {dataset_name} 不存在")
        
        dataset = self.eval_datasets[dataset_name]
        evaluator = RAGEvaluator(rag_instance=rag_instance, eval_dataset=dataset)
        
        results = evaluator.evaluate_all(top_k=top_k, save_results=save_results)
        
        eval_record = {
            "dataset_name": dataset_name,
            "top_k": top_k,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        self.evaluation_history.append(eval_record)
        
        return results
    
    def compare_strategies(
        self,
        dataset_name: str,
        rag_configs: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if dataset_name not in self.eval_datasets:
            raise ValueError(f"数据集 {dataset_name} 不存在")
        
        from llamaindex_rag import LlamaIndexRAG
        
        comparison_results = []
        
        for config in rag_configs:
            config_name = config.get("name", "default")
            logger.info(f"评估策略: {config_name}")
            
            try:
                rag = LlamaIndexRAG(
                    api_key=self.api_key,
                    embed_model=config.get("embed_model", self.embed_model),
                    llm_model=config.get("llm_model", self.llm_model),
                    index_name=config.get("index_name", "eval_index")
                )
                
                documents = rag.load_documents(config.get("content_file"))
                rag.build_index(documents)
                
                evaluator = RAGEvaluator(rag_instance=rag)
                results = evaluator.evaluate_all(top_k=top_k)
                
                comparison_results.append({
                    "strategy_name": config_name,
                    "config": config,
                    "hit_rate": results.get("hit_rate", 0.0),
                    "mrr": results.get("mrr", 0.0),
                    "total": results.get("total", 0)
                })
            except Exception as e:
                logger.error(f"评估策略 {config_name} 失败: {e}")
                comparison_results.append({
                    "strategy_name": config_name,
                    "config": config,
                    "error": str(e),
                    "hit_rate": 0.0,
                    "mrr": 0.0
                })
        
        comparison_results.sort(key=lambda x: x.get("hit_rate", 0), reverse=True)
        
        return comparison_results
    
    def get_history(self) -> List[Dict[str, Any]]:
        return self.evaluation_history
    
    def save_dataset(self, dataset_name: str, file_path: str = None):
        if dataset_name not in self.eval_datasets:
            raise ValueError(f"数据集 {dataset_name} 不存在")
        
        self.eval_datasets[dataset_name].save(file_path)
