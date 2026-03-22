import os
import uuid
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

app = FastAPI(title="LlamaIndex RAG 论文问答系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
OUTPUT_FOLDER = BASE_DIR / 'output'
TEMPLATES_DIR = BASE_DIR / 'templates'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'md'}

rag_instance = None
memory_system = None


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.on_event("startup")
async def startup_event():
    global rag_instance, memory_system
    from llamaindex_rag import HierarchicalMemorySystem
    
    memory_system = HierarchicalMemorySystem(
        short_term_max=10,
        storage_dir=str(BASE_DIR / "memory_storage"),
        session_id="default"
    )
    logger.info("LlamaIndex RAG 应用启动成功")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="未选择文件")

        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="不支持的文件类型")

        file_id = str(uuid.uuid4())
        filename = file.filename
        file_path = str(UPLOAD_FOLDER / f"{file_id}_{filename}")

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"文件上传成功: {file_path}")

        return {
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "file_path": file_path
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/parse")
async def parse_document(data: Dict[str, str] = None):
    try:
        if data is None:
            raise HTTPException(status_code=400, detail="请求体为空")

        file_path = data.get('file_path')

        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="文件不存在")

        from document_parsing import DocumentParser
        parser = DocumentParser(output_dir=str(OUTPUT_FOLDER))

        result = parser.parse_with_headings(file_path)

        content_file = result.get('content_file')
        if not content_file or not os.path.exists(content_file):
            raise HTTPException(status_code=500, detail="文档解析失败")

        return {
            "success": True,
            "content_list": result.get("content_list", []),
            "headings": result.get("headings", []),
            "content_file": content_file
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档解析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/index")
async def index_document(data: Dict[str, str] = None):
    try:
        global rag_instance
        
        if data is None:
            raise HTTPException(status_code=400, detail="请求体为空")

        content_file = data.get('content_file')

        if not content_file or not os.path.exists(content_file):
            raise HTTPException(status_code=400, detail="内容文件不存在")

        from llamaindex_rag import LlamaIndexRAG

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="未配置 DASHSCOPE_API_KEY")

        rag_instance = LlamaIndexRAG(
            api_key=api_key,
            embed_model="text-embedding-v3",
            llm_model="qwen-turbo",
            index_name=data.get('index_name', 'llamaindex_rag_docs')
        )

        documents = rag_instance.load_documents(content_file)
        rag_instance.build_index(documents)

        return {
            "success": True,
            "message": "文档索引成功"
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"文档索引失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(data: Dict[str, Any] = None):
    try:
        global rag_instance, memory_system
        
        if data is None:
            raise HTTPException(status_code=400, detail="请求体为空")

        question = data.get('question', '').strip()
        top_k = data.get('top_k', 5)

        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")

        if rag_instance is None:
            raise HTTPException(status_code=400, detail="请先上传并索引文档")

        from llamaindex_rag import QueryProcessor

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="未配置 DASHSCOPE_API_KEY")

        query_processor = QueryProcessor(api_key=api_key)
        
        context_for_query = memory_system.get_context_for_query(
            use_long_term=True,
            short_term_count=5,
            long_term_count=3
        ) if memory_system else ""
        
        if context_for_query:
            history_context = f"\n\n对话历史上下文：\n{context_for_query}\n"
        else:
            history_context = ""
        
        rewritten_query = query_processor.rewrite_query(question, memory_system.short_term.get_messages() if memory_system else None)

        search_results = rag_instance.retrieve(rewritten_query, similarity_top_k=top_k)

        if not search_results:
            return {
                "success": True,
                "answer": "抱歉，未找到相关内容来回答您的问题。",
                "sources": []
            }

        if len(search_results) > top_k:
            search_results = query_processor.rerank(rewritten_query, search_results, top_k)

        context_parts = []
        for i, result in enumerate(search_results):
            metadata = result.get("metadata", {})
            h1 = metadata.get("h1", "")
            h2 = metadata.get("h2", "")
            h3 = metadata.get("h3", "")

            heading_info = ""
            if h1:
                heading_info = f"【{h1}"
                if h2:
                    heading_info += f" > {h2}"
                if h3:
                    heading_info += f" > {h3}"
                heading_info += "】"

            context_parts.append(f"[文档 {i+1}] {heading_info}\n{result.get('content', '')}")

        context = "\n\n".join(context_parts)

        system_prompt = f"""你是一个专业的学术论文问答助手。请根据以下提供的上下文信息回答用户的问题。

上下文信息来自学术论文，包含章节标题层级结构（H1一级标题、H2二级标题、H3三级标题）。

请按照以下思维链(Chain of Thought)步骤思考：

1. 【理解问题】仔细分析用户的问题，明确问题的主语、限定条件和询问的具体内容
2. 【定位信息】在提供的上下文中搜索与问题相关的关键信息，注意章节标题的指引
3. 【推理分析】基于找到的信息进行逻辑推理，不要仅凭记忆或猜测
4. 【组织答案】整理推理过程，先给出结论，再提供支撑证据
5. 【引用来源】明确标注信息来自哪个章节

{history_context}要求：
1. 只根据提供的上下文信息回答，不要编造信息
2. 如果上下文中没有相关信息，请如实说明"根据提供的信息无法回答该问题"
3. 回答要准确，可以引用论文中的内容
4. 回答要使用学术化的语气
5. 在回答时指明引用信息来源于论文的哪个章节
6. 请展示你的思考过程

上下文：
{context}"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": question})

        from llamaindex_rag import LLMProvider
        llm_provider = LLMProvider(api_key=api_key)
        response = llm_provider.generate(messages)

        sources = []
        for result in search_results:
            metadata = result.get("metadata", {})
            section_path = metadata.get("section_path", "")
            
            sources.append({
                "content": result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
                "section_path": section_path,
                "source_file": metadata.get("source_file", "")
            })

        if memory_system:
            memory_system.add_interaction(question, response.get("content", ""), sources)

        return {
            "success": True,
            "answer": response.get("content", ""),
            "sources": sources,
            "question": question
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(data: Dict[str, Any] = None):
    try:
        global rag_instance, memory_system
        
        if data is None:
            raise HTTPException(status_code=400, detail="请求体为空")

        question = data.get('question', '').strip()
        top_k = data.get('top_k', 5)

        if not question:
            yield "data: {\"type\": \"error\", \"content\": \"问题不能为空\"}\n\n"
            return

        if rag_instance is None:
            yield "data: {\"type\": \"error\", \"content\": \"请先上传并索引文档\"}\n\n"
            return

        from llamaindex_rag import QueryProcessor

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            yield "data: {\"type\": \"error\", \"content\": \"未配置 DASHSCOPE_API_KEY\"}\n\n"
            return

        query_processor = QueryProcessor(api_key=api_key)
        
        context_for_query = memory_system.get_context_for_query(
            use_long_term=True,
            short_term_count=5,
            long_term_count=3
        ) if memory_system else ""
        
        if context_for_query:
            history_context = f"\n\n对话历史上下文：\n{context_for_query}\n"
        else:
            history_context = ""
        
        rewritten_query = query_processor.rewrite_query(question, memory_system.short_term.get_messages() if memory_system else None)

        search_results = rag_instance.retrieve(rewritten_query, similarity_top_k=top_k)

        if not search_results:
            yield "data: {\"type\": \"content\", \"content\": \"抱歉，未找到相关内容来回答您的问题。\"}\n\n"
            return

        if len(search_results) > top_k:
            search_results = query_processor.rerank(rewritten_query, search_results, top_k)

        context_parts = []
        for i, result in enumerate(search_results):
            metadata = result.get("metadata", {})
            h1 = metadata.get("h1", "")
            h2 = metadata.get("h2", "")
            h3 = metadata.get("h3", "")

            heading_info = ""
            if h1:
                heading_info = f"【{h1}"
                if h2:
                    heading_info += f" > {h2}"
                if h3:
                    heading_info += f" > {h3}"
                heading_info += "】"

            context_parts.append(f"[文档 {i+1}] {heading_info}\n{result.get('content', '')}")

        context = "\n\n".join(context_parts)

        system_prompt = f"""你是一个专业的学术论文问答助手。请根据以下提供的上下文信息回答用户的问题。

上下文信息来自学术论文，包含章节标题层级结构（H1一级标题、H2二级标题、H3三级标题）。

请按照以下思维链(Chain of Thought)步骤思考：

1. 【理解问题】仔细分析用户的问题，明确问题的主语、限定条件和询问的具体内容
2. 【定位信息】在提供的上下文中搜索与问题相关的关键信息，注意章节标题的指引
3. 【推理分析】基于找到的信息进行逻辑推理，不要仅凭记忆或猜测
4. 【组织答案】整理推理过程，先给出结论，再提供支撑证据
5. 【引用来源】明确标注信息来自哪个章节

{history_context}要求：
1. 只根据提供的上下文信息回答，不要编造信息
2. 如果上下文中没有相关信息，请如实说明"根据提供的信息无法回答该问题"
3. 回答要准确，可以引用论文中的内容
4. 回答要使用学术化的语气
5. 在回答时指明引用信息来源于论文的哪个章节
6. 请展示你的思考过程

上下文：
{context}"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": question})

        from llamaindex_rag import LLMProvider
        llm_provider = LLMProvider(api_key=api_key)
        
        def send_chunk(content: str):
            import json
            yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"

        stream_response = llm_provider.generate(messages, stream=True)
        logger.info(f"LLM生成器类型: {type(stream_response)}")
        
        full_answer = ""
        sources = []
        
        try:
            chunk_count = 0
            for chunk in stream_response:
                chunk_count += 1
                content = ""
                
                try:
                    if hasattr(chunk, 'output') and chunk.output:
                        if hasattr(chunk.output, 'choices') and chunk.output.choices:
                            choice = chunk.output.choices[0]
                            if hasattr(choice, 'message') and choice.message:
                                if hasattr(choice.message, 'content'):
                                    content = choice.message.content
                            elif hasattr(choice, 'delta') and choice.delta:
                                if hasattr(choice.delta, 'content'):
                                    content = choice.delta.content
                except Exception as e:
                    logger.warning(f"解析chunk失败: {e}")
                    continue
                
                if content:
                    full_answer += content
                    yield f"data: {json.dumps({'type': 'content', 'content': content}, ensure_ascii=False)}\n\n"
            
            logger.info(f"流式循环结束，chunk数量: {chunk_count}, full_answer长度: {len(full_answer)}")
        except Exception as e:
            logger.error(f"LLM流式生成失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        for result in search_results:
            metadata = result.get("metadata", {})
            section_path = metadata.get("section_path", "")
            
            sources.append({
                "content": result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
                "section_path": section_path,
                "source_file": metadata.get("source_file", "")
            })

        if memory_system:
            memory_system.add_interaction(question, full_answer, sources)
        
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    except Exception as e:
        logger.error(f"流式问答失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


@app.post("/api/clear")
async def clear_history():
    global memory_system
    
    if memory_system:
        memory_system.clear(clear_long_term=True)
    
    return {
        "success": True,
        "message": "对话历史已清除"
    }


@app.post("/api/eval/add_dataset")
async def add_eval_dataset(data: Dict[str, Any] = None):
    try:
        if data is None:
            raise HTTPException(status_code=400, detail="请求体为空")

        dataset_name = data.get("dataset_name", "default")
        questions = data.get("questions", [])

        from evaluation import EvalPipeline, EvaluationDataset

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="未配置 DASHSCOPE_API_KEY")

        pipeline = EvalPipeline(api_key=api_key)
        dataset = pipeline.create_dataset(name=dataset_name)

        for q in questions:
            dataset.add_question(
                question=q.get("question", ""),
                ground_truth=q.get("ground_truth", []),
                metadata=q.get("metadata")
            )

        dataset.save(str(BASE_DIR / "eval_datasets" / f"{dataset_name}.json"))

        return {
            "success": True,
            "message": f"数据集 {dataset_name} 添加成功",
            "question_count": len(dataset)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加数据集失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/eval/run")
async def run_evaluation(data: Dict[str, Any] = None):
    try:
        if data is None:
            raise HTTPException(status_code=400, detail="请求体为空")

        dataset_name = data.get("dataset_name", "default")
        top_k = data.get("top_k", 5)
        content_file = data.get("content_file")

        if not content_file or not os.path.exists(content_file):
            raise HTTPException(status_code=400, detail="内容文件不存在")

        from evaluation import EvalPipeline, EvaluationDataset, RAGEvaluator
        from llamaindex_rag import LlamaIndexRAG

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="未配置 DASHSCOPE_API_KEY")

        dataset = EvaluationDataset(str(BASE_DIR / "eval_datasets" / f"{dataset_name}.json"))

        if len(dataset) == 0:
            raise HTTPException(status_code=400, detail="数据集为空或不存在")

        rag = LlamaIndexRAG(
            api_key=api_key,
            embed_model="text-embedding-v3",
            llm_model="qwen-turbo",
            index_name=f"eval_{dataset_name}"
        )

        documents = rag.load_documents(content_file)
        rag.build_index(documents)

        evaluator = RAGEvaluator(rag_instance=rag, eval_dataset=dataset)
        results = evaluator.evaluate_all(
            top_k=top_k,
            save_results=str(BASE_DIR / "eval_results" / f"{dataset_name}_results.json")
        )

        return {
            "success": True,
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"运行评估失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/eval/history")
async def get_eval_history():
    return {
        "success": True,
        "message": "评估历史功能待实现"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
