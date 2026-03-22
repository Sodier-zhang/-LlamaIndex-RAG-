import os
import re
import json

from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger


class DocumentParser:
    def __init__(
        self,
        output_dir: str = "./output",
        language: str = "en",
        llamaparse_api_key: str = None,
        use_proxy: bool = False,
        proxy_url: str = None
    ):
        self.output_dir = output_dir
        self.language = language
        self.llamaparse_api_key = llamaparse_api_key or os.getenv("LLAMAPARSE_API_KEY")
        if not self.llamaparse_api_key:
            raise ValueError("需要设置 LLAMAPARSE_API_KEY")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.use_proxy = use_proxy
        self.proxy_url = proxy_url or os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")

    def parse_pdf(
        self,
        pdf_path: str
    ) -> dict:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        file_name = pdf_path.stem
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"开始解析 PDF: {pdf_path}")

        full_text = self._parse_with_llamaparse(pdf_path)
        logger.info(f"LlamaParse 解析完成, 文本长度: {len(full_text)}")
        
        md_file = output_dir / f"{file_name}.txt"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(full_text)

        content_blocks = self._parse_text_to_blocks(full_text)
        logger.info(f"解析得到 {len(content_blocks)} 个内容块")

        content_file = output_dir / f"{file_name}_content_list.json"
        with open(content_file, "w", encoding="utf-8") as f:
            json.dump({"content_list": content_blocks}, f, ensure_ascii=False, indent=2)

        logger.info(f"PDF 解析完成: {file_name}")

        return {
            "file_name": file_name,
            "output_dir": str(output_dir),
            "md_file": str(md_file),
            "content_list_file": str(content_file),
            "toc_file": None
        }

    def _parse_with_llamaparse(self, pdf_path: Path) -> str:
        import os
        saved_http = os.environ.get("HTTP_PROXY")
        saved_https = os.environ.get("HTTPS_PROXY")
        
        if self.use_proxy and self.proxy_url:
            os.environ["HTTP_PROXY"] = self.proxy_url
            os.environ["HTTPS_PROXY"] = self.proxy_url
            logger.info(f"设置代理: {self.proxy_url}")
        
        try:
            from llama_parse import LlamaParse
        except ImportError:
            if self.use_proxy and self.proxy_url:
                os.environ["HTTP_PROXY"] = saved_http or ""
                os.environ["HTTPS_PROXY"] = saved_https or ""
            raise ImportError("请安装 llama-parse: pip install llama-parse")

        logger.info(f"LlamaParse API key: {self.llamaparse_api_key[:10]}..." if self.llamaparse_api_key else "API key is None")
        
        parser = LlamaParse(
            api_key=self.llamaparse_api_key,
            result_type="markdown",
            num_workers=4,
            verbose=True
        )

        try:
            documents = parser.load_data(str(pdf_path))
            logger.info(f"LlamaParse loaded {len(documents)} documents")
        except Exception as e:
            logger.error(f"LlamaParse load_data failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            if self.use_proxy and self.proxy_url:
                if saved_http:
                    os.environ["HTTP_PROXY"] = saved_http
                else:
                    os.environ.pop("HTTP_PROXY", None)
                if saved_https:
                    os.environ["HTTPS_PROXY"] = saved_https
                else:
                    os.environ.pop("HTTPS_PROXY", None)

        md_content = ""
        for doc in documents:
            md_content += doc.text + "\n\n"

        return md_content

    def _parse_text_to_blocks(
        self, 
        text: str
    ) -> List[Dict[str, Any]]:
        blocks = []
        
        current_h1 = ""
        current_h2 = ""
        current_h3 = ""
        
        heading_positions = []
        
        lines = text.split('\n')
        
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        number_pattern = r'^(\d+(?:\.\d+)*)\s+(.+)$'
        uppercase_pattern = r'^([A-Z])\s+(.+)$'
        
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            
            heading_match = re.match(heading_pattern, stripped)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                heading_positions.append({
                    "index": idx,
                    "type": "markdown",
                    "level": level,
                    "text": heading_text
                })
                continue
            
            number_match = re.match(number_pattern, stripped)
            if number_match and len(stripped) < 100:
                heading_text = number_match.group(2).strip()
                number_str = number_match.group(1)
                
                if re.match(r'^\d+$', number_str):
                    heading_positions.append({
                        "index": idx,
                        "type": "number",
                        "level": 1,
                        "text": heading_text
                    })
                elif re.match(r'^\d+\.\d+$', number_str):
                    heading_positions.append({
                        "index": idx,
                        "type": "number",
                        "level": 2,
                        "text": heading_text
                    })
                continue
            
            uppercase_match = re.match(uppercase_pattern, stripped)
            if uppercase_match and len(stripped) < 50:
                heading_positions.append({
                    "index": idx,
                    "type": "uppercase",
                    "level": 1,
                    "text": uppercase_match.group(2).strip()
                })
        
        heading_positions.sort(key=lambda x: x["index"])
        
        sections = []
        for i, hp in enumerate(heading_positions):
            start_idx = hp["index"]
            end_idx = heading_positions[i + 1]["index"] if i + 1 < len(heading_positions) else len(lines)
            
            section_lines = []
            for j in range(start_idx + 1, end_idx):
                if j < len(lines):
                    section_lines.append(lines[j])
            
            sections.append({
                "heading": hp,
                "lines": section_lines
            })
        
        for section in sections:
            hp = section["heading"]
            level = hp["level"]
            heading_text = hp["text"]
            
            if level == 1:
                current_h1 = heading_text
                current_h2 = ""
                current_h3 = ""
            elif level == 2:
                current_h2 = heading_text
                current_h3 = ""
            elif level >= 3:
                current_h3 = heading_text
            
            heading_context = {
                "h1": current_h1,
                "h2": current_h2,
                "h3": current_h3
            }
            
            section_text = "\n".join(section["lines"])
            
            heading_id = f"h_{level}_{heading_text[:20]}"
            
            section_path = current_h1
            if current_h2:
                section_path += f" > {current_h2}"
            if current_h3:
                section_path += f" > {current_h3}"
            
            blocks.append({
                "type": "heading",
                "level": level,
                "content": heading_text,
                "heading_context": heading_context,
                "doc_id": heading_id,
                "parent_id": None,
                "chunk_level": "section",
                "section_path": section_path
            })
            
            if section_text.strip():
                blocks.append({
                    "type": "text",
                    "level": level,
                    "content": section_text.strip(),
                    "heading_context": heading_context,
                    "doc_id": f"text_{level}_{heading_text[:20]}",
                    "parent_id": heading_id,
                    "chunk_level": "content",
                    "section_path": section_path
                })
        
        return blocks
    
    def _split_by_punctuation(self, text: str, max_length: int = 500) -> List[str]:
        if not text:
            return []
        
        punctuation = r'[。！？；：\.\!\?\;\:]'
        
        parts = re.split(punctuation, text)
        
        results = []
        current_chunk = ""
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if len(current_chunk) + len(part) <= max_length:
                if current_chunk:
                    current_chunk += "。" + part
                else:
                    current_chunk = part
            else:
                if current_chunk:
                    results.append(current_chunk)
                if len(part) <= max_length:
                    current_chunk = part
                else:
                    sub_parts = self._split_long_text(part, max_length)
                    results.extend(sub_parts[:-1])
                    current_chunk = sub_parts[-1] if sub_parts else ""
        
        if current_chunk:
            results.append(current_chunk)
        
        return results
    
    def _split_long_text(self, text: str, max_length: int = 500) -> List[str]:
        if len(text) <= max_length:
            return [text]
        
        results = []
        sentences = re.split(r'[。！？；：\.\!\?\;\:]', text)
        
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current) + len(sentence) <= max_length:
                current = sentence if not current else current + "。" + sentence
            else:
                if current:
                    results.append(current)
                if len(sentence) <= max_length:
                    current = sentence
                else:
                    words = sentence.split()
                    current = ""
                    for word in words:
                        if len(current) + len(word) + 1 <= max_length:
                            current = word if not current else current + " " + word
                        else:
                            if current:
                                results.append(current)
                            current = word
        
        if current:
            results.append(current)
        
        return results

    def parse_with_headings(
        self, 
        pdf_path: str
    ) -> dict:
        result = self.parse_pdf(pdf_path)

        md_file = Path(result["md_file"])
        content_file = Path(result["content_list_file"])

        if not md_file.exists() or not content_file.exists():
            return {"content_list": [], "headings": []}

        with open(content_file, "r", encoding="utf-8") as f:
            content_data = json.load(f)

        content_items = content_data.get("content_list", [])

        headings = []
        for item in content_items:
            if item.get("type") == "heading":
                headings.append({
                    "level": item.get("level", 1),
                    "title": item.get("content", ""),
                    "h1": item.get("heading_context", {}).get("h1", ""),
                    "h2": item.get("heading_context", {}).get("h2", ""),
                    "h3": item.get("heading_context", {}).get("h3", "")
                })

        return {
            "content_list": content_items,
            "headings": headings,
            "content_file": result["content_list_file"],
            "md_file": result["md_file"]
        }

    def get_text_content(self, content_file: str) -> str:
        with open(content_file, "r", encoding="utf-8") as f:
            content_data = json.load(f)

        content_items = content_data.get("content_list", [])
        text_parts = []

        for item in content_items:
            if item.get("type") in ["text", "heading"]:
                content = item.get("content", "")
                heading_context = item.get("heading_context", {})
                
                h1 = heading_context.get("h1", "")
                h2 = heading_context.get("h2", "")
                h3 = heading_context.get("h3", "")

                heading_info = ""
                if h1:
                    heading_info = f"【{h1}"
                    if h2:
                        heading_info += f" > {h2}"
                    if h3:
                        heading_info += f" > {h3}"
                    heading_info += "】\n"

                text_parts.append(heading_info + content)

        return "\n\n".join(text_parts)
