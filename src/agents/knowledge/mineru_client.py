# src/agents/knowledge/mineru_client.py

from __future__ import annotations
import asyncio
import httpx
from pathlib import Path
from loguru import logger

from src.core.config import get_settings

settings = get_settings()


async def parse_document(
    file_path: str,
    file_name: str | None = None,
    backend: str | None = None,
    return_images: bool = False,
) -> str:
    """
    调用 MinerU API 解析文档，返回 Markdown 文本。
    支持 PDF、Word、图片等格式。

    优先使用异步接口（POST /tasks → 轮询），超大文件不会阻塞。
    """
    base_url = settings.MINERU_API_URL
    backend = backend or settings.MINERU_BACKEND
    timeout = settings.MINERU_TIMEOUT

    if not file_name:
        file_name = Path(file_path).name

    file_bytes = Path(file_path).read_bytes()

    async with httpx.AsyncClient(timeout=timeout) as client:
        # 提交异步解析任务
        resp = await client.post(
            f"{base_url}/tasks",
            files={"files": (file_name, file_bytes)},
            data={
                "backend": backend,
                "return_md": "true",
                "return_images": str(return_images).lower(),
                "formula_enable": "true",
                "table_enable": "true",
            },
        )
        resp.raise_for_status()
        task_data = resp.json()
        task_id = task_data.get("task_id")

        if not task_id:
            logger.warning(f"MinerU 未返回 task_id，尝试同步解析")
            return await _parse_sync(file_path, file_name, backend)

        # 轮询等待完成
        for _ in range(timeout // 2):
            await asyncio.sleep(2)
            status_resp = await client.get(f"{base_url}/tasks/{task_id}")
            status_resp.raise_for_status()
            status_data = status_resp.json()
            status = status_data.get("status", "")

            if status == "completed":
                result_resp = await client.get(f"{base_url}/tasks/{task_id}/result")
                if result_resp.status_code == 200:
                    result = result_resp.json()
                    return _extract_markdown(result)
                break
            elif status == "failed":
                error = status_data.get("error", "未知错误")
                logger.error(f"MinerU 解析失败: {error}")
                raise RuntimeError(f"MinerU 解析失败: {error}")

        raise TimeoutError(f"MinerU 解析超时 ({timeout}s)")


async def _parse_sync(
    file_path: str,
    file_name: str,
    backend: str,
) -> str:
    """同步解析（兜底方案）。"""
    base_url = settings.MINERU_API_URL
    file_bytes = Path(file_path).read_bytes()

    async with httpx.AsyncClient(timeout=settings.MINERU_TIMEOUT) as client:
        resp = await client.post(
            f"{base_url}/file_parse",
            files={"files": (file_name, file_bytes)},
            data={
                "backend": backend,
                "return_md": "true",
                "formula_enable": "true",
                "table_enable": "true",
            },
        )
        resp.raise_for_status()
        return _extract_markdown(resp.json())


def _extract_markdown(result: dict) -> str:
    """从 MinerU 响应中提取 Markdown 文本。"""
    # MinerU 返回格式：{"results": [{"md": "...", ...}]}
    results = result.get("results", [])
    if results:
        first = results[0]
        if isinstance(first, dict):
            md = first.get("md", "")
            if md:
                return md
    # 兜底：尝试直接取 md 字段
    if "md" in result:
        return result["md"]
    # 再兜底：整个 result 转字符串
    import json
    return json.dumps(result, ensure_ascii=False)


async def check_mineru_health() -> dict:
    """检查 MinerU 服务健康状态。"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{settings.MINERU_API_URL}/health")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}