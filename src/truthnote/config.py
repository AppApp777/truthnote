from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    llm_provider: str = "claude_cli"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"

    openai_api_key: str = ""
    openai_base_url: str = ""
    default_model: str = ""

    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-max"

    linuxdo_api_key: str = ""

    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-v3"

    qihoo_api_key: str = ""
    qihoo_base_url: str = "https://api.360.cn"

    search_provider: str = "mock"
    tavily_api_key: str = ""
    firecrawl_api_key: str = ""
    bocha_api_key: str = ""  # 博查搜索（中文权威源召回好，证实维度命脉）
    search_cache_db: str = "data/search_cache.db"

    # 官方辟谣库证据检索（debunk_index）：命中只进证据、不出判定（见 CONTRACTS.md INV-4）。
    # demo / 评委环节可在 .env 设 ENABLE_DEBUNK_INDEX=false 一键关，行为退回纯流式取证。
    enable_debunk_index: bool = True

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True

    def get_llm_config(self) -> dict:
        if self.llm_provider == "dashscope":
            return {
                "api_key": self.dashscope_api_key,
                "base_url": self.dashscope_base_url,
                "model": self.dashscope_model,
            }
        return {
            "api_key": self.llm_api_key,
            "base_url": self.llm_base_url,
            "model": self.llm_model,
        }


settings = Settings()
