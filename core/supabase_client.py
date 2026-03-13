import logging
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from core.config import settings

logger = logging.getLogger(__name__)

_client = None


class SupabaseClient:
    """Wrapper around supabase-py client with retry logic."""

    def __init__(self, client):
        self._client = client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def table_select(self, table: str, query: str = "*", limit: int = 100, order_by: Optional[str] = None, filters: Optional[dict] = None):
        try:
            q = self._client.table(table).select(query)
            if filters:
                for key, value in filters.items():
                    q = q.eq(key, value)
            if order_by:
                q = q.order(order_by, desc=True)
            q = q.limit(limit)
            result = q.execute()
            return result.data
        except Exception as e:
            logger.error(f"Supabase select error on {table}: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def table_insert(self, table: str, data: dict):
        try:
            result = self._client.table(table).insert(data).execute()
            return result.data
        except Exception as e:
            logger.error(f"Supabase insert error on {table}: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def table_upsert(self, table: str, data: dict):
        try:
            result = self._client.table(table).upsert(data).execute()
            return result.data
        except Exception as e:
            logger.error(f"Supabase upsert error on {table}: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def rpc(self, function_name: str, params: Optional[dict] = None):
        try:
            result = self._client.rpc(function_name, params or {}).execute()
            return result.data
        except Exception as e:
            logger.error(f"Supabase RPC error on {function_name}: {e}")
            raise

    def raw(self):
        return self._client


def init_client() -> SupabaseClient:
    global _client
    from supabase import create_client
    raw = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    _client = SupabaseClient(raw)
    return _client


def get_client() -> SupabaseClient:
    if _client is None:
        return init_client()
    return _client
