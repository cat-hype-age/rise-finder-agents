import asyncio
import json
import logging
import random
import time
from collections import deque
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from core.config import settings, update_agent_last_run
from core import config

logger = logging.getLogger(__name__)

gpu_collector: Optional["GPUMetricsCollector"] = None


class GPUMetricsCollector:
    def __init__(self):
        self._handle = None
        self._readings: deque = deque(maxlen=120)
        self._write_counter = 0
        self._peak_session = 0.0
        self._running = False

    def init_gpu(self):
        """Try to initialize NVIDIA GPU monitoring."""
        if settings.GPU_MOCK_MODE:
            logger.info("GPU metrics: running in MOCK mode")
            config.GPU_AVAILABLE = False
            return

        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            config.GPU_AVAILABLE = True
            device_name = pynvml.nvmlDeviceGetName(self._handle)
            logger.info(f"GPU initialized: {device_name}")
        except Exception as e:
            logger.info(f"No GPU available ({e}), will use mock metrics")
            config.GPU_AVAILABLE = False

    def _read_real_metrics(self) -> dict:
        """Read actual GPU metrics via pynvml."""
        import pynvml
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except Exception:
            power = 0.0

        return {
            "gpu_util_pct": util.gpu,
            "vram_used_mb": round(mem.used / (1024 * 1024), 1),
            "vram_total_mb": round(mem.total / (1024 * 1024), 1),
            "power_draw_w": round(power, 1),
            "temperature_c": temp,
            "is_mock": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _generate_mock_metrics(self) -> dict:
        """Generate realistic mock GPU metrics."""
        from agents.memo_generator import memo_in_flight

        if memo_in_flight > 0:
            util = max(70, min(95, random.gauss(82, 8)))
            vram_used = 8000 + (util * 120) + random.gauss(0, 200)
        else:
            util = max(10, min(30, random.gauss(20, 5)))
            vram_used = 4000 + random.gauss(0, 200)

        util = round(util, 1)
        vram_used = round(max(2000, vram_used), 1)
        temp = round(35 + (util * 0.55) + random.gauss(0, 2), 1)
        power = round(50 + (util * 2.5) + random.gauss(0, 5), 1)

        return {
            "gpu_util_pct": util,
            "vram_used_mb": vram_used,
            "vram_total_mb": 24576.0,
            "power_draw_w": max(30, power),
            "temperature_c": max(25, min(95, temp)),
            "is_mock": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _rolling_avg(self, minutes: float = 1.0) -> float:
        """Calculate rolling average GPU utilization."""
        if not self._readings:
            return 0.0
        cutoff = time.time() - (minutes * 60)
        recent = [
            r["gpu_util_pct"] for r in self._readings
            if datetime.fromisoformat(r["timestamp"]).timestamp() > cutoff
        ]
        return round(sum(recent) / len(recent), 1) if recent else 0.0

    async def collect_loop(self):
        """Main collection loop — runs every 1.5 seconds."""
        self._running = True
        logger.info("GPU metrics collection loop started")

        while self._running:
            try:
                if config.GPU_AVAILABLE and self._handle:
                    metrics = self._read_real_metrics()
                else:
                    metrics = self._generate_mock_metrics()

                # Track peak
                if metrics["gpu_util_pct"] > self._peak_session:
                    self._peak_session = metrics["gpu_util_pct"]

                metrics["rolling_avg_1min"] = self._rolling_avg()
                metrics["peak_session"] = self._peak_session

                self._readings.append(metrics)
                self._write_counter += 1

                # Write to DB every 3rd reading (~4.5s)
                if self._write_counter % 3 == 0:
                    try:
                        from core.supabase_client import get_client
                        sb = get_client()
                        await sb.table_insert("gpu_metrics", {
                            "gpu_util_pct": metrics["gpu_util_pct"],
                            "vram_used_mb": metrics["vram_used_mb"],
                            "vram_total_mb": metrics["vram_total_mb"],
                            "power_draw_w": metrics["power_draw_w"],
                            "temperature_c": metrics["temperature_c"],
                        })
                    except Exception as e:
                        logger.debug(f"GPU metrics DB write skipped: {e}")

                update_agent_last_run("gpu_metrics")

            except Exception as e:
                logger.error(f"GPU metrics collection error: {e}")

            await asyncio.sleep(1.5)

    async def stream(self) -> AsyncGenerator[str, None]:
        """SSE generator yielding GPU metrics every 1.5 seconds."""
        while True:
            if self._readings:
                latest = self._readings[-1].copy()
                latest["rolling_avg_1min"] = self._rolling_avg()
                latest["peak_session"] = self._peak_session
                yield json.dumps(latest)
            else:
                yield json.dumps({"status": "waiting_for_data"})
            await asyncio.sleep(1.5)

    def stop(self):
        self._running = False
