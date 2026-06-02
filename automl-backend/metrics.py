# ============================================================
# AutoML Studio — Prometheus Metrics & Middleware
# ============================================================
import os
import re
import time
import logging
from typing import Dict
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Gauge, Histogram

# --- METRICS DEFINITIONS ---

# 1. Infrastructure Metrics
INFRA_CPU = Gauge(
    "infra_cpu_usage_percent",
    "System-wide CPU usage percentage"
)

INFRA_RAM = Gauge(
    "infra_ram_usage_percent",
    "System-wide RAM usage percentage"
)

INFRA_DISK = Gauge(
    "infra_disk_usage_percent",
    "System-wide Disk usage percentage"
)

INFRA_NET_SENT = Gauge(
    "infra_network_bytes_sent_total",
    "System-wide cumulative network bytes sent"
)

INFRA_NET_RECV = Gauge(
    "infra_network_bytes_recv_total",
    "System-wide cumulative network bytes received"
)

# 2. Application Metrics
API_REQUESTS = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "path", "status_code"]
)

API_ERRORS = Counter(
    "api_errors_total",
    "Total number of API errors (status code >= 400)",
    ["method", "path", "status_code"]
)

API_RESPONSE_TIME = Histogram(
    "api_response_time_seconds",
    "API response time in seconds",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf"))
)

ACTIVE_USERS = Gauge(
    "app_active_users",
    "Number of unique active authenticated users in the last 5 minutes"
)

# 3. Security Metrics
FAILED_LOGINS = Counter(
    "security_failed_logins_total",
    "Total number of failed login attempts"
)

AUTH_ERRORS = Counter(
    "security_api_auth_errors_total",
    "Total number of API authorization errors (invalid/expired/missing JWT on protected routes)"
)


# --- HELPERS ---

# Registry for sliding-window active users: user_id -> last_seen_timestamp
_active_users_registry: Dict[str, float] = {}


def track_active_user(user_id: str) -> None:
    """Refreshes the last-seen timestamp of an authenticated user."""
    if user_id:
        _active_users_registry[user_id] = time.time()


def update_active_users_count() -> None:
    """Prunes expired users (> 5 minutes) and updates the Prometheus gauge."""
    now = time.time()
    cutoff = now - 300  # 5 minutes
    
    # Prune
    expired = [uid for uid, t in _active_users_registry.items() if t < cutoff]
    for uid in expired:
        _active_users_registry.pop(uid, None)
        
    ACTIVE_USERS.set(len(_active_users_registry))


def update_infra_metrics() -> None:
    """Updates physical system-level telemetry using psutil."""
    import psutil
    
    # 1. CPU Usage
    INFRA_CPU.set(psutil.cpu_percent(interval=None))
    
    # 2. RAM Usage
    INFRA_RAM.set(psutil.virtual_memory().percent)
    
    # 3. Disk Usage
    try:
        INFRA_DISK.set(psutil.disk_usage(os.getcwd()).percent)
    except Exception as exc:
        logging.warning("Could not read disk usage percentage: %s", exc)
        INFRA_DISK.set(0.0)
        
    # 4. Network Throughput
    try:
        net_io = psutil.net_io_counters()
        INFRA_NET_SENT.set(net_io.bytes_sent)
        INFRA_NET_RECV.set(net_io.bytes_recv)
    except Exception as exc:
        logging.warning("Could not read network cumulative bytes: %s", exc)
        INFRA_NET_SENT.set(0.0)
        INFRA_NET_RECV.set(0.0)


def _normalize_path(path: str) -> str:
    """
    Normalizes HTTP paths to reduce label cardinality in Prometheus.
    Examples:
        /datasets/310a0e5f-149b-4395-9bf6-b51f786de08d -> /datasets/{id}
        /auth/users/2 -> /auth/users/{id}
    """
    # Replace UUIDs (8-4-4-4-12 format) or hexadecimal hashes/IDs (8+ hex characters)
    path = re.sub(r'/[0-9a-fA-F\-]{8,}', '/{id}', path)
    # Replace numbers
    path = re.sub(r'/\d+', '/{id}', path)
    return path


# --- MIDDLEWARE ---

class MetricsMiddleware(BaseHTTPMiddleware):
    """
    FastAPI / ASGI Middleware that intercepts API requests
    to capture response times, request counts, and HTTP errors.
    """
    async def dispatch(self, request: Request, call_next):
        # Exclude /metrics path to prevent scrape self-noise
        if request.url.path == "/metrics":
            return await call_next(request)
            
        start_time = time.time()
        method = request.method
        path = request.url.path
        norm_path = _normalize_path(path)
        
        try:
            response = await call_next(request)
            
            # Record telemetry
            duration = time.time() - start_time
            status_code = str(response.status_code)
            
            API_RESPONSE_TIME.labels(method=method, path=norm_path).observe(duration)
            API_REQUESTS.labels(method=method, path=norm_path, status_code=status_code).inc()
            
            if response.status_code >= 400:
                API_ERRORS.labels(method=method, path=norm_path, status_code=status_code).inc()
                
            return response
            
        except Exception as exc:
            # Fallback for unhandled server exceptions (HTTP 500)
            duration = time.time() - start_time
            API_RESPONSE_TIME.labels(method=method, path=norm_path).observe(duration)
            API_REQUESTS.labels(method=method, path=norm_path, status_code="500").inc()
            API_ERRORS.labels(method=method, path=norm_path, status_code="500").inc()
            raise exc
