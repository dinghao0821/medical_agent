"""Shared infrastructure services (P2): Redis client, cache, rate limiter, object storage.

Every service in this package is designed to degrade gracefully: when its
backing infrastructure (Redis / object storage) is unavailable, it either
no-ops or falls back to a safe local behaviour, so the application remains
runnable in a plain local environment without Redis/MinIO.
"""
