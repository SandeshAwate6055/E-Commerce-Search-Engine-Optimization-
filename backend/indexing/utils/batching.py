"""Batching utilities"""
from typing import List, Iterator, TypeVar

T = TypeVar('T')


def create_batches(items: List[T], batch_size: int) -> Iterator[List[T]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
