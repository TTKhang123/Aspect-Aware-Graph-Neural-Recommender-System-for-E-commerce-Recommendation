from __future__ import annotations

import re
from typing import Any, Dict

import numpy as np


def clean_text(x: Any) -> str:
    if x is None:
        return ""

    if isinstance(x, float) and np.isnan(x):
        return ""

    if isinstance(x, (list, tuple)):
        return " ".join(clean_text(v) for v in x)

    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            parts.append(str(k))
            parts.append(clean_text(v))
        return " ".join(parts)

    s = str(x)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_item_content_text(row: Dict[str, Any], max_chars: int = 3000) -> str:
    title = clean_text(row.get("title"))
    brand = clean_text(row.get("brand"))
    categories = clean_text(row.get("categories"))
    feature = clean_text(row.get("feature"))
    description = clean_text(row.get("description"))
    price = clean_text(row.get("price"))
    sales_rank = clean_text(row.get("salesRank"))
    similar = clean_text(row.get("similar"))

    text = f"""
    Title: {title}
    Brand: {brand}
    Categories: {categories}
    Features: {feature}
    Description: {description}
    Price: {price}
    SalesRank: {sales_rank}
    Similar: {similar}
    """

    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]
