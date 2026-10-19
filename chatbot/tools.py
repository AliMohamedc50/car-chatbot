"""
tools.py - الأدوات (tools) اللي الموديل بينادي عليها + الكود اللي بينفّذها.

الموديل مبيشوفش الإكسيل مباشرة. هو بيطلب "أداة" باسمها ومعاملاتها، واحنا بننفّذ البحث ونرجّعله النتيجة،
وهو بيصيغ الرد منها. كده ضمنّا إن الأسعار والمواصفات والحلول جاية من الداتا بس.

تعريفات الأدوات هنا بصيغة محايدة (name / description / parameters = JSON Schema)،
وملف llm.py هو اللي بيحوّلها لصيغة Gemini.
"""
from __future__ import annotations

import copy
import json
import logging
import threading
from collections import OrderedDict

from .retriever import (CONFIDENCE_THRESHOLD, FEATURE_COLUMNS, SORT_OPTIONS, FaultRetriever,
                        search_cars)

log = logging.getLogger("carbot.tools")

# ---------- وصف الأدوات (JSON Schema) اللي بنبعته للموديل ----------
TOOL_SCHEMAS = [
    {
        "name": "search_cars",
        "description": (
            "Search the cars database (the ONLY source of car prices/specs). All filters are optional. "
            "Prices are in EGP. Brand/model can be given in English or Arabic (e.g. 'Toyota' or 'تويوتا'). "
            "If nothing matches, the tool automatically returns closest alternatives with relaxed=true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "min_price": {"type": "number", "description": "Minimum price in EGP"},
                "max_price": {"type": "number", "description": "Maximum price in EGP (the user's budget)"},
                "brand": {"type": "string", "description": "Brand, English or Arabic"},
                "model": {"type": "string", "description": "Model, English or Arabic"},
                "body_type": {"type": "string", "description": "Body type, e.g. sedan/SUV/هاتشباك"},
                "condition": {"type": "string", "description": "'جديد' (new) or 'مستعمل' (used)"},
                "min_year": {"type": "integer", "description": "Minimum model year"},
                "transmission": {"type": "string", "description": "automatic/manual (أوتوماتيك/مانيوال)"},
                "fuel_type": {"type": "string", "description": "e.g. بنزين / سولار / هايبرد / كهرباء"},
                "min_seats": {"type": "integer", "description": "Minimum number of seats"},
                "max_fuel_consumption": {"type": "number", "description": "Max liters per 100km"},
                "required_features": {
                    "type": "array",
                    "items": {"type": "string", "enum": FEATURE_COLUMNS},
                    "description": "Features the car must have",
                },
                "sort_by": {"type": "string", "enum": SORT_OPTIONS, "description": "Default price_asc"},
                "limit": {"type": "integer", "description": "How many cars to return (default 5, max 10)"},
            },
        },
    },
    {
        "name": "search_faults",
        "description": (
            "Search the car faults database (the ONLY source of causes, solutions, severity and costs). "
            "Pass the user's symptom description in Arabic (or what you see in an image). "
            "Returns matches with a similarity score, or status 'no_confident_match'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Symptom description in Arabic"},
                "category": {"type": "string", "description": "Optional fault category to narrow the search"},
                "limit": {"type": "integer", "description": "Max results (default 3, max 5)"},
            },
            "required": ["query"],
        },
    },
]


class ToolRunner:
    """
    بيربط اسم الأداة بالدالة اللي بتنفّذها.
    فيه كاش صغير (LRU) للنتايج: نفس البحث بنفس المعاملات مبيتحسبش تاني (بيوفّر وقت،
    ومن غير ما نغيّر النتيجة لأن الداتا ثابتة لحد ما الإكسيل يتغير ويتعمل ToolRunner جديد).
    """

    def __init__(self, cars_df, fault_retriever: FaultRetriever, cache_size: int = 256):
        self.cars = cars_df
        self.faults = fault_retriever
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()  # الكاش مشترك بين الجلسات

    def run(self, name: str, args: dict) -> dict:
        args = dict(args or {})
        key = name + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                log.info("TOOL CACHE HIT %s %s", name, args)
                return copy.deepcopy(self._cache[key])

        log.info("TOOL CALL %s %s", name, args)
        if name == "search_cars":
            result = search_cars(self.cars, **{k: v for k, v in args.items() if k in _CAR_ARGS})
        elif name == "search_faults":
            result = self._search_faults(**{k: v for k, v in args.items() if k in _FAULT_ARGS})
        else:
            raise ValueError(f"أداة غير معروفة: {name}")  # الأخطاء مبتتخزنش في الكاش
        log.info("TOOL RESULT %s -> %s", name, _short(result))

        with self._lock:
            self._cache[key] = copy.deepcopy(result)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result

    def _search_faults(self, query: str, category: str | None = None, limit: int = 3) -> dict:
        limit = max(1, min(int(limit or 3), 5))
        hits = self.faults.search(query, category=category, limit=limit)
        confident = [h for h in hits if h["score"] >= CONFIDENCE_THRESHOLD]
        if not confident:
            return {"status": "no_confident_match",
                    "message": "No fault in the database matches this description with enough confidence.",
                    "best_score": hits[0]["score"] if hits else 0.0}
        return {"status": "ok", "results": confident}


_CAR_ARGS = {"min_price", "max_price", "brand", "model", "body_type", "condition", "min_year",
             "transmission", "fuel_type", "min_seats", "max_fuel_consumption", "required_features",
             "sort_by", "limit"}
_FAULT_ARGS = {"query", "category", "limit"}


def _short(result: dict) -> str:
    """ملخص قصير للّوج (مش بنطبع كل الداتا)."""
    if "cars" in result:
        return f"count={result['count']} relaxed={result['relaxed']}"
    return f"status={result.get('status')} n={len(result.get('results', []))}"
