from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class Route:
    backend: str
    target_op_type: str


class RoutingTable:
    def __init__(self, routes: dict[str, Route], alltoall_op_type: str = "AllToAll") -> None:
        self._routes = dict(routes)
        self.alltoall_op_type = str(alltoall_op_type)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RoutingTable":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        route_payload = payload.get("routes", {})
        routes: dict[str, Route] = {}
        for backend in ("npu", "cpu", "ignore"):
            for op_type, target in dict(route_payload.get(backend, {})).items():
                routes[str(op_type)] = Route(backend=backend.upper(), target_op_type=str(target))
        synthetic = dict(payload.get("synthetic", {}))
        return cls(routes, alltoall_op_type=str(synthetic.get("alltoall_op_type", "AllToAll")))

    def route_for(self, op_type: str) -> Route:
        if op_type not in self._routes:
            raise KeyError(f"op_type {op_type!r} is not declared in the routing table")
        return self._routes[op_type]

    def validate_all(self, op_types: Iterable[str]) -> None:
        missing = sorted({str(op_type) for op_type in op_types if str(op_type) not in self._routes})
        if missing:
            raise KeyError(f"routing table is missing op_type(s): {', '.join(missing)}")

    def as_rows(self) -> list[dict[str, str]]:
        return [
            {"op_type": op_type, "backend": route.backend, "target_op_type": route.target_op_type}
            for op_type, route in sorted(self._routes.items())
        ]

