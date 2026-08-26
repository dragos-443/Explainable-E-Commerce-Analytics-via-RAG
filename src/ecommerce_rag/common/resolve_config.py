"""Print one scalar value from the selected environment profile."""

from __future__ import annotations

import argparse
from typing import Any

from ecommerce_rag.common.config import load_config


def resolve(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Configuration key not found: {dotted_key}")
        value = value[key]
    if isinstance(value, (dict, list)):
        raise TypeError(f"Configuration key is not a scalar: {dotted_key}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment")
    parser.add_argument("--get", required=True, dest="dotted_key")
    args = parser.parse_args()
    print(resolve(load_config(args.environment), args.dotted_key))


if __name__ == "__main__":
    main()
