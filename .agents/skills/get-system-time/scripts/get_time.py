#!/usr/bin/env python3
"""
Script de apoyo para la skill get-system-time.
Retorna un payload JSON estructurado con la información temporal del sistema.
"""

import datetime
import json
import time


def main():
    now_utc = datetime.datetime.now(datetime.UTC)
    now_local = datetime.datetime.now().astimezone()

    payload = {
        "timestamp_unix": int(time.time()),
        "iso_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "iso_local": now_local.isoformat(),
        "timezone_name": now_local.tzname(),
        "formatted_local": now_local.strftime("%A, %d de %B de %Y - %H:%M:%S")
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
