"""Add an Ollama tag to ~/.pi/agent/models.json the way every local entry is listed.

    uv run python scripts/register_pi_tag.py gemma4-e4b-tetris:20260905-ab12cd

Backs the file up to the next free models.json.bakN first. Idempotent.
"""

import json
import os
import shutil
import sys


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    tag = argv[1]
    path = os.path.expanduser("~/.pi/agent/models.json")
    n = 1
    while os.path.exists(f"{path}.bak{n}"):
        n += 1
    shutil.copy2(path, f"{path}.bak{n}")
    with open(path) as f:
        data = json.load(f)
    models = data["providers"]["ollama"]["models"]
    if any(m.get("id") == tag for m in models):
        print(f"already registered: {tag}")
        return 0
    models.append({"id": tag, "contextWindow": 131072, "input": ["text"], "reasoning": True,
                   "thinkingLevelMap": {"off": "none"}})
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"registered {tag} (backup: {path}.bak{n})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
