"""autotune — a local Try -> Check -> Reward -> Nudge loop that enforces a story in pokemon-kafka.

The loop wraps the pokemon-kafka agent as an environment:

    Try     -> run the agent N times (rollout.py)
    Check   -> verify each rollout against an ordered story (verifier.py + story.py)
    Reward  -> per-beat pass=1 / fail=0, on-story aggregate (verifier.py)
    Nudge   -> reinforce what passed, via either backend:
                 nudge_sft.py   -> rejection-sampling LoRA SFT of a local model (CUDA or MLX)
                 nudge_steer.py -> steer the existing param/Claude agent

The orchestrator that closes the loop lives in loop.py.
"""

__version__ = "0.1.0"
