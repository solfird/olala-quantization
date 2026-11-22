"""Build the default prompt file of the activation probe. Never runs a model.

64 chat prompts come from the inputs of the five test tasks (MGSM, HumanEval+, MBPP+, IFEval, AIME 2025), obtained from
``evals/run_tasks.py``'s own task factories through ``quant/gptq/extract_eval_items.py`` (which only builds the
``inspect_ai`` ``Task`` and reads its dataset; no model, no sandbox, no scorer). The items of each task are drawn with a
seeded generator, in equal quotas (64 = 13 + 13 + 13 + 13 + 12, the first tasks of the list getting the remainder). Each
becomes one user message that the probe renders with the model's chat template. 16 plain-text prompts (English prose,
French prose, Python code; the languages Appendix C tested) are written out below, 9 short and 7 longer ones made by
concatenating passages, so that the per-position growth of the mixer magnitudes can be seen beyond a few hundred tokens.

Run it in the harness environment (it has ``inspect_ai`` and ``inspect_evals``)::

    uv run --project ../evals python analysis/build_prompts.py --out analysis/prompts_default.jsonl

The dataset files are read from the Hugging Face cache (``HF_HUB_OFFLINE=1`` unless ``--online``).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
GPTQ_DIR = HERE.parent / "quant" / "gptq"
DEFAULT_EVALS_DIR = HERE.parent / "evals"
DEFAULT_TASKS = ("mgsm", "humaneval_plus", "mbpp_plus", "ifeval", "aime2025")

EN = [
    "The tides are produced mainly by the gravitational pull of the Moon, with a smaller contribution from the Sun. "
    "Because the water on the side of the Earth facing the Moon is pulled harder than the solid planet, and the water on "
    "the far side is pulled less, two bulges form and travel around the globe as the Earth turns. Most coasts therefore "
    "see two high tides and two low tides each day. When the Sun and the Moon line up, at new moon and full moon, their "
    "effects add and the tidal range is at its largest; engineers who design harbours and tidal barriers use tables "
    "that account for these cycles years in advance.",
    "Bread is among the oldest prepared foods. Archaeologists have found traces of flatbread made from wild cereals "
    "and ground roots at a site in present-day Jordan that is more than fourteen thousand years old, long before "
    "agriculture. Leavened loaves appeared later, probably by accident, when a mixture of flour and water caught wild "
    "yeast from the air. Over the centuries bakers learned to control fermentation with sourdough starters, and mills "
    "driven by water and wind made flour cheap enough for ordinary households. Today an enormous variety of loaves is "
    "baked, but the underlying chemistry of gluten, yeast and heat has not changed.",
    "Marta had worked at the post office for nineteen years when the letter without a stamp arrived. It was addressed, "
    "in careful blue ink, to a house on the street that had been demolished before she was born. She turned it over "
    "twice, weighed it in her hand, and put it on the shelf behind the counter where lost things waited. Every Friday "
    "afternoon she looked at it again. In the spring an old man came in, asked whether anything had arrived for the "
    "number he gave, and when she handed him the envelope he laughed quietly, as if a long argument had finally ended.",
    "Compound interest is the reason that small, regular savings can grow into a large sum. If you deposit one "
    "thousand euros at five percent a year, you earn fifty euros in the first year. In the second year you earn "
    "interest on one thousand and fifty euros, so the gain is slightly larger, and the effect repeats and accumulates. "
    "After thirty years the original amount has more than quadrupled without any further deposit. The same arithmetic "
    "works against borrowers: a debt that carries a high rate and is only partly repaid each month can grow faster "
    "than the payments reduce it, which is why the rate matters more than the size of the first instalment.",
    "From the gallery of the lighthouse the keeper could see the whole bay: the fishing boats returning one by one, the "
    "grey line of the breakwater, and the cliffs that turned orange in the last hour before dusk. He wound the clockwork "
    "that turned the lamp, polished the brass, and wrote the weather in a ledger that nobody ever read. When the fog "
    "came in he rang the bell by hand every two minutes until morning. Visitors asked whether he was lonely. He said "
    "that the sea was never quiet enough for that, and that a lighthouse is mostly a way of talking to strangers.",
]
FR = [
    "La Loire est le plus long fleuve de France. Elle prend sa source dans le Massif central, au pied du mont Gerbier "
    "de Jonc, et parcourt plus de mille kilomètres avant de se jeter dans l'océan Atlantique, près de Saint-Nazaire. "
    "Son cours est irrégulier : très haute en hiver, elle laisse apparaître de larges bancs de sable en été. Au fil des "
    "siècles, les rois et leurs courtisans ont bâti sur ses rives des châteaux célèbres, comme Chambord, Amboise ou "
    "Chenonceau, et la vallée est aujourd'hui inscrite au patrimoine mondial de l'humanité.",
    "Le fromage est né du besoin de conserver le lait. En laissant cailler le lait, puis en égouttant et en salant le "
    "caillé, les bergers obtenaient un aliment qui se gardait des mois dans les caves fraîches de la montagne. Chaque "
    "région a développé ses propres méthodes : pâte pressée cuite pour le comté, pâte molle à croûte fleurie pour le "
    "camembert, pâte persillée pour le roquefort. On dit parfois que la France compte autant de fromages que de jours "
    "dans l'année, mais les affineurs savent que la vraie différence tient souvent à la qualité de l'herbe et de la cave.",
    "Chaque samedi matin, le marché de la petite ville s'installait sous les platanes de la place. Les maraîchers "
    "disposaient leurs cageots de tomates, de haricots verts et de pêches encore tièdes, tandis que le poissonnier "
    "criait le prix des sardines. Léa, qui venait là depuis l'enfance, connaissait tous les visages et aucun des noms. "
    "Elle achetait toujours les mêmes abricots, saluait la vieille dame aux fleurs, puis prenait un café au comptoir "
    "du bar en regardant la foule passer lentement, comme si le temps, ce jour-là, acceptait de se laisser attendre.",
    "Le train à grande vitesse a transformé la géographie du pays. Lorsque la première ligne a relié Paris à Lyon, en "
    "1981, le trajet est passé de près de quatre heures à deux heures, et des villes autrefois éloignées sont devenues "
    "des voisines. Les ingénieurs ont dû résoudre des problèmes de freinage, d'alimentation électrique et de signalisation "
    "pour que les rames roulent en sécurité à plus de trois cents kilomètres par heure. Aujourd'hui, le réseau s'étend "
    "jusqu'à Bordeaux, Strasbourg et Marseille, et la comparaison avec l'avion fait toujours débat.",
]
CODE = [
    "from functools import lru_cache\n\n\n@lru_cache(maxsize=None)\ndef fibonacci(n: int) -> int:\n"
    '    """Return the n-th Fibonacci number (fibonacci(0) == 0)."""\n    if n < 0:\n        raise ValueError("n must be non-negative")\n'
    "    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n\n"
    'if __name__ == "__main__":\n    print([fibonacci(i) for i in range(15)])\n',
    "def binary_search(items: list[int], target: int) -> int:\n"
    '    """Index of target in the sorted list items, or -1 when it is absent."""\n'
    "    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n"
    "        if items[mid] == target:\n            return mid\n        if items[mid] < target:\n            lo = mid + 1\n"
    "        else:\n            hi = mid - 1\n    return -1\n\n\n"
    "assert binary_search([1, 3, 5, 7, 9], 7) == 3\nassert binary_search([1, 3, 5, 7, 9], 4) == -1\n",
    "class Stack:\n    def __init__(self) -> None:\n        self._items: list[int] = []\n\n"
    "    def push(self, value: int) -> None:\n        self._items.append(value)\n\n"
    '    def pop(self) -> int:\n        if not self._items:\n            raise IndexError("pop from empty stack")\n'
    "        return self._items.pop()\n\n    def peek(self) -> int:\n        return self._items[-1]\n\n"
    "    def __len__(self) -> int:\n        return len(self._items)\n",
    "import json\nfrom pathlib import Path\n\n\ndef load_config(path: str) -> dict:\n"
    '    """Read a JSON configuration file and fill in the defaults."""\n'
    '    defaults = {"retries": 3, "timeout": 30.0, "verbose": False}\n'
    '    data = json.loads(Path(path).read_text(encoding="utf-8"))\n'
    "    unknown = set(data) - set(defaults)\n    if unknown:\n"
    '        raise KeyError(f"unknown keys: {sorted(unknown)}")\n    return {**defaults, **data}\n',
]
# 16 plain prompts: (id, passages)
_ALL = [*EN, *FR, *CODE]
PLAIN_RECIPES: list[tuple[str, list[str]]] = [
    ("en-tides", [EN[0]]),
    ("en-bread", [EN[1]]),
    ("en-marta", [EN[2]]),
    ("en-interest", [EN[3]]),
    ("fr-loire", [FR[0]]),
    ("fr-fromage", [FR[1]]),
    ("fr-marche", [FR[2]]),
    ("code-fibonacci", [CODE[0]]),
    ("code-bsearch", [CODE[1]]),
    ("code-stack", [CODE[2]]),
    ("en-long-1", [EN[0], EN[1], EN[2]]),
    ("en-long-2", [EN[2], EN[3], EN[4], EN[0]]),
    ("fr-long-1", FR),
    ("code-long-1", CODE),
    ("mixed-1", [EN[3], FR[3], CODE[3]]),
    ("mixed-long", _ALL),
]


def plain_prompts(n: int = 16) -> list[dict[str, Any]]:
    """The first ``n`` (at most 16) built-in plain-text prompts, as prompt-file rows."""
    if not 0 <= n <= len(PLAIN_RECIPES):
        raise ValueError(f"at most {len(PLAIN_RECIPES)} plain prompts are available")
    return [{"id": f"plain:{pid}", "kind": "plain", "text": "\n\n".join(parts)} for pid, parts in PLAIN_RECIPES[:n]]


def _quotas(n: int, tasks: Sequence[str]) -> dict[str, int]:
    base, extra = divmod(n, len(tasks))
    return {t: base + (1 if i < extra else 0) for i, t in enumerate(tasks)}


def task_prompts(extracted: Mapping[str, Any], tasks: Sequence[str], n: int, seed: int) -> list[dict[str, Any]]:
    """Draw ``n`` chat prompts from the ``tasks`` of an ``extract_eval_items.extract`` result, in equal seeded quotas.

    A task with fewer usable items than its quota gives all of them and the shortfall goes to the tasks that still
    have items (in list order). Items without an ``input`` field are skipped.
    """
    pools = {t: [it for it in extracted["tasks"][t]["items"] if it["fields"].get("input")] for t in tasks}
    quota = _quotas(n, tasks)
    short = 0
    for t in tasks:
        if len(pools[t]) < quota[t]:
            short += quota[t] - len(pools[t])
            quota[t] = len(pools[t])
    for t in tasks:
        add = min(short, len(pools[t]) - quota[t])
        quota[t] += add
        short -= add
    if short:
        raise ValueError(f"the tasks have {sum(len(p) for p in pools.values())} usable items, fewer than the {n} asked")
    out = []
    for t in tasks:
        rng = random.Random(f"{seed}|{t}")
        for pos in sorted(rng.sample(range(len(pools[t])), quota[t])):
            item = pools[t][pos]
            out.append(
                {
                    "id": f"{t}:{item['id']}",
                    "kind": "chat",
                    "messages": [{"role": "user", "content": item["fields"]["input"]}],
                }
            )
    return out


def build_default_prompts(
    extracted: Mapping[str, Any],
    tasks: Sequence[str] = DEFAULT_TASKS,
    n_task: int = 64,
    n_plain: int = 16,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """``n_task`` chat prompts from the task items, then ``n_plain`` plain-text prompts."""
    return [*task_prompts(extracted, tasks, n_task, seed), *plain_prompts(n_plain)]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. Needs ``inspect_ai`` and ``inspect_evals`` importable (the harness environment)."""
    p = argparse.ArgumentParser(prog="build_prompts.py", description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--evals-dir", type=Path, default=DEFAULT_EVALS_DIR)
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--n-task-prompts", type=int, default=64)
    p.add_argument("--n-plain", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--online", action="store_true", help="allow dataset downloads (default: cache only)")
    args = p.parse_args(argv)
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    sys.path.insert(0, str(args.evals_dir))
    sys.path.insert(0, str(GPTQ_DIR))
    try:
        import extract_eval_items
        import run_tasks

        extracted = extract_eval_items.extract(run_tasks.TASKS, args.tasks)
        rows = build_default_prompts(extracted, args.tasks, args.n_task_prompts, args.n_plain, args.seed)
    except (ImportError, KeyError, ValueError, OSError) as e:
        print(f"error: {e!r}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(HERE))
    import probe_prompts

    probe_prompts.write_prompts(args.out, rows)
    print(f"wrote {args.out}: {args.n_task_prompts} task prompts + {args.n_plain} plain prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
