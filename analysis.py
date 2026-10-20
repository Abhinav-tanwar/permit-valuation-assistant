"""Train the saved permit pipeline and return its measured results."""

import json
from pathlib import Path
from train_model import main


def run():
    main()
    result = json.loads(
        (Path(__file__).parent / "artifacts/model_metrics.json").read_text()
    )
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
