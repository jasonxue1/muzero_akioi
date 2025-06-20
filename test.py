from pathlib import Path
import printer
from muzero.load_toml_and_pt import load_toml
from muzero.eval import eval


def test(times: int = 0) -> None:
    toml_path: Path = Path("train_config.toml")
    cfg = load_toml(toml_path)

    model_path: Path = Path("models") / cfg.model_name / "checkpoints" / "latest.pt"
    times = times or cfg.manual_test_times  # Python is amazing!!!
    if model_path.exists():
        print(f"load model from {model_path}\n")
        result = eval(times, model_path, True)
        result_table = [list(range(times)), result]
        printer.print_table(result_table)
    else:
        print(f"model not exist in {model_path}")


if __name__ == "__main__":
    test()
