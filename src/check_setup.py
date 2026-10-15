"""Quick environment check. Run:  python -m src.check_setup

Confirms the config loads, the core libraries import, and (if installed) that every
foundation model named in the config exists in timm. Nothing is downloaded.
"""
import importlib

from src.utils import load_config, set_seed


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    print(
        f"Config loaded. seed={cfg['seed']}, hold-out generator={cfg['splits']['holdout_generator']}")

    for lib in ["numpy", "pandas", "PIL", "yaml", "sklearn", "matplotlib", "imagehash", "torch", "timm"]:
        try:
            mod = importlib.import_module(lib)
            print(f"  [ok]      {lib:<11} {getattr(mod, '__version__', '')}")
        except ImportError:
            print(
                f"  [missing] {lib:<11} (needed later; pip install -r requirements.txt)")

    try:
        import timm

        print("\nFoundation models for Step 5:")
        for short, name in cfg["backbones"].items():
            found = name in timm.list_models(
                name.split(".")[0], pretrained=True)
            print(f"  [{'ok' if found else '??'}] {short:<12} {name}")
    except ImportError:
        pass

    try:
        import torch

        print(f"\nGPU available: {torch.cuda.is_available()}"
              + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else
                 " -> use Kaggle/Colab for Steps 2, 5 and 6"))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
