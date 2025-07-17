import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent  # This points to the outer swelancer directory
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added to sys.path: {project_root}")


def debug_module_loading(module_name: str) -> None:
    print(f"\n=== Debugging module: {module_name} ===")

    # Print current sys.path
    print("\nCurrent sys.path:")
    for i, path in enumerate(sys.path, 1):
        print(f"{i}. {path}")

    # Try to import the module
    try:
        module = __import__(module_name, fromlist=[""])
        print(
            f"\nSuccessfully imported {module_name} from: {getattr(module, '__file__', 'unknown')}"
        )
    except Exception as e:
        print(f"\nError importing {module_name}: {e}")


def main() -> None:
    module_name = "swelancer.eval"
    debug_module_loading(module_name)


if __name__ == "__main__":
    main()
