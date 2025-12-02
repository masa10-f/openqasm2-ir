"""Allow running the optimizer as a module: python -m optimizer"""

from optimizer.tket_optimizer import main

if __name__ == "__main__":
    raise SystemExit(main())
