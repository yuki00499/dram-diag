import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description="启动 DRAM 诊断服务")
    parser.add_argument("--deployment", default="artifacts/deployment.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn
    from dram_diag.api import create_app
    uvicorn.run(create_app(args.deployment), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
