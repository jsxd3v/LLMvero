import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="llmvero", description="Run the LLMvero server.")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="start the API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    args = parser.parse_args()

    if args.command == "serve":
        uvicorn.run(
            "llmvero.main:app_factory", factory=True, host=args.host, port=args.port, reload=args.reload
        )


if __name__ == "__main__":
    main()
