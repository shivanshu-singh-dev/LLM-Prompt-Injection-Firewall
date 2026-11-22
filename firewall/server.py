"""
Server launcher for Prompt Injection Firewall.

Usage:
    python -m firewall.server --host 0.0.0.0 --port 8000
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start the Prompt Injection Firewall proxy.")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host address")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"--> Starting Prompt Injection Firewall on http://{args.host}:{args.port}")
    uvicorn.run("firewall.proxy:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
