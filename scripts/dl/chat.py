#!/usr/bin/env python3
"""Interactive OpenAI-compatible chat client for SGLang / vLLM servers.

Usage:
  python chat.py                                        # interactive mode (connect to localhost:30000)
  python chat.py -q "hello"                             # quick single-shot
  python chat.py --url http://10.0.0.1:30000/v1         # custom endpoint
  python chat.py --model Qwen3-1.7B                     # explicit model name
"""

import argparse
import os
import sys
import time
from openai import OpenAI


def list_models(client: OpenAI) -> list[str]:
    return [m.id for m in client.models.list()]


def resolve_model(client: OpenAI, prefer: str) -> str:
    models = list_models(client)
    if not models:
        print("[chat] error: no models available on server", file=sys.stderr)
        sys.exit(1)
    if prefer:
        if prefer in models:
            return prefer
        close = [m for m in models if prefer.lower() in m.lower()]
        if close:
            return close[0]
        print(f"[chat] model '{prefer}' not found, available: {models}", file=sys.stderr)
        sys.exit(1)
    return models[0]


def quick_chat(client: OpenAI, model: str, message: str, system_prompt: str, stream: bool):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=model, messages=messages, stream=stream,
    )
    if stream:
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    print(delta.content, end="", flush=True)
        print()
    else:
        print(response.choices[0].message.content)


def interactive_chat(client: OpenAI, model: str, system_prompt: str, stream: bool):
    import readline  # enable arrow keys / line editing  # noqa: F401

    history: list[dict] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    print(f"[chat] connected to {client.base_url}, model={model}")
    print("[chat] type your messages, Ctrl+C or /exit to quit, /clear to reset history")
    print()

    try:
        while True:
            try:
                msg = input("> ").strip()
            except EOFError:
                print()
                break
            if not msg:
                continue
            if msg == "/exit":
                break
            if msg == "/clear":
                history = []
                if system_prompt:
                    history.append({"role": "system", "content": system_prompt})
                print("[chat] history cleared")
                continue

            history.append({"role": "user", "content": msg})

            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=model, messages=history, stream=stream,
            )

            collected = ""
            if stream:
                for chunk in response:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            collected += delta.content
                            print(delta.content, end="", flush=True)
                print()
            else:
                collected = response.choices[0].message.content
                print(collected)

            ttft = (time.perf_counter() - t0) * 1000
            print(f"[chat] {len(collected)} tokens, {ttft:.0f}ms TTFT")
            history.append({"role": "assistant", "content": collected})
    except KeyboardInterrupt:
        print()


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible chat client")
    parser.add_argument("--url", default=os.environ.get("CHAT_URL", "http://127.0.0.1:30000/v1"),
                        help="Server API base URL (default: $CHAT_URL or http://127.0.0.1:30000/v1)")
    parser.add_argument("--model", default=os.environ.get("CHAT_MODEL", ""),
                        help="Model name (default: auto-detect from /v1/models)")
    parser.add_argument("--system-prompt", default=os.environ.get("CHAT_SYSTEM_PROMPT", ""),
                        help="System prompt")
    parser.add_argument("-q", "--quick", default="", help="Single message (non-interactive)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    client = OpenAI(base_url=args.url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    model = resolve_model(client, args.model)
    stream = not args.no_stream

    if args.quick:
        quick_chat(client, model, args.quick, args.system_prompt, stream)
    else:
        interactive_chat(client, model, args.system_prompt, stream)


if __name__ == "__main__":
    main()
