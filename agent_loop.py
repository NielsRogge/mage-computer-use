# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.10,<2"]
# ///
"""Minimal agent loop for llama-server + the computer-use MCP server.

--format harness (default) reproduces the message format of Mage's own OSWorld harness
(mage_gui_agent.py): the tool schema lives in the system prompt (mage_prompt.py), the model's turns
are raw text with <tool_call> blocks, and each step's screenshot comes back in a user message as
"<tool_response>\n" + image + "\n</tool_response>". The first user message is image + instruction.

--format openai is what the llama.cpp WebUI does: MCP tools passed as OpenAI `tools`, results as
`tool` messages. With Mage this hurts grounding: one step into a task, its click on huggingface.co's
search bar lands ~80 px too low (y=111 px instead of 12-52); the harness format hits it (y=40 px).

Both formats follow the harness for screenshots:
- one screenshot per step (after the step's last action);
- only the newest --keep-images screenshots stay in the prompt, older ones are replaced by
  "[This screenshot has been collapsed.]" (the harness keeps 1; its hybrid agent keeps 3).

Run:  uv run agent_loop.py "Search for llama.cpp on GitHub and open the repository"
"""

import argparse
import asyncio
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mage_prompt import MAGE_GUI_SYSTEM_PROMPT  # noqa: E402

SYSTEM = """You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.

For every step, reply in EXACTLY this format with NO suffix:

<think>
your reasoning paragraph describing the current screen state, how it connects to the task goal, and the rationale for the chosen action
</think>

Action: one short imperative sentence describing the next move.

<tool_call>
{"name": "example_function_name", "arguments": {"example_parameter_1": "value_1"}}
</tool_call>

Rules:
- The screen's resolution is 1000x1000. Coordinates are in [0, 999] range.
- The browser address bar is not in the screenshot, but ctrl+l still focuses it (then type the URL and press enter). `open_url` works too.
- Be brief: one sentence for Action.
- Do not output anything after the last tool call.
- If finishing, use computer_use with action=terminate."""


def chat(url: str, model: str, messages: list, tools: list | None, max_tokens: int) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


async def run(args: argparse.Namespace) -> int:
    async with streamablehttp_client(args.mcp) as (read, write, _), ClientSession(read, write) as session:
        await session.initialize()
        if args.format == "harness":
            return await run_harness(args, session)
        mcp_tools = (await session.list_tools()).tools
        tools = [{"type": "function", "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema}} for t in mcp_tools]
        system = Path(args.system_file).read_text() if args.system_file else SYSTEM
        if args.blind_start:
            # like a WebUI chat: just the user's words, no screenshot before the first action
            messages = [{"role": "system", "content": system}, {"role": "user", "content": args.task}]
        else:
            first = await session.call_tool("computer_use", {"action": "screenshot"})
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": [{"type": "text", "text": f"Instruction: {args.task}"}] + to_parts(first.content)[1:]}]
            save(args, 0, first.content)
        for step in range(1, args.max_steps + 1):
            collapse_old_images(messages, args.keep_images)
            t0 = time.time()
            resp = chat(args.llm, args.model, messages, tools, args.max_tokens)
            msg = resp["choices"][0]["message"]
            usage = resp.get("usage", {})
            calls = msg.get("tool_calls") or []
            print(f"\n--- step {step} ({time.time() - t0:.1f}s, prompt {usage.get('prompt_tokens')} tok) {(msg.get('content') or '').strip()}", flush=True)
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}
                            | ({"reasoning_content": msg["reasoning_content"]} if msg.get("reasoning_content") else {}))
            if not calls:
                print("no tool call, stopping")
                return 1
            for i, call in enumerate(calls):
                name = call["function"]["name"]
                arguments = json.loads(call["function"]["arguments"] or "{}")
                print(f"    {name} {arguments}", flush=True)
                result = await session.call_tool(name, arguments)
                parts = to_parts(result.content)
                lines = parts[0]["text"].splitlines() if parts else []
                print(f"    -> {lines[-1] if lines else '(screenshot)'}", flush=True)
                if i < len(calls) - 1:
                    # one screenshot per step, like Mage's harness: intermediate results are text only
                    parts = [p for p in parts if p["type"] == "text"]
                else:
                    save(args, step, result.content)
                text_only = all(p["type"] == "text" for p in parts)
                content = "\n".join(p["text"] for p in parts) if text_only else parts
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
                if name == "computer_use" and arguments.get("action") == "terminate":
                    print(f"terminated: {arguments}")
                    return 0
        print("step limit reached")
        return 2


COLLAPSED = "[This screenshot has been collapsed.]"
HISTORY_N = 10  # harness: steps kept as full turns; older ones become "Step k: <action sentence>"


async def run_harness(args: argparse.Namespace, session: ClientSession) -> int:
    first = await session.call_tool("computer_use", {"action": "screenshot"})
    shot = image_b64(first.content)
    save(args, 0, first.content)
    responses: list[str] = []  # raw model output per step
    summaries: list[str] = []  # its "Action:" sentence
    shots: list[str] = []      # shots[k] = screenshot the model saw before step k+1
    for step in range(1, args.max_steps + 1):
        messages = harness_messages(args.task, shot, responses, summaries, shots, args.keep_images)
        t0 = time.time()
        resp = chat(args.llm, args.model, messages, None, args.max_tokens)
        msg = resp["choices"][0]["message"]
        raw = raw_response(msg)
        calls = [c for c in parse_tool_calls(raw) if c.get("name") == "computer_use"]
        action = re.search(r"Action:\s*(.*)", raw)
        summary = action.group(1).strip() if action else ""
        print(f"\n--- step {step} ({time.time() - t0:.1f}s, prompt {resp.get('usage', {}).get('prompt_tokens')} tok) Action: {summary}", flush=True)
        responses.append(raw)
        summaries.append(summary)
        shots.append(shot)
        if not calls:
            print("no tool call, stopping")
            return 1
        for call in calls:
            arguments = call.get("arguments") or {}
            print(f"    computer_use {arguments}", flush=True)
            result = await session.call_tool("computer_use", arguments)
            texts = [i.text for i in result.content if i.type == "text"]
            print(f"    -> {texts[0].splitlines()[-1] if texts else ''}", flush=True)
            if arguments.get("action") == "terminate":
                print(f"terminated: {arguments}")
                return 0
            shot = image_b64(result.content) or shot
        save(args, step, result.content)
    print("step limit reached")
    return 2


def harness_messages(task: str, current: str, responses: list, summaries: list, shots: list, keep: int) -> list:
    """Port of MAGEGUIAgent._build_messages from Mage's osworld_eval/mm_agents/mage_gui_agent.py."""
    current_step = len(responses) + 1
    start_step = max(1, current_step - HISTORY_N)
    previous = "\n".join(f"Step {s}: {summaries[s - 1]}" for s in range(1, start_step)) or "None"
    intro = ("\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
             f"Instruction: {task}\n\nPrevious actions:\n{previous}")

    def shot_for(step: int) -> str | None:
        if step <= current_step - max(1, keep):
            return None  # collapsed
        return current if step == current_step else shots[step - 1]

    def image(b64: str) -> dict:
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    first = shot_for(start_step)
    messages = [{"role": "system", "content": MAGE_GUI_SYSTEM_PROMPT},
                {"role": "user", "content": [image(first), {"type": "text", "text": intro}] if first else COLLAPSED + intro}]
    for step in range(start_step, current_step):
        messages.append({"role": "assistant", "content": responses[step - 1]})
        after = shot_for(step + 1)
        messages.append({"role": "user", "content": [{"type": "text", "text": "<tool_response>\n"}, image(after), {"type": "text", "text": "\n</tool_response>"}]
                         if after else f"<tool_response>\n{COLLAPSED}\n</tool_response>"})
    return messages


def raw_response(msg: dict) -> str:
    """The model's output as raw text, whether or not llama-server split out reasoning / tool calls."""
    raw = (msg.get("content") or "").strip()
    if msg.get("reasoning_content"):
        raw = f"<think>\n{msg['reasoning_content'].strip()}\n</think>\n\n{raw}"
    for c in msg.get("tool_calls") or []:
        raw += f'\n\n<tool_call>\n{{"name": "{c["function"]["name"]}", "arguments": {c["function"]["arguments"]}}}\n</tool_call>'
    return raw


def parse_tool_calls(raw: str) -> list[dict]:
    calls = []
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.S):
        try:
            calls.append(json.loads(block))
        except json.JSONDecodeError:
            print(f"    unparsable tool call: {block[:200]!r}", flush=True)
    return calls


def image_b64(content: list) -> str | None:
    return next((i.data for i in content if i.type == "image"), None)


def collapse_old_images(messages: list, keep: int) -> None:
    """Replace all but the newest `keep` screenshots with a placeholder text (in place)."""
    seen = 0
    for msg in reversed(messages):
        if not isinstance(msg.get("content"), list):
            continue
        for j in range(len(msg["content"]) - 1, -1, -1):
            if msg["content"][j]["type"] == "image_url":
                seen += 1
                if seen > keep:
                    msg["content"][j] = {"type": "text", "text": COLLAPSED}


def save(args: argparse.Namespace, step: int, content: list) -> None:
    if not args.save_dir:
        return
    for item in content:
        if item.type == "image":
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.save_dir) / f"step{step:02d}.jpg").write_bytes(base64.b64decode(item.data))


def to_parts(content: list) -> list:
    parts = []
    for item in content:
        if item.type == "text":
            parts.append({"type": "text", "text": item.text})
        elif item.type == "image":
            parts.append({"type": "image_url", "image_url": {"url": f"data:{item.mimeType};base64,{item.data}"}})
    if parts and parts[0]["type"] != "text":
        parts.insert(0, {"type": "text", "text": ""})
    return parts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task")
    p.add_argument("--llm", default="http://127.0.0.1:9932")
    p.add_argument("--mcp", default="http://127.0.0.1:8765/mcp")
    p.add_argument("--model", default="Mage-VL-Cowork")
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--keep-images", type=int, default=1, help="screenshots kept in the prompt (Mage's harness: 1)")
    p.add_argument("--save-dir", help="save the screenshot the model sees at each step here")
    p.add_argument("--format", choices=["harness", "openai"], default="harness",
                   help="harness: Mage's OSWorld message format (better grounding); openai: tools + tool messages, like the WebUI")
    p.add_argument("--blind-start", action="store_true", help="openai format: no screenshot with the task, like a WebUI chat")
    p.add_argument("--system-file", help="openai format: read the system message from this file instead of SYSTEM")
    args = p.parse_args()
    check_servers(args)
    raise SystemExit(asyncio.run(run(args)))


def check_servers(args: argparse.Namespace) -> None:
    """Fail with a readable message if the model server or the MCP server isn't running."""
    missing = []
    for name, url, how in (
        ("model server", args.llm + "/health", "cd ~/Documents/python_projecten/llama.cpp && ./build-metal/bin/llama serve ... --port 9932 (see README)"),
        ("MCP server", args.mcp, "cd ~/Documents/python_projecten/mage-computer-use && uv run server.py"),
    ):
        try:
            urllib.request.urlopen(url, timeout=3)
        except urllib.error.HTTPError:
            pass  # it answered (the MCP endpoint rejects a plain GET), so it is up
        except OSError:
            missing.append(f"  - {name} at {url}\n    start it in another terminal: {how}")
    if missing:
        raise SystemExit("Can't reach:\n" + "\n".join(missing))


if __name__ == "__main__":
    main()
