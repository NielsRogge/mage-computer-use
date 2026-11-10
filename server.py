# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.10,<2", "playwright>=1.45", "uvicorn", "starlette"]
# ///
"""Computer-use MCP server for Mage-VL-Cowork style models.

Drives one Chrome window through Playwright and exposes Mage's `computer_use` tool:
coordinates are normalized to [0, 999], every action returns a screenshot.
The viewport is 1920x1080 because Mage-VL-Cowork was trained on OSWorld screens of that size;
smaller screenshots make its clicks drift.

Run:  uv run server.py                      # http://127.0.0.1:8765/mcp, visible Chrome
      uv run server.py --headless --browser chromium
"""

import argparse
import asyncio
import base64
import contextlib
import urllib.parse
from pathlib import Path
from typing import Annotated, Literal

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent
from playwright.async_api import BrowserContext, Page, async_playwright
from pydantic import Field
from starlette.middleware.cors import CORSMiddleware

WIDTH, HEIGHT = 1920, 1080
END_AGENTIC_LOOP = "llama.cpp/endAgenticLoop"  # _meta key the llama.cpp WebUI checks to stop its agentic loop
# _meta key with a one-line, human-readable description of what the action did ({"summary", "url",
# "title", "navigated"}). The llama.cpp WebUI shows it in the chat; it is never sent to the model.
ACTIVITY = "llama.cpp/activity"

KEY_NAMES = {
    "ctrl": "ControlOrMeta", "control": "ControlOrMeta",
    "cmd": "Meta", "command": "Meta", "meta": "Meta", "super": "Meta", "win": "Meta",
    "alt": "Alt", "option": "Alt", "shift": "Shift",
    "enter": "Enter", "return": "Enter", "esc": "Escape", "escape": "Escape",
    "backspace": "Backspace", "delete": "Delete", "del": "Delete", "tab": "Tab", "space": "Space",
    "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
    "pageup": "PageUp", "page_up": "PageUp", "pagedown": "PageDown", "page_down": "PageDown",
    "home": "Home", "end": "End", "insert": "Insert",
}


def key_name(key: str) -> str:
    k = key.strip()
    low = k.lower()
    if low in KEY_NAMES:
        return KEY_NAMES[low]
    if low.startswith("arrow"):
        return "Arrow" + low[5:].capitalize()
    if len(low) in (2, 3) and low[0] == "f" and low[1:].isdigit():
        return low.upper()
    return k


class Browser:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = asyncio.Lock()
        self.pw = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        # Emulated address bar: the screenshot only shows the page, but models trained on full
        # desktops press Ctrl+L, type a URL and press Enter. None = not focused.
        self.omnibox: str | None = None

    async def navigate(self, text: str) -> str:
        page = await self.get_page()
        url = omnibox_url(text)
        self.omnibox = None
        with contextlib.suppress(Exception):
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        return url

    async def get_page(self) -> Page:
        if self.context is None:
            self.pw = await async_playwright().start()
            channel = None if self.args.browser == "chromium" else self.args.browser
            Path(self.args.profile).mkdir(parents=True, exist_ok=True)
            self.context = await self.pw.chromium.launch_persistent_context(
                self.args.profile,
                channel=channel,
                headless=self.args.headless,
                # Playwright starts Chrome with --no-sandbox by default, which shows a warning bar
                chromium_sandbox=True,
                viewport={"width": WIDTH, "height": HEIGHT},
                device_scale_factor=1,
                args=[f"--window-size={WIDTH},{HEIGHT + 140}"],
            )
            self.context.on("page", self._on_page)
            self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
            await self.page.goto(self.args.start_url)
        if self.page is None or self.page.is_closed():
            open_pages = [p for p in self.context.pages if not p.is_closed()]
            self.page = open_pages[-1] if open_pages else await self.context.new_page()
        return self.page

    def _on_page(self, page: Page) -> None:
        # follow links that open in a new tab
        self.page = page

    async def observe(self, note: str, summary: str, before_url: str | None = None) -> CallToolResult:
        page = await self.get_page()
        with contextlib.suppress(Exception):
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        await asyncio.sleep(self.args.settle)
        shot = await page.screenshot(type="jpeg", quality=90)
        title = ""
        with contextlib.suppress(Exception):
            title = await page.title()
        text = f"{note}\nCurrent page: {title!r} ({page.url})"
        if self.omnibox is not None:
            text += f"\nAddress bar (focused): {self.omnibox!r}"
        print(text.replace("\n", " | "), flush=True)
        # Screenshot only, like Mage's OSWorld harness. Any text before the image (even just
        # "Current page: ...") made Mage click ~70 px too low one step into a task.
        content: list = [ImageContent(type="image", data=base64.b64encode(shot).decode(), mimeType="image/jpeg")]
        if self.args.text:
            content.insert(0, TextContent(type="text", text=text))
        # For people, not the model: clients show this instead of the raw arguments
        activity = {"summary": summary, "url": page.url, "title": title,
                    "navigated": before_url is not None and page.url != before_url}
        return CallToolResult(content=content, _meta={ACTIVITY: activity})


ADDRESS_BAR_COMBOS = {("ControlOrMeta", "l"), ("Meta", "l"), ("Alt", "d"), ("F6",)}

# What a click hit, for the activity summary: the nearest interactive ancestor's kind and label.
DESCRIBE_POINT_JS = """([x, y]) => {
  const hit = document.elementFromPoint(x, y);
  if (!hit) return null;
  const interactive = hit.closest('a,button,input,textarea,select,summary,label,[role=button],[role=link],[role=tab],[role=menuitem],[role=checkbox],[role=option],[role=combobox],[role=searchbox]');
  if (!interactive) {
    // plain content: name it only by its own short text (a heading, a product name), not a whole section
    const text = (hit.innerText || '').trim().replace(/\\s+/g, ' ');
    return {kind: '', label: text.length <= 60 ? text : ''};
  }
  const el = interactive;
  const tag = el.tagName.toLowerCase();
  const inputKinds = {checkbox: 'checkbox', radio: 'option', submit: 'button', button: 'button'};
  const kind = el.getAttribute('role') || {a: 'link', button: 'button', textarea: 'text field', select: 'dropdown',
    summary: 'toggle', iframe: 'embedded frame', img: 'image'}[tag] || (tag === 'input' ? inputKinds[el.type] || 'text field' : '');
  const label = el.getAttribute('aria-label') || (tag === 'input' || tag === 'textarea' ? el.placeholder || el.value : el.innerText)
    || el.getAttribute('title') || el.getAttribute('alt') || '';
  return {kind, label: label.trim().replace(/\\s+/g, ' ').slice(0, 60)};
}"""


async def describe_point(page: Page, x: float, y: float) -> str:
    """'link “Models”', 'text field “Search models...”', or 'the page' if nothing useful is there."""
    with contextlib.suppress(Exception):
        hit = await page.evaluate(DESCRIBE_POINT_JS, [x, y])
        if hit and hit.get("label"):
            return f'{hit["kind"]} “{hit["label"]}”'.strip()
        if hit and hit.get("kind"):
            return hit["kind"]
        if hit is not None:
            return "the page"
    return ""


async def has_text_focus(page: Page) -> bool:
    """True if keystrokes would go into something editable (or into a frame we can't inspect)."""
    with contextlib.suppress(Exception):
        return await page.evaluate("""() => {
            const e = document.activeElement;
            if (!e || e === document.body || e === document.documentElement) return false;
            return e.isContentEditable || e.tagName === 'IFRAME' || e.tagName === 'TEXTAREA' || e.tagName === 'SELECT'
                || (e.tagName === 'INPUT' && !['button', 'checkbox', 'radio', 'submit', 'reset', 'image', 'file', 'range', 'color'].includes(e.type));
        }""")
    return True  # if we can't tell, type as asked


def short_url(url: str) -> str:
    u = urllib.parse.urlsplit(url)
    s = u.netloc + (f"{u.path}?{u.query}" if u.query else u.path.rstrip("/")) or url
    return s if len(s) <= 60 else s[:57] + "..."


def pretty_combo(names: tuple[str, ...]) -> str:
    rename = {"ControlOrMeta": "Ctrl", "Meta": "Cmd", "ArrowUp": "Up", "ArrowDown": "Down", "ArrowLeft": "Left", "ArrowRight": "Right"}
    return "+".join(rename.get(n, n.upper() if len(n) == 1 else n) for n in names)


def quoted(text: str) -> str:
    t = text.replace("\n", "⏎")
    return f"“{t if len(t) <= 60 else t[:57] + '...'}”"


def omnibox_url(text: str) -> str:
    """What Chrome's omnibox does with typed text: open it as a URL, or search for it."""
    t = text.strip()
    if t.startswith(("http://", "https://", "file://", "about:", "chrome://")):
        return t
    if " " not in t and ("." in t or t.startswith("localhost")):
        return ("http://" if t.startswith(("localhost", "127.0.0.1")) else "https://") + t
    return "https://duckduckgo.com/?q=" + urllib.parse.quote_plus(t)



def to_px(coord: list[float] | None) -> tuple[float, float]:
    if not coord or len(coord) != 2:
        raise ValueError("coordinate must be [x, y] in [0, 999]")
    x = min(max(float(coord[0]), 0.0), 999.0)
    y = min(max(float(coord[1]), 0.0), 999.0)
    return x / 999.0 * (WIDTH - 1), y / 999.0 * (HEIGHT - 1)


def build_server(args: argparse.Namespace) -> tuple[FastMCP, Browser]:
    mcp = FastMCP("computer-use", host=args.host, port=args.port, stateless_http=True)
    browser = Browser(args)

    @mcp.tool(name="computer_use", description="Use a mouse and keyboard to interact with a computer screen. The screen's resolution is 1000x1000: coordinates are in [0, 999].")
    async def computer_use(
        action: Annotated[Literal["click", "double_click", "triple_click", "move", "drag", "type", "keypress", "scroll", "wait", "screenshot", "terminate"], Field(description=(
            "* `click`: Click at (x,y). Use `button` for left/right/middle.\n* `double_click`: Double click at (x,y).\n"
            "* `triple_click`: Triple click at (x,y) to select line/paragraph.\n* `move`: Move cursor to (x,y) without clicking.\n"
            "* `drag`: Drag from `start` to `end` coordinates.\n* `type`: Type text string.\n* `keypress`: Press key(s). Single key or combo.\n"
            "* `scroll`: Scroll at optional (x,y) position. scroll_x/scroll_y in [-999,999], |999|=full screen. Negative scroll_y=down, negative scroll_x=left.\n"
            "* `wait`: Wait specified seconds.\n* `screenshot`: Capture the current screen without changing anything.\n* `terminate`: End the task."))],
        coordinate: Annotated[list[float] | None, Field(description="(x,y) in [0,999].")] = None,
        button: Annotated[Literal["left", "right", "middle"], Field(description="Mouse button for click: left/right/middle.")] = "left",
        start: Annotated[list[float] | None, Field(description="(x,y) drag start in [0,999].")] = None,
        end: Annotated[list[float] | None, Field(description="(x,y) drag end in [0,999].")] = None,
        keys: Annotated[list[str] | str | None, Field(description="Key list for keypress.")] = None,
        text: Annotated[str | None, Field(description="Text for action=type.")] = None,
        scroll_x: Annotated[float, Field(description="Horizontal scroll [-999,999]. Negative=left.")] = 0,
        scroll_y: Annotated[float, Field(description="Vertical scroll [-999,999]. Negative=down.")] = 0,
        time: Annotated[float, Field(description="Seconds for wait.")] = 1,
        status: Annotated[Literal["success", "failure"], Field(description="success/failure for terminate.")] = "success",
        msg_to_user: Annotated[str | None, Field(description="Optional message for terminate.")] = None,
    ) -> CallToolResult:
        if action == "terminate":
            # Mage keeps calling tools after terminate, so ask the client (llama.cpp WebUI) to end
            # its agentic loop; other clients ignore the _meta keys.
            summary = ("Finished" if status == "success" else "Gave up") + (f": {msg_to_user}" if msg_to_user else "")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Task ended ({status}). {msg_to_user or ''}".strip())],
                _meta={END_AGENTIC_LOOP: True, ACTIVITY: {"summary": summary}},
            )
        async with browser.lock:
            page = await browser.get_page()
            mouse = page.mouse
            before_url = page.url
            if action in ("click", "double_click", "triple_click", "move", "drag", "scroll"):
                browser.omnibox = None  # clicking the page takes focus away from the address bar
            if action in ("click", "double_click", "triple_click"):
                x, y = to_px(coordinate)
                target = await describe_point(page, x, y)  # before the click, which may navigate away
                count = {"click": 1, "double_click": 2, "triple_click": 3}[action]
                await mouse.click(x, y, button=button, click_count=count)
                note = f"{action} at ({x:.0f}, {y:.0f}) px"
                verb = {"click": "Clicked", "double_click": "Double-clicked", "triple_click": "Triple-clicked"}[action]
                if button == "right":
                    verb = "Right-clicked"
                summary = f"{verb} {target}" if target else f"{verb} at ({x:.0f}, {y:.0f})"
            elif action == "move":
                x, y = to_px(coordinate)
                target = await describe_point(page, x, y)
                await mouse.move(x, y)
                note = f"moved to ({x:.0f}, {y:.0f}) px"
                summary = f"Hovered over {target}" if target else "Moved the mouse"
            elif action == "drag":
                x0, y0 = to_px(start or coordinate)
                x1, y1 = to_px(end)
                await mouse.move(x0, y0)
                await mouse.down()
                await mouse.move(x1, y1, steps=15)
                await mouse.up()
                note = f"dragged ({x0:.0f}, {y0:.0f}) -> ({x1:.0f}, {y1:.0f}) px"
                summary = "Dragged"
            elif action == "type" and browser.omnibox is not None:
                browser.omnibox += (text or "").replace("\n", "")
                note = f"typed {text!r} into the address bar"
                summary = f"Typed {quoted(browser.omnibox)} in the address bar"
                if (text or "").endswith("\n"):
                    opened = await browser.navigate(browser.omnibox)
                    note += f", opened {opened}"
                    summary = f"Opened {short_url(opened)}"
            elif action == "type" and not await has_text_focus(page):
                # Typing into the page itself only triggers shortcuts (space scrolls down a screen),
                # which hides what the model was looking for. Happens when a click missed, e.g. the
                # blind first step of a WebUI chat, which starts without a screenshot.
                note = f"did not type {text!r}: no text field has focus"
                summary = f"Skipped typing {quoted(text or '')}: no text field selected"
            elif action == "type":
                await page.keyboard.type(text or "", delay=15)
                note = f"typed {text!r}"
                summary = f"Typed {quoted(text or '')}"
            elif action == "keypress":
                parts = keys.split("+") if isinstance(keys, str) else list(keys or [])
                names = tuple(key_name(k) for k in parts)
                combo = "+".join(names)
                if not combo:
                    raise ValueError("keypress needs keys")
                summary = f"Pressed {pretty_combo(names)}"
                if tuple(n.lower() if len(n) == 1 else n for n in names) in ADDRESS_BAR_COMBOS:
                    browser.omnibox = ""
                    note = f"pressed {combo}, address bar focused (current URL selected)"
                    summary = "Focused the address bar"
                elif browser.omnibox is not None and names[-1] == "Enter":  # also Alt/Shift+Enter (new tab/window)
                    opened = await browser.navigate(browser.omnibox)
                    note = f"pressed Enter in the address bar, opened {opened}"
                    summary = f"Opened {short_url(opened)}"
                elif browser.omnibox is not None and names == ("Escape",):
                    browser.omnibox = None
                    note = "pressed Escape, address bar unfocused"
                elif browser.omnibox is not None and names == ("Backspace",):
                    browser.omnibox = browser.omnibox[:-1]
                    note = "pressed Backspace in the address bar"
                elif browser.omnibox is not None and names in (("ControlOrMeta", "a"), ("Meta", "a")):
                    browser.omnibox = ""
                    note = f"pressed {combo} in the address bar"
                else:
                    browser.omnibox = None
                    await page.keyboard.press(combo)
                    note = f"pressed {combo}"
            elif action == "scroll":
                if coordinate:
                    await mouse.move(*to_px(coordinate))
                dx = float(scroll_x) / 999.0 * WIDTH
                dy = -float(scroll_y) / 999.0 * HEIGHT
                await mouse.wheel(dx, dy)
                note = f"scrolled ({scroll_x}, {scroll_y})"
                summary = "Scrolled " + ("down" if dy > 0 else "up" if dy < 0 else "right" if dx > 0 else "left")
            elif action == "wait":
                await asyncio.sleep(min(max(float(time), 0.0), 30.0))
                note = f"waited {time}s"
                summary = f"Waited {float(time):g} s"
            else:
                note = "screenshot"
                summary = "Took a screenshot"
            return await browser.observe(note, summary, before_url)

    @mcp.tool(name="open_url", description="Open a URL in the browser. The address bar is not part of the screenshot, so use this to go to a website.")
    async def open_url(url: Annotated[str, Field(description="The URL to open, e.g. https://huggingface.co")]) -> CallToolResult:
        url = omnibox_url(url)  # adds http:// for localhost, https:// otherwise; searches non-URLs
        async with browser.lock:
            page = await browser.get_page()
            before_url = page.url
            with contextlib.suppress(Exception):
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return await browser.observe(f"opened {url}", f"Opened {short_url(url)}", before_url)

    return mcp, browser


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--browser", default="chrome", help="'chrome' (installed Google Chrome) or 'chromium' (Playwright's)")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--profile", default=str(Path.home() / ".cache" / "mage-computer-use" / "profile"), help="separate browser profile, never your own")
    p.add_argument("--start-url", default="https://duckduckgo.com")
    p.add_argument("--settle", type=float, default=0.7, help="seconds to wait after an action before the screenshot")
    p.add_argument("--lazy-browser", action="store_true", help="start Chrome on the first action instead of at startup")
    p.add_argument("--allow-origin", action="append", default=None, help="extra CORS origin allowed to connect (the WebUI)")
    p.add_argument("--text", action="store_true", help="also return a text note (action, page title, URL) with each screenshot; hurts Mage's grounding")
    args = p.parse_args()

    mcp, browser = build_server(args)
    app = mcp.streamable_http_app()
    origins = args.allow_origin or [f"http://{h}:{port}" for h in ("127.0.0.1", "localhost") for port in (9931, 9932)]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["mcp-session-id", "mcp-protocol-version"],
    )
    if not args.lazy_browser:
        # open Chrome right away (in the server's event loop, where the tools use it) instead of on
        # the first action, so the window is there before the first chat
        server_lifespan = app.router.lifespan_context

        async def open_browser() -> None:
            try:
                async with browser.lock:
                    await browser.get_page()
                print(f"Chrome is open on {args.start_url}", flush=True)
            except Exception as e:  # the first action retries the launch
                print(f"Could not open Chrome at startup: {e}", flush=True)

        @contextlib.asynccontextmanager
        async def lifespan(a):
            async with server_lifespan(a):
                task = asyncio.create_task(open_browser())
                yield
                task.cancel()

        app.router.lifespan_context = lifespan
    # the browser is a child process of Playwright's driver and exits with this process
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
