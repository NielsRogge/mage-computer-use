# mage-computer-use

A small MCP server that lets a local computer-use model (Mage-VL-Cowork on llama.cpp) drive a Chrome window, from the llama.cpp WebUI or a test agent loop.

- `server.py`: MCP server (streamable HTTP on `http://127.0.0.1:8765/mcp`). Tools:
  - `computer_use`: Mage's action space (`click`, `double_click`, `triple_click`, `move`, `drag`, `type`, `keypress`, `scroll`, `wait`, `screenshot`, `terminate`) with coordinates in [0, 999]. Every action returns only a screenshot (`--text` adds a note with the page title and URL, but that hurts Mage's clicks, see below). `terminate` sets `_meta["llama.cpp/endAgenticLoop"]` so the WebUI stops its agentic loop.
  - `open_url`: go to a website.
  - Every result also carries `_meta["llama.cpp/activity"]`: a one-line description for people (`Clicked button “Add to cart”`, `Typed “red running shoes”`, `Opened huggingface.co/models`) plus the page URL and title. Clicks are named after the element under the cursor. The branch's WebUI shows these lines instead of raw `computer_use` JSON; expanding one shows the screenshot taken after that action. The model never sees them.
- The screenshot shows only the page, not Chrome's toolbar. Mage still navigates the way it learned on full desktops (Ctrl+L, type a URL, Enter), so the server emulates the address bar: Ctrl+L / Alt+D / F6 focuses it, the next `type` fills it, and Enter opens the URL (or a DuckDuckGo search for non-URLs).
- `agent_loop.py`: drives llama-server + the MCP server in the message format of Mage's own OSWorld harness (`mage_gui_agent.py`; system prompt in `mage_prompt.py`). `--format openai` uses OpenAI `tools` + `tool` messages, like the WebUI. `--save-dir DIR` saves what the model saw at each step.
- `webui-config.json`: WebUI presets for Mage: the system message (added to every new chat, hidden from view; Settings > General > "Show system message" shows it again), and keep 1 tool screenshot.
- `test_site/`: a tiny offline shop for a repeatable test task.

The browser viewport is fixed at 1920x1080. Mage-VL-Cowork was trained on 1920x1080 OSWorld screens; on smaller screenshots its clicks drift left (1-2/5 hits at 1280x800 vs 5/5 at 1920x1080 in our grounding test).

Chrome runs with its own profile in `~/.cache/mage-computer-use/profile`, never your normal profile. There is no shell tool.

## What it took to make Mage work

Measured by replaying the same failing step (click huggingface.co's search bar, y = 14-50 px, one step into a task):

| Tool result content | Click y |
|---|---|
| text note ("Current page: ...") + screenshot | 111-113 px (miss) |
| screenshot only | 42 px (hit) |
| Mage's exact harness format | 42 px (hit) |

The system prompt and tool schema made no difference. The other changes:

- **Screenshot history.** The WebUI kept every action's screenshot (~2,400 tokens each; Mage often does 3-4 actions per turn). The `mage-vl-cowork` branch adds a WebUI setting, *Tool images kept in context* (Settings > Tools), that collapses older tool screenshots to "[This screenshot has been collapsed.]", as Mage's harness does.
- **Ending the task.** Mage ends with `computer_use(action="terminate")` and then keeps calling tools. The branch's WebUI stops its agentic loop when a tool result sets `_meta["llama.cpp/endAgenticLoop"]`.

## Run it on the Mac (WebUI)

1. Build the WebUI from the `mage-vl-cowork` branch (once, or after changes):

   ```bash
   cd ~/Documents/python_projecten/llama.cpp/tools/ui && npm ci && npm run build
   ```

2. Start the model side by side with the Llama app, serving that WebUI with Mage presets:

   ```bash
   cd ~/Documents/python_projecten/llama.cpp
   ./build-metal/bin/llama serve -m models/mage-vl-cowork/Mage-VL-Cowork-Q8_0.gguf --mmproj models/mage-vl-cowork/mmproj-Mage-VL-Cowork-F16.gguf --host 127.0.0.1 --port 9932 -c 16384 -np 1 --jinja -a Mage-VL-Cowork --path tools/ui/dist --ui-config-file ~/Documents/python_projecten/mage-computer-use/webui-config.json
   ```

   ~7 GB of memory; on an M-series Mac a step takes ~10 s.

3. Start the MCP server (opens a Chrome window on the first action):

   ```bash
   cd ~/Documents/python_projecten/mage-computer-use && uv run server.py
   ```

4. Open `http://127.0.0.1:9932`:
   - MCP (sidebar) > Add New Server, URL `http://127.0.0.1:8765/mcp`.
   - Settings > Tools: untick the built-in Browser tools so Mage only sees `computer-use`.
   - Ask for a task, e.g. "Go to huggingface.co, search for Mage-VL-Cowork and open the model page of microsoft/Mage-VL-Cowork". Allow `computer_use` when asked ("Always allow tool" avoids a prompt per action).

## Or with the agent loop

Same model server (no `--path` needed), then:

```bash
uv run agent_loop.py "Go to huggingface.co and open the model page of microsoft/Mage-VL-Cowork"
```

## Known issues

- A WebUI chat starts without a screenshot (in training the task always came with one), so Mage's first step is blind and its first click often misses. Asking it in the system message to take a screenshot first did not work. What made it fail was the follow-up: typing into the page itself (spaces scroll down a screen) hid the search box, after which Mage thought it was on a desktop (Chrome icon, Alt+Tab, Super key). The server therefore skips `type` when no text field has focus ("Skipped typing ...: no text field selected"); Mage then sees the unchanged page and clicks the right field on step 2. The agent loop sends a screenshot with the task, like training.
- The WebUI samples with its own defaults; Mage's harness uses temperature 0.7, top_p 0.8.
