# tuichat

A full-screen terminal chat client for **local models (Ollama)** and **cloud models (OpenAI or any OpenAI-compatible server)**.

```text
  🦙 ollama (deepseek-r1:14b) • Local • Private
────────────────────────────────────────────────────────────────────────────────
  ──────────────────────────────
  👤 You
  Explain what a mutex is in one sentence.

  🦙 ollama
  A mutex is a lock that lets only one thread use a shared resource at a time.

────────────────────────────────────────────────────────────────────────────────
  ┌─────| Enter=send │ Tab=scroll │ ^A=copy all │ ^Y=copy reply │ ^C=exit |─────┐
  │❯                                                                            │
  └─────────────────────────────────────────────────────────────────────────────┘
```

*(Layout is a real capture from the app; the conversation text is illustrative.)*

## Where this came from

I wrote two near-identical scripts for my own use, `deepseek-cli` (Ollama) and `openai-cli` (OpenAI). About 95% of their code was duplicated. This repository merges them into one tested package with a small backend interface, so adding another model server means writing one class.

It is a thin client, not an AI system: all the intelligence is in the model you connect it to. The value here is the interface and the engineering around it.

## Features

- Streaming replies in a scrollable full-screen layout with mouse support
- **Queue messages while a reply is still streaming**; they are answered in order
- Hides `<think>…</think>` reasoning blocks (for example deepseek-r1) and shows a spinner meanwhile. The filter handles tags split across stream chunks
- `/read file1 file2` loads text files into the conversation (quotes, escaped spaces, `~` and drag-and-drop `@` prefixes all work)
- `/model <name>` switches model without restarting; `/clear` resets the conversation
- Copy the last reply, your last message, or the whole chat to the clipboard (macOS, Linux and Windows tools are detected)
- Basic markup: `**bold**` and `__bold__` render as bold in replies
- A failed request is shown in the chat and removed from the history, so a retry is not duplicated

## Install and run

```bash
pip install -e ".[ollama]"      # local models; or ".[openai]"; or ".[ollama,openai]"

tuichat                          # Ollama, default model deepseek-r1:14b
tuichat -m llama3.2              # another local model (must be pulled in Ollama)

export OPENAI_API_KEY="..."
tuichat -b openai -m gpt-4o      # OpenAI
tuichat -b openai --base-url http://localhost:1234/v1 -m my-model   # any OpenAI-compatible server

tuichat -b echo                  # offline demo backend: no model, no network
```

Requires Python 3.10+. For Ollama, run `ollama serve` and `ollama pull <model>` first.

| Key | Action |
|---|---|
| Enter | send |
| Alt+Enter (Esc, Enter) | new line |
| Tab | switch between input and scroll area |
| PageUp / PageDown | scroll |
| Ctrl+Y / Ctrl+U / Ctrl+A | copy last reply / your last message / whole chat |
| Ctrl+C or Ctrl+D | exit |

## Design

```
tuichat/
  app.py         the prompt_toolkit application: layout, keys, conversation loop, spinner
  backends.py    Backend interface; Ollama, OpenAI-compatible and Echo implementations
  thinking.py    streaming <think> filter (buffers partial tags across chunks)
  text.py        wrapping, bold markup, user-line marking (pure functions)
  files.py       /read: path parsing and file loading
  commands.py    slash-command parsing
  clipboard.py   cross-platform clipboard
  cli.py         argument parsing
```

A backend implements one method, `stream(messages)`, an async generator of text chunks, and raises `BackendError` with a message worth showing. The rest of the app does not know which model is behind it.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

49 tests, none of which need a model or a network. Backends are tested against fakes of the `ollama` and `openai` clients. The application itself is tested headlessly with a fake terminal, including a full run that types a message, waits for the reply and exits with Ctrl-D.

## Limitations

- The Ollama and OpenAI adapters are tested against fakes of their client libraries. I ran the Ollama path against a real local model by hand; the OpenAI path has not been run against the live API in this repository.
- `<think>` filtering is tag-based. A model that marks reasoning differently is shown as-is.
- The whole conversation is resent on every turn. There is no context-window management, so a very long chat will eventually exceed the model's limit.
- Conversations are not saved to disk.
- Text rendering is plain: only bold markup is interpreted, not code blocks or other Markdown.

## AI assistance

The two original scripts were written by me. This merged and refactored version, with its tests, was developed with AI assistance (Claude Code). I specified the design and reviewed the result.

## License

MIT.
