# Building a Minimal MCP Server and Client with FastMCP 2.0

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) standardizes how AI assistants discover and safely call
external tools. [FastMCP 2.0](https://pypi.org/project/fastmcp/) is a batteries-included Python framework that streamlines both
server and client development while remaining fully compatible with the protocol. This note walks through creating a single
`greet` tool that runs inside a FastMCP server and exercising it from a FastMCP client.

## Prerequisites

- Python 3.10+
- `pipx` (recommended) or `pip`
- The `fastmcp` package pinned to the 2.x series for both the server and the client

Install FastMCP globally with `pipx`:

```bash
pipx install "fastmcp==2.*"
```

or within a virtual environment that you manage manually:

```bash
python -m venv .venv
source .venv/bin/activate
pip install "fastmcp==2.*"
```

FastMCP bundles an async runtime, a schema-first tool decorator, and a modern client that can speak to any MCP-compliant
server.

## Project Layout

```
fastmcp-demo/
├── client.py
└── server.py
```

The `server.py` script exposes the `greet` tool, while `client.py` starts the server as a subprocess and issues requests through
the MCP transport.

## Implementing the FastMCP Server

Create `server.py` with the following contents:

```python
"""Minimal FastMCP 2.0 server exposing a single greeting tool."""
import asyncio
from dataclasses import dataclass

from fastmcp import FastMCP, tool

mcp = FastMCP("fast-greeter")


@dataclass
class GreetArgs:
    """Typed input schema for the greet tool."""

    name: str


@tool(mcp)
async def greet(args: GreetArgs) -> str:
    """Return a friendly greeting for the provided name."""

    person = args.name.strip() or "there"
    return f"Hello, {person}!"


def main() -> None:
    asyncio.run(mcp.run_stdio())


if __name__ == "__main__":
    main()
```

Key FastMCP 2.0 features illustrated above:

- `FastMCP("fast-greeter")` registers the server with a human-readable identifier that will surface in client handshakes.
- `@tool(mcp)` wires the coroutine into the server, automatically deriving the MCP JSON schema from the `dataclass` argument.
- `mcp.run_stdio()` launches the event loop and keeps the server alive on the standard input/output transport expected by most
  assistants.

## Implementing the FastMCP Client

Create `client.py` next:

```python
"""Minimal FastMCP 2.0 client that calls the greet tool."""
import asyncio
from fastmcp import FastMCPClient


async def main() -> None:
    async with FastMCPClient.from_subprocess(["python", "server.py"]) as session:
        tools = await session.list_tools()
        print("Tools:", tools)

        response = await session.call_tool("greet", {"name": "Ada"})
        print("Tool response:", response)


if __name__ == "__main__":
    asyncio.run(main())
```

Highlights:

- `FastMCPClient.from_subprocess()` launches the server as a subprocess and negotiates the MCP handshake automatically.
- `session.list_tools()` returns the tool metadata exposed by the server.
- `session.call_tool("greet", ...)` invokes the server tool and returns the structured response that the assistant will see.

## Running the Demo

1. Start the server in one terminal tab (or leave it to the client to spawn automatically):

   ```bash
   python server.py
   ```

   The process waits for MCP requests while streaming logs to the console.

2. In a second terminal, execute the client:

   ```bash
   python client.py
   ```

   Expected output resembles the following:

   ```text
   Tools: [{'name': 'greet', 'description': 'Return a friendly greeting', ...}]
   Tool response: {'type': 'text', 'text': 'Hello, Ada!'}
   ```

3. Stop the server with `Ctrl+C` when finished.

## Extending the Example

- Add more tools by defining additional `@tool(mcp)` coroutines and associated dataclasses for their inputs.
- Switch to transports such as WebSocket or TCP by replacing `mcp.run_stdio()` with the corresponding `fastmcp` helper.
- Use FastMCP's dependency-injection hooks (for example, `@tool(mcp, inject=[...])`) to share database handles or API clients
  across tools.
- Integrate with the `mcp` developer CLI (`mcp run ...`) or other MCP-compliant clients by pointing them at `server.py`.

FastMCP 2.0 keeps the boilerplate minimal while providing type-safe tooling definitions, making it an excellent starting point
for more sophisticated assistants.
