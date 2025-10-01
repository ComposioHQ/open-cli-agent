# Open CLI Agent

A powerful, interactive CLI agent that provides access to 500+ tools and integrations through [Composio](https://composio.dev). Built with LangChain and LangGraph for sophisticated AI workflows with a Claude Code-like experience.

## Features

- **Interactive Chat Interface**: Natural language interaction with an AI agent
- **500+ Integrations**: Access to tools via Composio's extensive toolkit ecosystem
- **Authentication Management**: Secure API key management and configuration
- **Rich Terminal UI**: Beautiful, colorful output with progress indicators
- **Tool Router**: Intelligent tool selection and execution via Composio's experimental tool router
- **Conversation History**: Persistent chat history with shortcuts and commands
- **Flexible Configuration**: Customizable models, temperature, and toolkit filtering

## Quick Start

### Installation

```bash
# Clone and install
git clone <repository-url>
cd cli-agent
pip install -e .
```

### Setup

First, configure your API keys:

```bash
composio-cli setup
```

You'll be prompted for:
- **Composio API Key**: Get yours from [composio.dev](https://composio.dev)
- **OpenAI API Key**: Required for the LLM
- **Email**: Used as default user ID

### Basic Usage

Start the CLI agent:

```bash
composio-cli
```

Other available commands:

```bash
# List available toolkits
composio-cli toolkits

# Show current configuration
composio-cli config-show
```

You can also pass options directly to the main command:

```bash
# Use specific toolkits
composio-cli --toolkits "github,slack" --auth-configs "config1,config2"

# Use different model settings
composio-cli --model "gpt-5"
```

## Usage

The CLI provides a Claude Code-like experience:

```
> What GitHub repositories do I have access to?
● [Agent fetches and lists your repositories using GitHub tools]

> Send a message to my team Slack channel about the project update
● [Agent uses Slack integration to send the message]
```

### Available Commands

| Command | Description |
|---------|-------------|
| `?` | Show help and shortcuts |
| `/exit` | Exit the CLI |
| `/clear` | Clear conversation history |
| `/history` | Show conversation history |

## Configuration

### Command Line Options

```bash
composio-cli [OPTIONS]

Options:
  --toolkits TEXT       Comma-separated list of toolkits to filter
  --auth-configs TEXT   Comma-separated list of auth config IDs
  --user-id TEXT        User ID for tool access
  --model TEXT          OpenAI model to use (default: gpt-5)
  --temperature FLOAT   Model temperature (default: 0.1)
  --verbose, -v         Enable verbose output
```

### Environment Variables

You can also set configuration via environment variables:

```bash
export COMPOSIO_API_KEY="your_composio_key"
export OPENAI_API_KEY="your_openai_key"
```

## Architecture

- **LangChain**: Core LLM framework for chat models
- **LangGraph**: State management and workflow orchestration
- **Composio Tool Router**: Intelligent tool selection and execution
- **MCP (Model Context Protocol)**: For tool communication
- **Rich**: Beautiful terminal formatting and progress indicators

## Available Toolkits

The agent has access to 500+ tools across various categories:

- **Communication**: Slack, Discord, Email
- **Development**: GitHub, GitLab, Jira, Linear
- **Productivity**: Google Workspace, Microsoft 365, Notion
- **Marketing**: Social media platforms, analytics tools
- **And many more...

Run `composio-cli toolkits` to see all available integrations.

## Development

### Project Structure

```
composio_cli_agent/
├── agent/              # Core agent implementation
│   └── composio_agent.py
├── cli/                # CLI commands and interface
│   ├── commands.py     # Main CLI commands
│   └── interactive.py  # Interactive mode logic
├── utils/              # Utilities
│   ├── banner.py       # Welcome banner
│   └── config.py       # Configuration management
└── main.py             # Entry point
```

### Requirements

- Python >= 3.11
- Dependencies managed via `pyproject.toml`
- Uses UV for fast package resolution

## Links

- **[Composio Platform](https://composio.dev)** - Get your API key and explore integrations
- **[Documentation](https://docs.composio.dev)** - Comprehensive guides and API reference
- **[GitHub Repository](https://github.com/composiohq/composio)** - Main Composio project

## License

MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

- **Issues**: Report bugs or request features via GitHub issues
- **Documentation**: Check [docs.composio.dev](https://docs.composio.dev) for detailed guides
- **Community**: Join the Composio community for support and discussions