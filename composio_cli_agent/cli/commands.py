"""CLI commands for Composio CLI Agent"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional
import asyncio

from ..utils.banner import print_banner
from ..utils.config import config
from ..agent.composio_agent import create_tool_router_agent, AgentConfig
from composio import Composio
from composio_langchain import LangchainProvider
from .interactive import run_interactive_mode

console = Console()


def require_composio_key() -> Optional[str]:
    """Get Composio API key or show error and return None"""
    composio_key = config.composio_api_key
    if not composio_key:
        console.print("❌ Composio API key not found. Run 'composio-cli setup' first.", style="red")
        return None
    return composio_key


def get_composio_client(api_key: str) -> Composio:
    """Create a Composio client instance"""
    return Composio(api_key=api_key, provider=LangchainProvider())


@click.group(invoke_without_command=True)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose: bool):
    """Composio CLI Agent - A general-purpose CLI agent with authentication management"""
    if verbose:
        config.set('verbose', True)
    
    # If no subcommand is provided, run interactive mode by default
    if ctx.invoked_subcommand is None:
        ctx.invoke(interactive)


@cli.command()
@click.option('--composio-key', help='Composio API key')
@click.option('--openai-key', help='OpenAI API key')
@click.option('--email', help='Your email address')
def setup(composio_key: Optional[str], openai_key: Optional[str], email: Optional[str]):
    """Setup API keys and configuration"""
    print_banner()
    
    if not composio_key:
        composio_key = click.prompt('Enter your Composio API key', hide_input=True)
    
    if not openai_key:
        openai_key = click.prompt('Enter your OpenAI API key', hide_input=True)
    
    if not email:
        email = click.prompt('Enter your email address')
    
    config.set_api_key('composio', composio_key)
    config.set_api_key('openai', openai_key)
    config.set('user_email', email)
    config.set('default_user_id', email)  # Use email as default user ID
    
    console.print("✅ Configuration saved successfully!", style="green")


@cli.command()
def toolkits():
    """List available toolkits"""
    
    composio_key = require_composio_key()
    if not composio_key:
        return
    
    try:
        # Initialize Composio client directly
        composio_client = get_composio_client(composio_key)
        
        with console.status("[bold green]Fetching toolkits..."):
            # Get available toolkits using the Composio client
            available_toolkits = composio_client.tools.get_available_toolkits()
        
        if available_toolkits:
            console.print("\n📦 Available Toolkits:", style="bold blue")
            for toolkit in sorted(available_toolkits):
                console.print(f"  • {toolkit}", style="green")
        else:
            console.print("No toolkits available.", style="yellow")
            
    except Exception as e:
        console.print(f"❌ Error listing toolkits: {e}", style="red")


@cli.command()
def config_show():
    """Show current configuration"""
    
    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Default User ID", config.default_user_id)
    table.add_row("Default Model", config.default_model)
    table.add_row("Default Temperature", str(config.default_temperature))
    table.add_row("Verbose", str(config.verbose))
    table.add_row("Composio API Key", "Set" if config.composio_api_key else "Not set")
    table.add_row("OpenAI API Key", "Set" if config.openai_api_key else "Not set")
    
    console.print(table)


@click.option('--toolkits', default=None, help='Comma-separated list of toolkits to filter (optional)')
@click.option('--auth-configs', default=None, help='Comma-separated list of auth config IDs (optional, use with --toolkits)')
@click.option('--user-id', default=None, help='User ID for tool access')
@click.option('--model', default=None, help='OpenAI model to use')
@click.option('--temperature', default=None, type=float, help='Model temperature')
def interactive(toolkits: Optional[str] = None, auth_configs: Optional[str] = None, 
                user_id: Optional[str] = None, model: Optional[str] = None, temperature: Optional[float] = None):
    """Start interactive chat session (Claude Code-like experience)"""
    asyncio.run(run_interactive_mode(toolkits, auth_configs, user_id, model, temperature))



