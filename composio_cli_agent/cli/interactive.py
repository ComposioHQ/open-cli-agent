"""Interactive mode functionality for Composio CLI Agent"""

import asyncio
from typing import Optional, List, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
import sys

from ..utils.config import config
from ..agent.composio_agent import create_tool_router_agent, AgentConfig
from ..utils.banner import print_banner

# Disable buffering for immediate output, disable auto-highlighting
console = Console(force_terminal=True, force_interactive=True, highlight=False)


class InteractiveSession:
    """Manages an interactive CLI session"""
    
    def __init__(self, toolkits: Optional[str] = None, auth_configs: Optional[str] = None,
                 user_id: Optional[str] = None, model: Optional[str] = None, 
                 temperature: Optional[float] = None):
        self.toolkits = toolkits
        self.auth_configs = auth_configs
        self.user_id = user_id
        self.model = model
        self.temperature = temperature
        self.conversation_history: List[Tuple[str, str]] = []
        self.agent = None
        
        # Setup autocomplete for commands
        commands = ['/exit', '/quit', '/clear', '/history', '?']
        self.completer = WordCompleter(commands, ignore_case=True, sentence=True)
        self.prompt_session = PromptSession(completer=self.completer)
        
    def show_welcome_banner(self) -> None:
        """Display welcome banner and tips"""
        print_banner()
        console.print()
        console.print("[dim]Access to 500+ tools • Type ? for help • /exit to quit[/dim]")
        
    def get_effective_user_id(self) -> str:
        """Determine the effective user ID to use"""
        return (self.user_id or 
                getattr(config, 'user_email', None) or 
                config.default_user_id)
    
    def validate_toolkit_config(self) -> Optional[List[dict]]:
        """Validate and parse toolkit configuration"""
        if self.toolkits and self.auth_configs:
            toolkit_list = [t.strip() for t in self.toolkits.split(',')]
            auth_config_list = [a.strip() for a in self.auth_configs.split(',')]
            
            if len(toolkit_list) != len(auth_config_list):
                console.print("❌ Number of toolkits must match number of auth configs", style="red")
                return None
            
            toolkits_config = [
                {'toolkit': toolkit, 'auth_config_id': auth_config}
                for toolkit, auth_config in zip(toolkit_list, auth_config_list)
            ]
            console.print(f"[dim]Filtering to toolkits: {self.toolkits}[/dim]")
            return toolkits_config
        else:
            console.print("[dim]Using all available tools[/dim]")
            return []
    
    async def initialize_agent(self, composio_key: str, openai_key: str) -> bool:
        """Initialize the agent and session"""
        effective_user_id = self.get_effective_user_id()
        
        try:
            agent_config = AgentConfig(
                model_name=self.model or config.default_model,
                temperature=self.temperature or config.default_temperature,
                user_id=effective_user_id,
            )
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                # Create agent
                self.agent = create_tool_router_agent(
                    composio_api_key=composio_key,
                    openai_api_key=openai_key,
                    user_id=effective_user_id,
                    config=agent_config,
                )
                
                # Validate toolkit configuration
                toolkits_config = self.validate_toolkit_config()
                if toolkits_config is None:
                    return False
                
                progress.stop()
            
            # Create session and setup graph
            self.agent.create_session(toolkits_config)
            await self.agent.setup_graph_async()
            
            console.print(f"[green]✓[/green] Agent ready!")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error initializing agent: {e}[/red]")
            return False
    
    def show_help(self) -> None:
        """Display available shortcuts"""
        console.print()
        console.print()
        console.print("[bold cyan]Available shortcuts:[/bold cyan]")
        console.print("  [bold]?[/bold]         - Show this help")
        console.print("  [bold]/exit[/bold]     - Exit interactive mode")
        console.print("  [bold]/clear[/bold]    - Clear conversation history")
        console.print("  [bold]/history[/bold]  - Show conversation history")
        console.print()
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
        if self.agent:
            self.agent.clear_conversation_history()
        console.print()
        console.print()
        console.print("[dim]Conversation history cleared.[/dim]")
        console.print()
    
    def show_history(self) -> None:
        """Display conversation history"""
        console.print()
        console.print()
        if not self.conversation_history:
            console.print("[dim]No conversation history yet.[/dim]")
        else:
            console.print("[bold cyan]Conversation History:[/bold cyan]")
            console.print()
            for role, msg in self.conversation_history:
                prefix = ">" if role == "user" else "●"
                console.print(f"{prefix} {msg}")
        console.print()
    
    def handle_command(self, user_input: str) -> bool:
        """Handle special commands. Returns True if should continue loop."""
        if user_input == "?":
            self.show_help()
            return True
        
        if user_input in ["/exit", "/quit", "exit", "quit"]:
            console.print()
            console.print()
            console.print("[dim]👋 Goodbye![/dim]")
            console.print()
            return False
        
        if user_input == "/clear":
            self.clear_history()
            return True
        
        if user_input == "/history":
            self.show_history()
            return True
        
        return True
    
    async def process_user_input(self, user_input: str) -> None:
        """Process user input and stream agent response with tool calls"""
        # Add to history
        self.conversation_history.append(("user", user_input))
        
        console.print()
        
        full_response = ""
        ai_message = ""
        first_chunk = True
        live = None
        active_tools = set()
        
        try:
            # Start with spinner
            spinner = Spinner("dots", text="", style="cyan")
            live = Live(spinner, console=console, refresh_per_second=20, transient=True)
            live.start()
            
            async for event in self.agent.run_async_stream(user_input):
                event_type = event.get("type")
                
                if event_type == "ai_chunk":
                    # Stream AI text as it comes
                    if first_chunk:
                        if live and live.is_started:
                            live.stop()
                        console.print("● ", end="", style="bold cyan")
                        first_chunk = False
                    
                    chunk = event.get("content", "")
                    console.print(chunk, end="")
                    ai_message += chunk
                
                elif event_type == "tool_call_start":
                    # Compact tool notification
                    tool_name = event.get("tool_name", "unknown")
                    active_tools.add(tool_name)
                    
                    if not first_chunk and ai_message:
                        console.print()
                    elif first_chunk:
                        if live and live.is_started:
                            live.stop()
                        first_chunk = False
                    
                    console.print(f"  [dim]🔧 {tool_name}...[/dim]", end="")
                
                elif event_type == "tool_call_end":
                    tool_name = event.get("tool_name", "unknown")
                    if tool_name in active_tools:
                        console.print(f" [dim green]✓[/dim green]")
                        active_tools.remove(tool_name)
                
                elif event_type == "error":
                    if live and live.is_started:
                        live.stop()
                    error_msg = event.get("content", "Unknown error")
                    console.print()
                    console.print(f"[red]✗ {error_msg}[/red]")
                    full_response = f"Error: {error_msg}"
                
        except Exception as e:
            if live and live.is_started:
                live.stop()
            console.print()
            console.print(f"[red]✗ {e}[/red]")
            full_response = f"Error: {e}"
        finally:
            if live and live.is_started:
                live.stop()
        
        # Save response to history
        if not full_response:
            full_response = ai_message
        
        self.conversation_history.append(("agent", full_response))
        # Add spacing after response
        console.print()
        console.print()
    
    async def run_main_loop(self) -> None:
        """Run the main interaction loop"""
        console.print()
        console.print("[dim]Type your message or ? for help[/dim]")
        console.print()
        
        while True:
            try:
                # Show prompt and get input with autocomplete
                user_input = await self.prompt_session.prompt_async(
                    HTML('<ansibrightcyan><b>></b></ansibrightcyan> ')
                )
                user_input = user_input.strip()
                
                # Handle empty input
                if not user_input:
                    continue
                
                # Handle special commands
                if not self.handle_command(user_input):
                    break
                
                # Skip processing if it was a command
                if user_input.startswith("?") or user_input.startswith("/"):
                    continue
                
                # Process regular user input
                await self.process_user_input(user_input)
                
            except (KeyboardInterrupt, EOFError):
                console.print()
                console.print()
                console.print("[dim]👋 Goodbye![/dim]")
                console.print()
                break
            except Exception as e:
                console.print()
                console.print()
                console.print(f"[red]Unexpected error: {e}[/red]")
                console.print()
                continue


async def run_interactive_mode(toolkits: Optional[str] = None, auth_configs: Optional[str] = None,
                               user_id: Optional[str] = None, model: Optional[str] = None, 
                               temperature: Optional[float] = None) -> None:
    """Main entry point for interactive mode"""
    from .commands import require_composio_key
    
    # Create session
    session = InteractiveSession(toolkits, auth_configs, user_id, model, temperature)
    
    # Show welcome
    session.show_welcome_banner()
    
    # Validate API keys
    composio_key = require_composio_key()
    if not composio_key:
        return
    
    openai_key = config.openai_api_key
    if not openai_key:
        console.print("❌ OpenAI API key not found. Run 'composio-cli setup' first.", style="red")
        return
    
    # Initialize agent
    if not await session.initialize_agent(composio_key, openai_key):
        return
    
    # Run main loop
    await session.run_main_loop()