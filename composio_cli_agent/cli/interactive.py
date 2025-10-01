"""Interactive mode functionality for Composio CLI Agent"""

import asyncio
from typing import Optional, List, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML

from ..utils.config import config
from ..agent.composio_agent import create_tool_router_agent, AgentConfig
from ..utils.banner import print_banner

console = Console()


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
        
        console.print("\n[bold cyan]Tips for getting started:[/bold cyan]\n")
        console.print("  1. Type your question or task naturally")
        console.print("  2. Use the agent with access to 500+ tools via Composio")
        console.print("  3. Type [bold]?[/bold] for shortcuts or [bold]/exit[/bold] to quit\n")
        
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
        console.print(f"[dim]User ID: {effective_user_id}[/dim]")
        
        try:
            agent_config = AgentConfig(
                model_name=self.model or config.default_model,
                temperature=self.temperature or config.default_temperature,
                user_id=effective_user_id,
            )
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                # Create agent
                progress.add_task("Creating agent...", total=None)
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
            
            console.print(f"[green]✓[/green] Agent ready!\n")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error initializing agent: {e}[/red]")
            return False
    
    def show_help(self) -> None:
        """Display available shortcuts"""
        console.print("\n[bold cyan]Available shortcuts:[/bold cyan]")
        console.print("  [bold]?[/bold]         - Show this help")
        console.print("  [bold]/exit[/bold]     - Exit interactive mode")
        console.print("  [bold]/clear[/bold]    - Clear conversation history")
        console.print("  [bold]/history[/bold]  - Show conversation history")
        console.print()
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
        console.print("\n[dim]Conversation history cleared.[/dim]\n")
    
    def show_history(self) -> None:
        """Display conversation history"""
        if not self.conversation_history:
            console.print("\n[dim]No conversation history yet.[/dim]\n")
        else:
            console.print("\n[bold cyan]Conversation History:[/bold cyan]\n")
            for role, msg in self.conversation_history:
                prefix = ">" if role == "user" else "●"
                console.print(f"{prefix} {msg}\n")
    
    def handle_command(self, user_input: str) -> bool:
        """Handle special commands. Returns True if should continue loop."""
        if user_input == "?":
            self.show_help()
            return True
        
        if user_input in ["/exit", "/quit", "exit", "quit"]:
            console.print("\n[dim]👋 Goodbye![/dim]\n")
            return False
        
        if user_input == "/clear":
            self.clear_history()
            return True
        
        if user_input == "/history":
            self.show_history()
            return True
        
        return True
    
    async def process_user_input(self, user_input: str) -> None:
        """Process user input and get agent response"""
        # Add to history
        self.conversation_history.append(("user", user_input))
        
        # Get response from agent
        console.print("\n[bold white]●[/bold white] ", end="")
        try:
            response = await self.agent.run_async(user_input)
            console.print(response)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            response = f"Error: {e}"
        
        self.conversation_history.append(("agent", response))
        console.print()
    
    async def run_main_loop(self) -> None:
        """Run the main interaction loop"""
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
                console.print("\n\n[dim]👋 Goodbye![/dim]\n")
                break
            except Exception as e:
                console.print(f"\n[red]Unexpected error: {e}[/red]\n")
                continue
        
        console.print("[dim]? for shortcuts[/dim]\n")


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