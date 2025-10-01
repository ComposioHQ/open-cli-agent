from typing import List, Dict, Any, Optional
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from composio import Composio
from composio_langchain import LangchainProvider
from pydantic import BaseModel, Field
import os
import asyncio


class AgentConfig(BaseModel):
    model_name: str = Field(default="gpt-5", description="OpenAI model to use")
    temperature: float = Field(default=0.1, description="Model temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens")
    user_id: str = Field(default="default", description="User ID for tool access")


class ToolRouterAgent:
    """Agent that uses Composio Tool Router with LangGraph"""
    
    def __init__(
        self,
        composio_api_key: str,
        openai_api_key: str,
        user_id: str = "default",
        config: Optional[AgentConfig] = None,
    ):
        self.config = config or AgentConfig()
        self.composio_api_key = composio_api_key
        self.openai_api_key = openai_api_key
        self.user_id = user_id
        self.session = None
        self.graph = None
        self.model = None
        self.client = None
        
        # Set OpenAI API key in environment
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
    def create_session(self, toolkits: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Create a tool router session"""
        composio = Composio(
            api_key=self.composio_api_key,
            provider=LangchainProvider()
        )

        # Create session with or without specific toolkits
        if toolkits and len(toolkits) > 0:
            self.session = composio.experimental.tool_router.create_session(
                user_id=self.user_id,
                toolkits=toolkits
            )
        else:
            # Without toolkits - use all available
            self.session = composio.experimental.tool_router.create_session(
                user_id=self.user_id
            )
        return self.session

    async def setup_graph_async(self):
        """Setup the LangGraph with MCP client and tools (async)"""
        if not self.session:
            raise ValueError("Session not created. Call create_session first.")
        
        # Initialize the model
        self.model = init_chat_model(f"openai:{self.config.model_name}")
        
        # Set up MCP client
        self.client = MultiServerMCPClient(
            {
                "tool_router": {
                    "url": self.session["url"],
                    "transport": "streamable_http",
                }
            }
        )
        tools = await self.client.get_tools()
        
        # Bind tools to model
        model_with_tools = self.model.bind_tools(tools)
        
        # Create ToolNode
        tool_node = ToolNode(tools)
        
        def should_continue(state: MessagesState):
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return "tools"
            return END
        
        # Define call_model function
        async def call_model(state: MessagesState):
            messages = state["messages"]
            response = await model_with_tools.ainvoke(messages)
            return {"messages": [response]}
        
        # Build the graph
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", tool_node)
        
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges(
            "call_model",
            should_continue,
        )
        builder.add_edge("tools", "call_model")
        
        # Compile the graph
        self.graph = builder.compile()
        
        return True
    
    def _extract_message_content(self, response: Dict[str, Any]) -> str:
        """Extract content from agent response"""
        messages = response.get("messages", [])
        if messages:
            final_message = messages[-1]
            return getattr(final_message, 'content', str(final_message))
        return "No response generated"
    
    async def run_async(self, query: str) -> str:
        """Run the agent asynchronously"""
        if not self.graph:
            return "Agent not initialized. Please create session and setup graph first."
        
        try:
            response = await self.graph.ainvoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            return self._extract_message_content(response)
        except Exception as e:
            return f"Error running agent: {e}"


def create_tool_router_agent(
    composio_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    user_id: str = "default",
    config: Optional[AgentConfig] = None,
) -> ToolRouterAgent:
    """Factory function to create a ToolRouterAgent"""
    
    # Get API keys from environment if not provided
    composio_key = composio_api_key or os.getenv("COMPOSIO_API_KEY")
    openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    
    if not composio_key:
        raise ValueError("COMPOSIO_API_KEY is required")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is required")
    
    return ToolRouterAgent(
        composio_api_key=composio_key,
        openai_api_key=openai_key,
        user_id=user_id,
        config=config,
    )