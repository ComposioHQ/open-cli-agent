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
    model_name: str = Field(default="gpt-4.1", description="OpenAI model to use")
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
        self.conversation_history = []
        
        # Set OpenAI API key in environment
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # System prompt - Claude Code style for general purpose tasks
        self.system_prompt = """You are a highly capable AI assistant with access to 500+ tools and integrations through Composio. Your goal is to help users accomplish their tasks efficiently and completely.

        Key principles:
        - Be proactive and action-oriented: Execute tasks fully rather than just explaining what could be done
        - Minimize questions: Only ask clarifying questions when absolutely necessary for critical missing information
        - Think step-by-step but act decisively: Break down complex tasks but execute each step without hesitation
        - Use tools liberally: You have access to many apps and services - use them to get things done
        - Complete the entire task: Don't stop halfway or leave steps for the user to finish manually
        - Be concise: Communicate progress briefly and clearly without excessive explanations
        - Handle errors gracefully: If something fails, try alternative approaches automatically

        When given a task:
        1. Understand the full scope of what needs to be done
        2. Plan the necessary steps mentally
        3. Execute all steps using available tools
        4. Provide a brief summary of what was accomplished

        Focus on execution and results, not on asking permission for every step.

        """
        
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
        
        # Initialize the model with streaming enabled
        self.model = init_chat_model(
            f"openai:{self.config.model_name}",
            temperature=self.config.temperature,
            streaming=True,
        )
        
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
        
        # Bind tools to model with streaming
        model_with_tools = self.model.bind_tools(tools)
        
        # Create ToolNode
        tool_node = ToolNode(tools)
        
        def should_continue(state: MessagesState):
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return "tools"
            return END
        
        # Define call_model function with streaming
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
        
        # Compile the graph with streaming support
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
        """Run the agent asynchronously (non-streaming)"""
        if not self.graph:
            return "Agent not initialized. Please create session and setup graph first."
        
        try:
            response = await self.graph.ainvoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            return self._extract_message_content(response)
        except Exception as e:
            return f"Error running agent: {e}"
    
    def add_user_message(self, content: str):
        """Add a user message to conversation history"""
        self.conversation_history.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str):
        """Add an assistant message to conversation history"""
        self.conversation_history.append({"role": "assistant", "content": content})
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
    
    def get_messages_for_llm(self):
        """Get all messages including system prompt for the LLM"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        return messages
    
    async def run_async_stream(self, query: str):
        """Run the agent asynchronously with detailed streaming events"""
        if not self.graph:
            yield {"type": "error", "content": "Agent not initialized."}
            return
        
        try:
            tool_calls_seen = set()
            assistant_response = ""
            
            # Add user message to history
            self.add_user_message(query)
            
            # Use astream_events with stream_mode for immediate streaming
            async for event in self.graph.astream_events(
                {"messages": self.get_messages_for_llm()},
                version="v2",
                stream_mode="values"
            ):
                kind = event.get("event")
                
                # Stream AI message chunks immediately
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        assistant_response += chunk.content
                        yield {"type": "ai_chunk", "content": chunk.content}
                
                # Tool execution starts - use this for immediate notification
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_key = f"{tool_name}_{event.get('run_id', '')}"
                    
                    if tool_key not in tool_calls_seen:
                        tool_calls_seen.add(tool_key)
                        yield {
                            "type": "tool_call_start",
                            "tool_name": tool_name,
                            "tool_input": event.get("data", {}).get("input", {})
                        }
                
                # Tool execution complete
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    output = event.get("data", {}).get("output")
                    yield {
                        "type": "tool_call_end",
                        "tool_name": tool_name,
                        "output": output
                    }
            
            if assistant_response:
                self.add_assistant_message(assistant_response)
                    
        except Exception as e:
            yield {"type": "error", "content": f"Error: {e}"}


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