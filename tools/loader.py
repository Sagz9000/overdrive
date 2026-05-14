"""
YAML Tool Descriptor Loader
=============================
Inspired by Dreadnode's Robopages pattern: tool interfaces are defined in
YAML for deterministic schema description, while execution logic stays in
Python. This reduces hallucination from SLMs by providing clean, structured
tool definitions.

This module:
1. Reads all *.yaml files from the tools/ directory
2. Validates their schemas
3. Generates metadata that can be used to document tools for LLM prompts
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ToolParameter:
    """A single parameter for a tool."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDescriptor:
    """Parsed YAML tool descriptor."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    returns_type: str = "string"
    returns_description: str = ""
    container: bool = False
    category: str = "general"
    
    def to_schema(self) -> dict:
        """Convert to JSON Schema format (for LLM tool calling)."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.default is not None:
                properties[param.name]["default"] = param.default
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
    
    def to_prompt_text(self) -> str:
        """Generate human-readable tool description for system prompts."""
        lines = [f"- {self.name}: {self.description.strip()}"]
        for param in self.parameters:
            req = " (required)" if param.required else " (optional)"
            lines.append(f"    • {param.name} ({param.type}){req}: {param.description}")
        return "\n".join(lines)


def load_tool_descriptors(tools_dir: str = None) -> Dict[str, ToolDescriptor]:
    """
    Load all YAML tool descriptors from the tools directory.
    
    Args:
        tools_dir: Path to the tools/ directory. If None, uses project default.
        
    Returns:
        Dictionary mapping tool name → ToolDescriptor.
    """
    if tools_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tools_dir = os.path.join(project_root, "tools")
    
    if not os.path.exists(tools_dir):
        print(f"Warning: Tools directory not found at {tools_dir}")
        return {}
    
    descriptors = {}
    
    for filename in sorted(os.listdir(tools_dir)):
        if not filename.endswith('.yaml') and not filename.endswith('.yml'):
            continue
        
        filepath = os.path.join(tools_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            if not data or 'name' not in data:
                print(f"Warning: Skipping invalid tool descriptor: {filename}")
                continue
            
            # Parse parameters
            params = []
            for param_name, param_data in data.get('parameters', {}).items():
                if isinstance(param_data, dict):
                    params.append(ToolParameter(
                        name=param_name,
                        type=param_data.get('type', 'string'),
                        description=param_data.get('description', ''),
                        required=param_data.get('required', True),
                        default=param_data.get('default', None),
                    ))
            
            # Parse returns
            returns = data.get('returns', {})
            
            descriptor = ToolDescriptor(
                name=data['name'],
                description=data.get('description', ''),
                parameters=params,
                returns_type=returns.get('type', 'string') if isinstance(returns, dict) else 'string',
                returns_description=returns.get('description', '') if isinstance(returns, dict) else '',
                container=data.get('container', False),
                category=data.get('category', 'general'),
            )
            
            descriptors[descriptor.name] = descriptor
            
        except Exception as e:
            print(f"Warning: Failed to parse tool descriptor {filename}: {e}")
    
    if descriptors:
        print(f"Loaded {len(descriptors)} tool descriptor(s) from {tools_dir}")
    
    return descriptors


def generate_tools_prompt(descriptors: Dict[str, ToolDescriptor], 
                          tool_names: List[str] = None) -> str:
    """
    Generate a formatted tools section for LLM system prompts.
    
    Args:
        descriptors: Dictionary of all available tool descriptors.
        tool_names: Optional list to filter which tools to include.
        
    Returns:
        Formatted string describing available tools.
    """
    lines = ["Available Tools:"]
    
    for name, desc in descriptors.items():
        if tool_names and name not in tool_names:
            continue
        lines.append(desc.to_prompt_text())
    
    return "\n".join(lines)
