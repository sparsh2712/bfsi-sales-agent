"""
Tool manager for integrating external tools with the sales agent
"""
import os
import logging
import importlib
import inspect
from typing import Dict, Any, List, Callable

class ToolManager:
    """
    Manages external tools that can be used by the sales agent
    Provides a framework for adding new tools dynamically
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the tool manager
        
        Args:
            config: Configuration for tools
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.tools: Dict[str, Any] = {}
        
        # Load configured tools
        self._load_tools()
    
    def _load_tools(self):
        """Load tools defined in the configuration"""
        tool_configs = self.config.get("tools", {})
        
        for tool_name, tool_config in tool_configs.items():
            if not tool_config.get("enabled", True):
                self.logger.info(f"Tool '{tool_name}' is disabled, skipping")
                continue
            
            try:
                # Get module and class information
                module_path = tool_config.get("module", f"src.tools.{tool_name}")
                class_name = tool_config.get("class_name", f"{tool_name.capitalize()}Tool")
                
                # Dynamically import the tool
                module = importlib.import_module(module_path)
                tool_class = getattr(module, class_name)
                
                # Initialize the tool
                tool_instance = tool_class(tool_config)
                
                # Register the tool
                self.register_tool(tool_name, tool_instance)
                
                self.logger.info(f"Successfully loaded tool: {tool_name}")
            
            except Exception as e:
                self.logger.error(f"Failed to load tool '{tool_name}': {str(e)}")
    
    def register_tool(self, name: str, tool_instance: Any):
        """
        Register a new tool
        
        Args:
            name: Name of the tool
            tool_instance: Instance of the tool class
        """
        if name in self.tools:
            self.logger.warning(f"Overwriting existing tool: {name}")
        
        self.tools[name] = tool_instance
        self.logger.info(f"Registered tool: {name}")
    
    def unregister_tool(self, name: str):
        """
        Unregister a tool
        
        Args:
            name: Name of the tool to unregister
        """
        if name in self.tools:
            del self.tools[name]
            self.logger.info(f"Unregistered tool: {name}")
        else:
            self.logger.warning(f"Attempted to unregister non-existent tool: {name}")
    
    def get_tool(self, name: str):
        """
        Get a tool by name
        
        Args:
            name: Name of the tool to retrieve
            
        Returns:
            Tool instance or None if not found
        """
        if name in self.tools:
            return self.tools[name]
        
        self.logger.warning(f"Tool not found: {name}")
        return None
    
    def execute_tool(self, name: str, method: str = "execute", **kwargs):
        """
        Execute a method on a tool
        
        Args:
            name: Name of the tool
            method: Name of the method to execute
            **kwargs: Arguments to pass to the method
            
        Returns:
            Result of the tool execution or None if failed
        """
        tool = self.get_tool(name)
        
        if not tool:
            return None
        
        try:
            # Get the method
            if not hasattr(tool, method):
                self.logger.error(f"Method '{method}' not found on tool '{name}'")
                return None
            
            tool_method = getattr(tool, method)
            
            # Execute the method
            result = tool_method(**kwargs)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error executing tool '{name}.{method}': {str(e)}")
            return None
    
    def list_available_tools(self):
        """
        List all available tools and their methods
        
        Returns:
            Dictionary of tool information
        """
        tool_info = {}
        
        for name, tool in self.tools.items():
            methods = []
            
            # Get all public methods
            for method_name, method in inspect.getmembers(tool, inspect.ismethod):
                # Skip private and special methods
                if method_name.startswith("_"):
                    continue
                
                # Get method signature and docstring
                signature = str(inspect.signature(method))
                doc = inspect.getdoc(method) or "No documentation available"
                
                methods.append({
                    "name": method_name,
                    "signature": signature,
                    "description": doc
                })
            
            # Get tool description
            description = inspect.getdoc(tool) or "No description available"
            
            tool_info[name] = {
                "description": description,
                "methods": methods
            }
        
        return tool_info


# Example tool implementation
class CalculatorTool:
    """Simple calculator tool for financial calculations"""
    
    def __init__(self, config=None):
        """Initialize the calculator tool"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def calculate_emi(self, principal, rate, tenure_months):
        """
        Calculate EMI for a loan
        
        Args:
            principal: Loan amount
            rate: Annual interest rate (in percentage)
            tenure_months: Loan tenure in months
            
        Returns:
            Monthly EMI amount
        """
        try:
            # Convert annual rate to monthly and decimal
            rate_monthly = (rate / 100) / 12
            
            # Calculate EMI
            emi = principal * rate_monthly * (1 + rate_monthly) ** tenure_months / ((1 + rate_monthly) ** tenure_months - 1)
            
            return round(emi, 2)
        
        except Exception as e:
            self.logger.error(f"Error calculating EMI: {str(e)}")
            return None
    
    def calculate_returns(self, principal, rate, years, compounding_frequency="annually"):
        """
        Calculate returns on an investment
        
        Args:
            principal: Initial investment amount
            rate: Annual interest rate (in percentage)
            years: Investment period in years
            compounding_frequency: Frequency of compounding (annually, quarterly, monthly, daily)
            
        Returns:
            Final amount and total interest earned
        """
        try:
            # Convert rate to decimal
            rate_decimal = rate / 100
            
            # Determine compounding periods per year
            compounding_periods = {
                "annually": 1,
                "semi_annually": 2,
                "quarterly": 4,
                "monthly": 12,
                "daily": 365
            }
            
            periods = compounding_periods.get(compounding_frequency, 1)
            
            # Calculate final amount
            final_amount = principal * (1 + rate_decimal / periods) ** (periods * years)
            interest_earned = final_amount - principal
            
            return {
                "final_amount": round(final_amount, 2),
                "interest_earned": round(interest_earned, 2)
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating returns: {str(e)}")
            return None