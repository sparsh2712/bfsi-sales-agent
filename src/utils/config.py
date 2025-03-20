"""
Configuration utilities for the BFSI Sales Agent
"""
import os
import yaml
import logging

class ConfigLoader:
    """Handles loading and parsing of configuration files"""
    
    def __init__(self, config_path):
        """
        Initialize the config loader
        
        Args:
            config_path (str): Path to the main configuration file
        """
        self.config_path = config_path
        self.config_dir = os.path.dirname(config_path)
        self.logger = logging.getLogger(__name__)
    
    def load_config(self):
        """
        Load the main configuration file
        
        Returns:
            dict: Configuration as a dictionary
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                self.logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def load_use_case_config(self, use_case):
        """
        Load a specific use case configuration
        
        Args:
            use_case (str): Name of the use case
            
        Returns:
            dict: Use case configuration as a dictionary
        """
        use_case_path = os.path.join(self.config_dir, "use_cases", f"{use_case}.yaml")
        
        try:
            with open(use_case_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                self.logger.info(f"Loaded use case configuration from {use_case_path}")
                return config
        except Exception as e:
            self.logger.error(f"Failed to load use case configuration: {str(e)}")
            raise