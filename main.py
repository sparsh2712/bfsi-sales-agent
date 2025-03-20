#!/usr/bin/env python
"""
BFSI Sales Agent - Main Entry Point
"""
import os
import argparse
import logging
from dotenv import load_dotenv

from src.utils.config import ConfigLoader
from src.agent.sales_agent import SalesAgent
from src.voice.call_handler import CallHandler
from src.knowledge.knowledge_base import KnowledgeBase


def setup_logging():
    """Configure logging for the application"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/bfsi_agent.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    """Main entry point for the BFSI Sales Agent"""
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting BFSI Sales Agent")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="BFSI Sales Agent")
    parser.add_argument("--config", default="config/agent_config.yaml", help="Path to main config file")
    parser.add_argument("--use-case", default="mutual_funds", help="Use case to load")
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    main_config = config_loader.load_config()
    use_case_config = config_loader.load_use_case_config(args.use_case)
    
    # Initialize knowledge base
    knowledge_base = KnowledgeBase(main_config.get("knowledge_base", {}))
    
    # Initialize the sales agent
    sales_agent = SalesAgent(
        main_config.get("agent", {}),
        use_case_config,
        knowledge_base
    )
    
    # Initialize call handler
    call_handler = CallHandler(
        main_config.get("voice", {}),
        sales_agent
    )
    
    # Start the call handler
    call_handler.start()


if __name__ == "__main__":
    main()