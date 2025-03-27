#!/usr/bin/env python
"""
Test script for BFSI Sales Agent

This script provides a simple command-line interface for testing the BFSI Sales Agent
without requiring Twilio or ElevenLabs integration.

Usage:
    python test_agent.py [--config CONFIG_FILE] [--use-case USE_CASE] [--script SCRIPT_FILE] [--debug]

Examples:
    # Interactive mode
    python test_agent.py --use-case mutual_funds
    
    # Script mode
    python test_agent.py --script tests/comprehensive_test.txt
    
    # Debug mode
    python test_agent.py --debug
"""
import os
import sys
import argparse
import logging
import uuid
import time
import json
from pathlib import Path
from dotenv import load_dotenv


def setup_logging(debug=False):
    """Configure logging for the test script"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/test_agent.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def check_dependencies(config_path, use_case):
    """
    Check if required files and directories exist
    
    Args:
        config_path (str): Path to main config file
        use_case (str): Use case to load
        
    Returns:
        bool: True if all dependencies are met, False otherwise
    """
    missing_requirements = []
    
    # Check main config file
    if not Path(config_path).exists():
        print(f"Error: Config file not found at {config_path}")
        return False
    
    # Check use case config
    use_case_path = Path(os.path.dirname(config_path)) / "use_cases" / f"{use_case}.yaml"
    if not use_case_path.exists():
        print(f"Error: Use case config not found at {use_case_path}")
        return False
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("The LLM integration will not work without this key.")
        missing_requirements.append("OPENAI_API_KEY")
    
    # Check for required Python packages
    try:
        import sentence_transformers
        print("✓ sentence-transformers package found")
    except ImportError:
        print("Error: sentence-transformers package not found. Please install it with:")
        print("  pip install sentence-transformers")
        missing_requirements.append("sentence-transformers")
    
    try:
        import sklearn
        print("✓ scikit-learn package found")
    except ImportError:
        print("Error: scikit-learn package not found. Please install it with:")
        print("  pip install scikit-learn")
        missing_requirements.append("scikit-learn")
    
    try:
        import numpy
        print("✓ numpy package found")
    except ImportError:
        print("Error: numpy package not found. Please install it with:")
        print("  pip install numpy")
        missing_requirements.append("numpy")
    
    try:
        import openai
        print("✓ openai package found")
    except ImportError:
        print("Error: openai package not found. Please install it with:")
        print("  pip install openai")
        missing_requirements.append("openai")
    
    # Check knowledge base directory
    try:
        from src.utils.config import ConfigLoader
        config_loader = ConfigLoader(config_path)
        main_config = config_loader.load_config()
        docs_dir = main_config.get("knowledge_base", {}).get("docs_dir", "data/documents")
        
        if not Path(docs_dir).exists():
            print(f"Warning: Knowledge base directory not found at {docs_dir}")
            print("The agent may not have access to document knowledge.")
        else:
            # Check if any documents exist in the directory
            doc_files = list(Path(docs_dir).glob("*.*"))
            if not doc_files:
                print(f"Warning: No document files found in {docs_dir}")
            else:
                print(f"✓ Found {len(doc_files)} document files in knowledge base")
    except Exception as e:
        print(f"Warning: Could not check knowledge base directory: {e}")
    
    # If missing requirements, provide instructions
    if missing_requirements:
        print("\nTo install all missing requirements:")
        print("  pip install " + " ".join(missing_requirements))
        return False if "sentence-transformers" in missing_requirements or "openai" in missing_requirements else True
    
    return True


def run_interactive_mode(sales_agent, session_id, use_case_name, debug=False):
    """
    Run the agent in interactive mode
    
    Args:
        sales_agent: Sales agent instance
        session_id (str): Session ID
        use_case_name (str): Name of the use case
        debug (bool): Whether to show debug information
    """
    print("\n===== BFSI Sales Agent Test Interface =====")
    print(f"Use case: {use_case_name}")
    print('Type "exit", "quit", or "bye" to end the session')
    print('Type "debug" to toggle debug mode')
    print("==========================================\n")
    
    # Initial greeting
    initial_response = sales_agent.process_input(session_id, "hello")
    print(f"Agent: {initial_response}\n")
    
    # Main interaction loop
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nEnding session. Goodbye!")
            break
        
        if user_input.lower() == "debug":
            debug = not debug
            print(f"\nDebug mode {'enabled' if debug else 'disabled'}\n")
            continue
        
        # Measure response time
        start_time = time.time()
        
        try:
            response = sales_agent.process_input(session_id, user_input)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            print(f"\nAgent: {response}\n")
            
            # Show debug info if enabled
            if debug:
                print(f"[DEBUG] Response time: {response_time:.2f} seconds")
                try:
                    # Try to get and display session context
                    if hasattr(sales_agent, 'sessions') and session_id in sales_agent.sessions:
                        context = sales_agent.sessions[session_id].get('context', {})
                        print(f"[DEBUG] Session context: {json.dumps(context, indent=2)}")
                        
                        # Show detected intent if available
                        if sales_agent.sessions[session_id]['history']:
                            last_intent = sales_agent.sessions[session_id]['history'][-1].get('intent', 'unknown')
                            print(f"[DEBUG] Detected intent: {last_intent}")
                except Exception as e:
                    print(f"[DEBUG] Error accessing session context: {e}")
        
        except Exception as e:
            print(f"\nError processing input: {e}\n")


def run_script_mode(script_file, sales_agent, session_id, debug=False):
    """
    Run the agent in script mode, using inputs from a file
    
    Args:
        script_file (str): Path to script file
        sales_agent: Sales agent instance
        session_id (str): Session ID
        debug (bool): Whether to show debug information
    """
    try:
        with open(script_file, 'r') as f:
            script_lines = f.readlines()
        
        print("\n===== Running in Script Mode =====")
        print(f"Using script file: {script_file}")
        print("==================================\n")
        
        # Initial greeting
        start_time = time.time()
        initial_response = sales_agent.process_input(session_id, "hello")
        response_time = time.time() - start_time
        
        print(f"Agent: {initial_response}\n")
        if debug:
            print(f"[DEBUG] Response time: {response_time:.2f} seconds")
        
        total_response_time = 0
        successful_queries = 0
        
        for i, line in enumerate(script_lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip empty lines and comments
            
            print(f"You: {line}")
            
            # Measure response time
            start_time = time.time()
            
            try:
                response = sales_agent.process_input(session_id, line)
                response_time = time.time() - start_time
                total_response_time += response_time
                successful_queries += 1
                
                print(f"\nAgent: {response}\n")
                
                # Show debug info if enabled
                if debug:
                    print(f"[DEBUG] Response time: {response_time:.2f} seconds")
                    try:
                        # Show detected intent if available
                        if hasattr(sales_agent, 'sessions') and session_id in sales_agent.sessions:
                            if sales_agent.sessions[session_id]['history']:
                                last_intent = sales_agent.sessions[session_id]['history'][-1].get('intent', 'unknown')
                                print(f"[DEBUG] Detected intent: {last_intent}")
                    except Exception as e:
                        print(f"[DEBUG] Error accessing session data: {e}")
            
            except Exception as e:
                print(f"\nError processing input: {e}\n")
            
            # Small delay to make it more readable
            time.sleep(0.5)
        
        # Print performance summary
        if successful_queries > 0:
            avg_response_time = total_response_time / successful_queries
            print(f"\nPerformance Summary:")
            print(f"Average response time: {avg_response_time:.2f} seconds")
            print(f"Total queries: {successful_queries}")
    
    except Exception as e:
        print(f"Error in script mode: {e}")


def main():
    """Main entry point for the test script"""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="BFSI Sales Agent Test Script")
    parser.add_argument("--config", default="config/agent_config.yaml", help="Path to main config file")
    parser.add_argument("--use-case", default="mutual_funds", help="Use case to load")
    parser.add_argument("--script", help="Path to script file with predefined inputs")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    # Setup logging with debug level if requested
    logger = setup_logging(args.debug)
    logger.info("Starting BFSI Sales Agent Test Script")
    
    # Check dependencies
    if not check_dependencies(args.config, args.use_case):
        print("\nWarning: Some dependencies are missing. The agent may not work correctly.")
        proceed = input("Do you want to proceed anyway? (y/n): ").lower()
        if proceed != 'y':
            print("Exiting.")
            return
    
    try:
        # Import modules
        from src.utils.config import ConfigLoader
        from src.agent.sales_agent import SalesAgent
        from src.knowledge.knowledge_base import KnowledgeBase
        
        # Load configuration
        config_loader = ConfigLoader(args.config)
        main_config = config_loader.load_config()
        use_case_config = config_loader.load_use_case_config(args.use_case)
        
        # Initialize knowledge base with timing
        print("Initializing knowledge base...")
        kb_start_time = time.time()
        knowledge_base = KnowledgeBase(main_config.get("knowledge_base", {}))
        kb_time = time.time() - kb_start_time
        print(f"Knowledge base initialized in {kb_time:.2f} seconds")
        
        # Initialize the sales agent
        print("Initializing sales agent...")
        agent_start_time = time.time()
        sales_agent = SalesAgent(
            main_config.get("agent", {}),
            use_case_config,
            knowledge_base
        )
        agent_time = time.time() - agent_start_time
        print(f"Sales agent initialized in {agent_time:.2f} seconds")
        
        # Generate a random session ID for testing
        session_id = str(uuid.uuid4())
        logger.info(f"Test session ID: {session_id}")
        
        # Run in script mode or interactive mode
        if args.script:
            run_script_mode(args.script, sales_agent, session_id, args.debug)
        else:
            run_interactive_mode(sales_agent, session_id, use_case_config.get('name', args.use_case), args.debug)
        
        # Print session summary at the end
        summary = sales_agent.get_session_summary(session_id)
        print("\n===== Session Summary =====")
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("============================")
    
    except Exception as e:
        logger.error(f"Error running test script: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("Check the log file for more details.")


if __name__ == "__main__":
    main()