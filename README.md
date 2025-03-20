# BFSI Sales Agent

An AI-powered sales agent for the Banking, Financial Services, and Insurance (BFSI) sector. This agent engages potential customers in natural, human-like conversations using a female voice with a South Indian accent.

## Features

- Voice-based interactions via Twilio
- Text-to-Speech with Eleven Labs
- Support for Indian languages
- Knowledge base for handling FAQs and product details
- Extensible architecture for adding new tools and use cases
- Configurable voice parameters and conversation flows

## Technical Architecture

- **Text-to-Speech (TTS)**: Eleven Labs
- **AI Model**: UltraVox with LLM processing
- **Call Setup**: Twilio-powered cloud-based phone number
- **Knowledge Base**: Local document storage with simple retrieval
- **Tool Integration**: Extensible framework for adding tools

## Project Structure

```
bfsi-sales-agent/
├── config/                      # Configuration files
│   ├── agent_config.yaml        # Main agent configuration
│   ├── voice_config.yaml        # Voice-related settings
│   └── use_cases/               # Use case specific configs
│       ├── mutual_funds.yaml    # Config for mutual fund sales
│       └── insurance.yaml       # Config for insurance sales
├── src/
│   ├── agent/                   # Core agent functionality
│   ├── voice/                   # Voice processing components
│   ├── knowledge/               # Knowledge base handling
│   ├── tools/                   # Extensible tools
│   └── utils/                   # Utility functions
├── data/                        # Knowledge base documents
├── logs/                        # Log files
└── main.py                      # Entry point
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- Eleven Labs API key
- Twilio account with phone number

### Installation with uv

1. Install [uv](https://github.com/astral-sh/uv) if you haven't already:
   ```bash
   pip install uv
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bfsi-sales-agent.git
   cd bfsi-sales-agent
   ```

3. Create and activate a virtual environment with uv:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```

4. Install dependencies with uv:
   ```bash
   uv pip install -r requirements.txt
   ```

5. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file to include your API keys and configuration.

### Running the Agent

1. Start the server:
   ```bash
   python main.py --config config/agent_config.yaml --use-case mutual_funds
   ```

2. For development, you may want to use ngrok to expose your local server:
   ```bash
   ngrok http 5000
   ```
   Then update your Twilio webhook URL to the ngrok URL.

## Extending the Agent

### Adding a New Use Case

1. Create a new use case configuration in `config/use_cases/your_use_case.yaml`
2. Add relevant knowledge base documents in `data/documents/`
3. Run the agent with your new use case:
   ```bash
   python main.py --use-case your_use_case
   ```

### Adding a New Tool

1. Create a new tool class in `src/tools/your_tool.py`
2. Add tool configuration in `config/agent_config.yaml`
3. Register the tool in the agent during initialization

## Environment Variables

- `ELEVENLABS_API_KEY`: Your Eleven Labs API key
- `TWILIO_ACCOUNT_SID`: Your Twilio account SID
- `TWILIO_AUTH_TOKEN`: Your Twilio auth token
- `TWILIO_PHONE_NUMBER`: Your Twilio phone number

## License

This project is licensed under the MIT License - see the LICENSE file for details.