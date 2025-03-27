"""
LLM-integrated sales agent for the BFSI sector
"""
import logging
import json
import re
import time
import os
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
from typing import Dict, List, Any, Optional, Tuple

class SalesAgent:
    """
    LLM-powered sales agent for BFSI use cases.
    Uses LLM for natural language understanding and response generation
    while maintaining conversation context and business constraints.
    """
    
    def __init__(self, agent_config, use_case_config, knowledge_base):
        """
        Initialize the sales agent
        
        Args:
            agent_config (dict): Agent configuration
            use_case_config (dict): Use case specific configuration
            knowledge_base (KnowledgeBase): Knowledge base instance
        """
        self.config = agent_config
        self.use_case_config = use_case_config
        self.knowledge_base = knowledge_base
        self.logger = logging.getLogger(__name__)
        
        # Track conversation state for each session
        self.sessions = {}
        
        # Load responses from use case config (used for guidance, not directly)
        self.response_templates = use_case_config.get("responses", {})
        
        # Configure LLM
        self.llm_config = agent_config.get("llm", {})
        self.default_api_key = os.environ.get("OPENAI_API_KEY")
        self.llm_provider = self.llm_config.get("provider", "openai")
        self.model_name = self.llm_config.get("model_name", "gpt-4o")
        self.temperature = self.llm_config.get("temperature", 0.7)
        self.max_tokens = self.llm_config.get("max_tokens", 500)
        
        # Intent patterns are used as fallback for intent detection
        self._init_intent_patterns()
        
        # System message components
        self.system_prompt_base = self._build_system_prompt()
        
        # Custom use case variables for template filling
        self.custom_vars = use_case_config.get("custom_variables", {})
        
        # Validate configuration
        self._validate_config()
        
        self.logger.info(f"LLM-powered sales agent initialized for use case: {use_case_config.get('name', 'unnamed')}")
    
    def _validate_config(self):
        """Validate the agent configuration and log warnings for missing elements"""
        # Check for required configuration
        if not self.llm_config and not self.default_api_key:
            self.logger.warning("No LLM API key configured. Agent will have limited functionality.")
        
        if not self.use_case_config.get("name"):
            self.logger.warning("Use case name not specified in configuration.")
        
        # Validate response templates
        essential_templates = ["initial_greeting", "return_greeting", "goodbye", "fallback"]
        for template in essential_templates:
            if template not in self.response_templates or not self.response_templates[template]:
                self.logger.warning(f"Missing essential response template: {template}")
    
    def _init_intent_patterns(self):
        """Initialize the intent recognition patterns (used as fallback)"""
        self.intent_patterns = {
            "greeting": [r"hello", r"hi", r"hey", r"good morning", r"good afternoon", r"namaste"],
            "goodbye": [r"bye", r"goodbye", r"see you", r"thank you", r"thanks"],
            "product_info": [r"product", r"tell me about", r"what is", r"information", r"details"],
            "pricing": [r"price", r"cost", r"fee", r"how much", r"rate", r"interest"],
            "eligibility": [r"eligible", r"qualify", r"can i", r"requirements", r"criteria"],
            "process": [r"process", r"how to", r"steps", r"procedure", r"what do i need to do"],
            "speak_to_human": [r"human", r"agent", r"person", r"representative", r"speak to someone"],
            "fund_performance": [r"performance", r"return", r"yield", r"how good", r"net gain"],
            "risk": [r"risk", r"safe", r"secure", r"guarantee", r"lose money"],
            "investment_duration": [r"duration", r"how long", r"term", r"maturity", r"period"],
            "contact": [r"contact", r"phone", r"email", r"office", r"branch", r"location"]
        }
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for the LLM
        
        Returns:
            str: System prompt
        """
        # Load base prompt for the use case
        if self.use_case_config.get("name") == "Mutual Funds Sales Agent":
            return """
You are an AI voice assistant for a mutual funds company. You help potential customers understand mutual fund products, 
answer questions about fund performance, explain investment options, and guide them through the investment process.

IMPORTANT GUIDELINES:
1. Keep your responses concise and clear - you're a voice assistant, so people are hearing, not reading your responses
2. Use a professional but warm, conversational tone appropriate for financial services
3. Speak with the knowledge of an investment advisor
4. When explaining complex financial concepts, use simple language and analogies
5. If asked about specific returns or guarantees, emphasize that past performance is not indicative of future results
6. If you don't know an answer, don't make up information - offer to connect the person with a human advisor
7. Always make sure your information is accurate and up-to-date
8. Never ask for sensitive personal or financial information
9. When the customer wishes to invest or needs specific advice beyond your capabilities, offer to transfer them to a human advisor

When referring to the company, use "mOSAIC Investment" and mention our multi-yield series mutual funds when relevant.
Our leadership team includes Maneesh Dangi (CEO) and R. Gopi Krishna (CIO).
"""
        else:
            # Generic BFSI assistant prompt
            return """
You are an AI voice assistant for a financial services company. You help customers understand financial products, 
answer questions about services, and guide them through various processes.

IMPORTANT GUIDELINES:
1. Keep your responses concise and clear - you're a voice assistant, so people are hearing, not reading your responses
2. Use a professional but warm, conversational tone appropriate for financial services
3. Speak with the knowledge of a financial advisor
4. When explaining complex financial concepts, use simple language and analogies
5. If you don't know an answer, don't make up information - offer to connect the person with a human advisor
6. Always make sure your information is accurate and up-to-date
7. Never ask for sensitive personal or financial information
8. When the customer needs specific advice beyond your capabilities, offer to transfer them to a human advisor
"""
    
    def process_input(self, session_id: str, user_input: str) -> str:
        """
        Process user input and generate a response using the LLM
        
        Args:
            session_id (str): Unique session identifier
            user_input (str): User's input text
            
        Returns:
            str: Agent's response
        """
        # Ensure session exists
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "context": {},
                "created_at": datetime.now().isoformat()
            }
        
        session = self.sessions[session_id]
        
        # Log the input
        self.logger.info(f"Session {session_id}: Received input: '{user_input}'")
        
        # Identify intent (both with rule-based and later with LLM)
        rule_based_intent = self._identify_intent_rule_based(user_input)
        
        # Special handling for simple intents to avoid LLM call
        if rule_based_intent == "greeting" and self._is_first_interaction(session):
            response = self._get_random_response("initial_greeting")
            self._record_interaction(session, user_input, response, "greeting")
            return response
        elif rule_based_intent == "greeting" and not self._is_first_interaction(session):
            response = self._get_random_response("return_greeting")
            self._record_interaction(session, user_input, response, "greeting")
            return response
        elif rule_based_intent == "goodbye":
            response = self._get_random_response("goodbye")
            self._record_interaction(session, user_input, response, "goodbye")
            return response
        elif rule_based_intent == "speak_to_human":
            response = self._get_random_response("speak_to_human")
            self._record_interaction(session, user_input, response, "speak_to_human")
            return response
        
        # For more complex intents, use the LLM
        response, llm_intent = self._generate_llm_response(session, user_input, rule_based_intent)
        
        # Record in history
        self._record_interaction(session, user_input, response, llm_intent or rule_based_intent)
        
        return response
    
    def _record_interaction(self, session: Dict, user_input: str, response: str, intent: str):
        """
        Record an interaction in the session history
        
        Args:
            session (dict): Session information
            user_input (str): User's input
            response (str): Agent's response
            intent (str): Identified intent
        """
        session["history"].append({
            "user": user_input,
            "agent": response,
            "intent": intent,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update context based on interaction
        self._update_context(session, intent, user_input, response)
    
    def _is_first_interaction(self, session: Dict) -> bool:
        """Check if this is the first interaction in the session"""
        return len(session["history"]) == 0
    
    def _identify_intent_rule_based(self, text: str) -> str:
        """
        Identify the user's intent from their input using rule-based patterns
        
        Args:
            text (str): User's input text
            
        Returns:
            str: Identified intent
        """
        text = text.lower()
        
        # Check against all patterns
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(r'\b' + pattern + r'\b', text):
                    self.logger.debug(f"Matched intent: {intent} with pattern: {pattern}")
                    return intent
        
        # Default to general inquiry if no specific intent is detected
        return "general_inquiry"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _generate_llm_response(self, session: Dict, user_input: str, rule_based_intent: str) -> Tuple[str, Optional[str]]:
        """
        Generate a response using the LLM
        
        Args:
            session (dict): Session information
            user_input (str): User's input text
            rule_based_intent (str): Intent identified by rule-based method
            
        Returns:
            tuple: (generated_response, llm_detected_intent)
        """
        # Get relevant context from knowledge base
        relevant_context = self.knowledge_base.get_relevant_context(user_input, rule_based_intent)
        
        # Build conversation history for context
        conversation_history = self._format_conversation_history(session)
        
        # Construct the full system prompt
        system_prompt = self._build_full_system_prompt(relevant_context)
        
        # Extract product information from use case config
        product_info = self._format_product_info()
        
        # Add response guidance based on the detected intent
        response_guidance = self._get_response_guidance(rule_based_intent)
        
        # Construct the user message with all context
        user_message = self._build_user_message(
            user_input=user_input,
            rule_based_intent=rule_based_intent,
            product_info=product_info,
            response_guidance=response_guidance
        )
        
        # Prepare messages for the LLM
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current user query
        messages.append({"role": "user", "content": user_message})
        
        # Call the LLM
        try:
            if self.llm_provider == "openai":
                response_content, detected_intent = self._call_openai(messages)
            else:
                # Fallback to template response if LLM call fails or is not configured
                self.logger.warning(f"Unsupported LLM provider: {self.llm_provider}")
                return self._get_fallback_response(rule_based_intent), rule_based_intent
            
            # Process and return the response
            return response_content, detected_intent
        
        except Exception as e:
            self.logger.error(f"Error generating LLM response: {str(e)}")
            # Fallback to template
            return self._get_fallback_response(rule_based_intent), rule_based_intent
    
    def _build_full_system_prompt(self, relevant_context: str) -> str:
        """
        Build the full system prompt including relevant context
        
        Args:
            relevant_context (str): Relevant context from knowledge base
            
        Returns:
            str: Complete system prompt
        """
        system_prompt = self.system_prompt_base
        
        # Add use case specific information
        system_prompt += f"\n\nUse case: {self.use_case_config.get('name', 'BFSI Sales Agent')}"
        system_prompt += f"\nDescription: {self.use_case_config.get('description', '')}"
        
        # Add voice personality guidance
        system_prompt += """
\nVOICE PERSONALITY:
- You are a female voice assistant with a South Indian accent
- Speak in a clear, articulate manner with natural pauses
- Your voice conveys trustworthiness, intelligence, and warmth
- You may occasionally use simple Indian English phrases if appropriate
- Keep responses concise (1-3 sentences when possible) as you're speaking, not writing
"""
        
        # Add relevant context if available
        if relevant_context:
            system_prompt += "\n\nRELEVANT INFORMATION:\n" + relevant_context
        
        # Add response structure guidance
        system_prompt += """
\nRESPONSE STRUCTURE:
- Begin by addressing the customer's question or concern directly
- Provide clear, accurate information in simple language
- Avoid jargon unless necessary, and explain any technical terms
- End with a helpful follow-up question or offer of additional assistance when appropriate
- Never fabricate information - if you don't know, offer to connect to a human agent
- Format your response to be spoken aloud - avoid lists, bullets, or complex structures
- Keep your response concise and focused, as this will be spoken to the user
"""
        
        return system_prompt
    
    def _build_user_message(self, user_input: str, rule_based_intent: str, 
                           product_info: str, response_guidance: str) -> str:
        """
        Build the complete user message with all context and guidance
        
        Args:
            user_input (str): User's input text
            rule_based_intent (str): Rule-based intent detection
            product_info (str): Product information
            response_guidance (str): Guidance for response
            
        Returns:
            str: Formatted user message
        """
        message = f"Customer message: {user_input}\n\n"
        
        message += f"Likely intent: {rule_based_intent}\n\n"
        
        if product_info:
            message += f"Product information:\n{product_info}\n\n"
        
        if response_guidance:
            message += f"Response guidance:\n{response_guidance}\n\n"
        
        message += "Please provide a natural, conversational response to the customer's message. " + \
                  "Keep your response concise and focused, as it will be spoken to the user. " + \
                  "Also, determine what you think the customer's intent is and include it at the end of your response " + \
                  "in the format [INTENT: intent_name]. This intent label will be removed before the response is sent to the user."
        
        return message
    
    def _format_conversation_history(self, session: Dict) -> List[Dict]:
        """
        Format the conversation history for the LLM
        
        Args:
            session (dict): Session information
            
        Returns:
            list: Formatted conversation history
        """
        # Limit to most recent interactions to avoid context length issues
        max_history = min(self.config.get("max_history_length", 10), 10)
        recent_history = session["history"][-max_history:] if session["history"] else []
        
        formatted_history = []
        for interaction in recent_history:
            formatted_history.append({
                "role": "user",
                "content": interaction["user"]
            })
            formatted_history.append({
                "role": "assistant",
                "content": interaction["agent"]
            })
        
        return formatted_history
    
    def _format_product_info(self) -> str:
        """
        Format product information from the use case config
        
        Returns:
            str: Formatted product information
        """
        products = self.use_case_config.get("products", {})
        if not products:
            return ""
        
        product_info = "Available products:\n"
        for product_id, product in products.items():
            product_info += f"- {product.get('name', product_id)}:\n"
            for key, value in product.items():
                if key != "name":
                    product_info += f"  - {key}: {value}\n"
        
        return product_info
    
    def _get_response_guidance(self, intent: str) -> str:
        """
        Get guidance for response based on intent
        
        Args:
            intent (str): Identified intent
            
        Returns:
            str: Response guidance
        """
        # Check if we have templates for this intent
        if intent in self.response_templates:
            templates = self.response_templates[intent]
            if templates:
                return f"For {intent} queries, consider responses like:\n" + "\n".join([f"- {t}" for t in templates[:3]])
        
        return ""
    
    def _call_openai(self, messages: List[Dict]) -> Tuple[str, Optional[str]]:
        """
        Call the OpenAI API
        
        Args:
            messages (list): List of message dictionaries
            
        Returns:
            tuple: (response_content, detected_intent)
        """
        api_key = self.llm_config.get("api_key") or self.default_api_key
        if not api_key:
            raise ValueError("OpenAI API key not configured")
        
        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            response_content = result["choices"][0]["message"]["content"]
            
            # Extract intent if available
            intent_match = re.search(r'\[INTENT:\s*(\w+)\]', response_content)
            detected_intent = intent_match.group(1) if intent_match else None
            
            # Remove the intent tag from the response
            if intent_match:
                response_content = re.sub(r'\s*\[INTENT:\s*\w+\]\s*$', '', response_content)
            
            return response_content, detected_intent
        
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def _get_random_response(self, response_type: str) -> str:
        """
        Get a random response of the specified type from templates
        
        Args:
            response_type (str): Type of response to get
            
        Returns:
            str: Selected response
        """
        import random
        
        responses = self.response_templates.get(response_type, [])
        
        # Fallback to default responses if none configured for this type
        if not responses:
            default_responses = {
                "initial_greeting": [
                    "Hello! I'm your financial advisor assistant. How may I help you today?",
                    "Namaste! I'm here to assist you with information about our financial products. What would you like to know?"
                ],
                "return_greeting": [
                    "Welcome back! What can I help you with today?",
                    "Hello again! How can I assist you further?"
                ],
                "goodbye": [
                    "Thank you for your time. Have a great day!",
                    "It was nice speaking with you. If you have more questions, feel free to reach out."
                ],
                "speak_to_human": [
                    "I understand you'd like to speak with a human agent. Let me transfer you to one of our representatives.",
                    "I'll connect you with a human representative who can assist you further."
                ],
                "fallback": [
                    "I apologize, but I don't have specific information about that. Could you please ask something else about our products or services?",
                    "I'm sorry, I couldn't find relevant information. Would you like to know about our financial products instead?"
                ]
            }
            responses = default_responses.get(response_type, ["I'm sorry, I don't understand. Could you rephrase that?"])
        
        # Replace any custom variables
        selected_response = random.choice(responses)
        for var_name, var_value in self.custom_vars.items():
            selected_response = selected_response.replace(f"{{{var_name}}}", str(var_value))
        
        return selected_response
    
    def _get_fallback_response(self, intent: str) -> str:
        """
        Get an appropriate fallback response based on intent
        
        Args:
            intent (str): Identified intent
            
        Returns:
            str: Fallback response
        """
        # If we have template responses for this intent, use one
        if intent in self.response_templates and self.response_templates[intent]:
            return self._get_random_response(intent)
        
        # Otherwise use a generic fallback
        return self._get_random_response("fallback")
    
    def _update_context(self, session: Dict, intent: str, user_input: str, response: str):
        """
        Update the session context based on the current interaction
        
        Args:
            session (dict): Session information
            intent (str): Identified intent
            user_input (str): User's input text
            response (str): Agent's response
        """
        context = session.get("context", {})
        
        # Update last interaction time
        context["last_interaction"] = datetime.now().isoformat()
        
        # Track topics discussed
        if "topics_discussed" not in context:
            context["topics_discussed"] = []
        
        if intent not in context["topics_discussed"]:
            context["topics_discussed"].append(intent)
        
        # Extract and store entities if applicable
        entities = self._extract_entities(user_input, intent)
        if entities:
            if "entities" not in context:
                context["entities"] = {}
            context["entities"].update(entities)
        
        session["context"] = context
    
    def _extract_entities(self, text: str, intent: str) -> Dict:
        """
        Extract relevant entities from user input based on intent
        
        Args:
            text (str): User's input text
            intent (str): Identified intent
            
        Returns:
            dict: Extracted entities
        """
        entities = {}
        
        # Extract based on intent
        if intent == "product_info":
            # Try to identify specific products mentioned
            products = self.use_case_config.get("products", {})
            for product_id, product in products.items():
                product_name = product.get("name", "").lower()
                if product_name in text.lower():
                    entities["product"] = product_id
                    break
        
        elif intent == "pricing":
            # Look for amounts
            amount_match = re.search(r'(\d+)(?:\s*(?:rs|rupees|lakhs?|k|crores?|cr))?', text.lower())
            if amount_match:
                entities["amount"] = amount_match.group(1)
        
        return entities

    def get_session_summary(self, session_id: str) -> Dict:
        """
        Get a summary of the conversation session
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            dict: Session summary
        """
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "interaction_count": len(session["history"]),
            "topics_discussed": session.get("context", {}).get("topics_discussed", []),
            "last_interaction": session.get("context", {}).get("last_interaction", "")
        }