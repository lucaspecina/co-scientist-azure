"""
ReAct agent implementation for DeepResearch.
Uses a reasoning-action-observation cycle to solve tasks.
"""

import re
import json
import logging
from openai import AzureOpenAI
import deepresearch_azure.config as config
from deepresearch_azure.search_tools import get_all_tools
from deepresearch_azure.prompts import REACT_PROMPT
from deepresearch_azure.session_manager import SessionManager

class ReActAgent:
    """
    ReAct agent that uses a reasoning-action-observation cycle.
    """
    
    def __init__(self, verbose=False, session_id=None, skip_session_creation=False, gui_mode=False):
        """Initialize the ReAct agent
        
        Args:
            verbose (bool): Whether to enable verbose logging
            session_id (str): Optional session ID to load
            skip_session_creation (bool): If True, won't create a new session automatically
        """
        self.tools = {tool.name: tool for tool in get_all_tools()}
        
        # Setup logging
        self.verbose = verbose
        self.logger = logging.getLogger('deepresearch.agent')
        if verbose:
            self.logger.setLevel(logging.INFO)
        
        # Initialize OpenAI client
        self.client = AzureOpenAI(
            api_key=config.AZURE_API_KEY,
            api_version=config.AZURE_API_VERSION,
            azure_endpoint=config.AZURE_ENDPOINT
        )
        
        self.model = config.AGENT_MODEL_DEPLOYMENT
        
        # Format the tools for the prompt
        self.tools_description = self._format_tools_for_prompt()
        
        # Track which tools have been used
        self.used_tools = set()
        
        # Initialize conversation context
        self.context = []
        self.original_query = None
        
        # Set min iterations before final answer (to encourage tool use)
        self.min_iterations = 2

        # Initialize session manager
        self.session_manager = SessionManager()
        self.skip_session_creation = skip_session_creation
        if session_id:
            self._load_session(session_id)
        
        if not skip_session_creation:
            self.logger.info(f"ReAct agent initialized with model: {self.model}")
            print(f"ReAct agent initialized with model: {self.model}")
            self.logger.info(f"Available tools: {', '.join(self.tools.keys())}")
            print(f"Available tools: {', '.join(self.tools.keys())}")

        self.gui_mode = gui_mode
        self.pending_user_query = None
        self.current_iteration = 0

    def _load_session(self, session_id: str):
        """Load an existing session"""
        session = self.session_manager.load_session(session_id)
        # Restore the last query's context if it exists
        if session["queries"]:
            last_query = session["queries"][-1]
            self.context = last_query["context"]
            self.used_tools = set(last_query["used_tools"])
            self.original_query = last_query["query"]

    def _format_tools_for_prompt(self):
        """Format the available tools for the prompt"""
        tools_list = []
        for name, tool in self.tools.items():
            tool_str = f"- {name}: {tool.description}\n"
            tool_str += f"  Takes inputs: {{'query': 'The search query to execute'}}\n"
            tool_str += f"  Returns an output of type: string"
            tools_list.append(tool_str)
            
        # Add final_answer tool
        tools_list.append(
            "- final_answer: Provide the final answer to the query\n"
            "  Takes inputs: {'answer': 'The final answer to the query'}\n"
            "  Returns an output of type: string"
        )
        
        return "\n".join(tools_list)
    
    def _repair_json(self, json_str):
        """Simple JSON repair to balance braces and brackets"""
        open_braces = 0
        open_brackets = 0
        in_string = False
        repaired = ''
        for char in json_str:
            repaired += char
            if char == '"' and (len(repaired) == 1 or repaired[-2] != '\\'):
                in_string = not in_string
            if not in_string:
                if char == '{':
                    open_braces += 1
                elif char == '}':
                    open_braces -= 1
                elif char == '[':
                    open_brackets += 1
                elif char == ']':
                    open_brackets -= 1
        # Add closing braces/brackets if needed
        repaired += ']' * open_brackets
        repaired += '}' * open_braces
        return repaired

    def _parse_action(self, response):
        """Parse the action from the model response"""
        import re
        import json
        
        # Find the Action block, allowing for optional code fences and whitespace
        action_match = re.search(r'Action:\s*(?:```(?:json)?\s*)?\{([\s\S]*?)\}(?:\s*```)?', response, re.DOTALL | re.IGNORECASE)
        if not action_match:
            self.logger.warning("No action found in response")
            return None
        
        # Extract the content inside the braces
        action_str = '{' + action_match.group(1) + '}'
        
        self.logger.info(f"Raw action string: {action_str}")
        
        # Clean up whitespace
        action_str = re.sub(r'\s+', ' ', action_str).strip()
        
        # Remove trailing comma
        action_str = re.sub(r',\s*(?=}|])', '', action_str)
        
        # Escape any unescaped double quotes inside values (assume values are between ": and ")
        action_str = re.sub(r'(?<=:")([^"]*?)(?=")', lambda m: m.group(1).replace('"', '\\"'), action_str)
        
        # Now safely replace structural single quotes with double quotes (keys and simple values)
        # This regex targets 'key': 'value' patterns, replacing the quotes
        action_str = re.sub(r"'([a-zA-Z0-9_]+)':", r'"\1":', action_str)  # Keys
        action_str = re.sub(r": '([^']*?)'", r': "\1"', action_str)  # Values (simple, no inner quotes)
        
        self.logger.info(f"Cleaned action string: {action_str}")
        
        action_str = self._repair_json(action_str)
        self.logger.info(f"Repaired action string: {action_str}")
        
        try:
            action = json.loads(action_str)
            self.logger.info(f"Parsed action: {action}")
            return action
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing action JSON: {e}")
            print(f"Failed to parse action: {action_str} - Error: {e}")  # Print to stdout for terminal visibility
            return None
    
    def _execute_action(self, action):
        """Execute the specified action"""
        name = action.get("name")
        arguments = action.get("arguments", {})
        
        if name == "final_answer":
            self.logger.info("Executing final_answer action")
            
            # Print a summary of all searches performed before the final answer
            if self.verbose:
                print("\n" + "="*60)
                print("SEARCH SUMMARY BEFORE FINAL ANSWER".center(60))
                print("="*60)
                if "search_rag" in self.used_tools:
                    print("✓ RESEARCH PAPERS were searched (RAG)")
                else:
                    print("✗ RESEARCH PAPERS were NOT searched (RAG)")
                    
                if "search_web" in self.used_tools:
                    print("✓ WEB SOURCES were searched (Bing)")
                else:
                    print("✗ WEB SOURCES were NOT searched (Bing)")
                    
                if "read_paper" in self.used_tools:
                    print("✓ PAPERS were downloaded and analyzed in detail")
                else:
                    print("✗ No papers were analyzed in detail")
                print("="*60 + "\n")
            
            # Instead of returning is_final=True, we return is_checkpoint=True
            # This indicates we want to save progress but continue the conversation
            return {
                "result": arguments.get("answer", "No answer provided"), 
                "is_final": False,
                "is_checkpoint": True
            }
        
        if name not in self.tools:
            self.logger.warning(f"Tool '{name}' not found")
            return {"result": f"Error: Tool '{name}' not found", "is_final": False}
        
        # Track which tools have been used
        self.used_tools.add(name)
        
        tool = self.tools[name]
        
        if self.gui_mode and name == "ask_user":
            return {"result": None, "is_ask_user": True, "query": arguments.get("query", ""), "is_final": False, "is_checkpoint": False}
        
        # Handle different argument types for different tools
        if name == "read_paper":
            url = arguments.get("url", "")
            # Get the research question from the current context
            research_question = self._get_research_question_from_context()
            self.logger.info(f"Executing {name} with URL: {url} and research question: {research_question}")
            result = tool.execute(url, research_question)
            formatted_result = tool.format_result(url, result)
        else:
            query = arguments.get("query", "")
            # Print detailed info for the user to see what's happening
            if self.verbose:
                if name == "search_rag":
                    print(f"\n[USING RAG SEARCH] Searching research papers for: {query}")
                    print("-" * 60)
                    print("This search looks through academic papers, research documents, and scientific literature.")
                    print("Results will include information from peer-reviewed sources and academic publications.")
                    print("-" * 60)
                elif name == "search_web":
                    print(f"\n[USING BING SEARCH] Searching the web for: {query}")
                    print("-" * 60)
                    print("This search looks through web pages, news articles, blogs, and other online sources.")
                    print("Results will include the most recent and relevant information from the internet.")
                    print("-" * 60)
                elif name == "search_arxiv":
                    print(f"\n[USING ARXIV SEARCH] Searching arxiv papers for: {query}")
                    print("-" * 60)
                    print("This search looks through academic papers on Arxiv.")
                    print("Results will include recent research papers and preprints.")
                    print("-" * 60)
            
            self.logger.info(f"Executing {name} with query: {query}")
            result = tool.execute(query)
            formatted_result = tool.format_result(query, result)
        
        return {"result": formatted_result, "is_final": False}
    
    def _get_research_question_from_context(self):
        """Extract the research question from the conversation context"""
        # First try to use the original query if available
        if self.original_query:
            return self.original_query
        
        # Otherwise try to find it in the context
        for message in self.context:
            if message["role"] == "user":
                return message["content"].strip()
        
        return "What are the main findings and implications of this research?"
    
    def run(self, query):
        """Run the ReAct agent on a query"""
        self.logger.info(f"Running agent with query: {query}")
        print(f"\nQuery: {query}")
        
        if self.skip_session_creation:
            raise ValueError("This agent instance was created for listing sessions only")
        
        # Initialize system prompt at the start
        system_prompt = REACT_PROMPT.system_prompt.replace("{tools}", self.tools_description)
        
        # If no active session, create one and initialize context
        if not self.session_manager.current_session:
            self.session_manager.create_session(query)
            self.context = []
            self.used_tools = set()  # Reset tools for new session
        
        # Format the initial message with instructions
        initial_message = f"""
{query}

IMPORTANT INSTRUCTIONS:
You have to approach research like a human researcher collaborating with you:

1. You have to first reflect on your question to understand what you're asking and plan your approach.
2. You have main research tools:
   - search_rag: For searching internal documents and research papers
   - search_web: For searching public information on the internet
   - search_arxiv: For searching academic papers on Arxiv.org
   - read_paper: For downloading and reading academic papers
   - ask_user: Ask the user (supervisor) for feedback, clarification, or scope (don't use it unless you really need to)

3. For technical questions like "How can I quantify paraffin content in crude oil?", you have to check both internal resources and public information, asking clarifying questions when needed.

4. For factual questions like sports results, you have to primarily use web search and provide direct answers when available.

5. For company-specific questions like financial results, you have to prioritize internal documents while confirming with me if you need more context.

6. You have to think critically throughout the process - planning, analyzing, reconsidering approaches and ensuring you're addressing the needs effectively.

**ALWAYS CALL AN ACTION, don't forget about it.**

Remember: Your answers are checkpoints in an ongoing conversation. The user may provide feedback or ask follow-up questions.
"""
        
        # For existing sessions with context, add a separator
        if self.context:
            self.context.append({
                "role": "system",
                "content": "\n=== New Research Question ===\n"
            })
            self.used_tools = set()  # Reset tools for new query
        
        # Add the new query with instructions
        self.context.append({"role": "user", "content": initial_message})
        self.original_query = query
        
        iteration = 0
        final_answer = None
        while iteration < config.MAX_ITERATIONS:
            iteration += 1
            self.logger.info(f"Starting iteration {iteration}")
            print(f"\nIteration {iteration}----------------------------------")
            
            try:
                # Generate the next action
                self.logger.info("Generating model response")
                messages = [{"role": "system", "content": system_prompt}]  # Always include system prompt
                messages.extend(self.context)  # Add conversation context

                # Compute and display token count for context messages
                try:
                    import tiktoken
                    try:
                        encoding = tiktoken.encoding_for_model(self.model)
                    except Exception:
                        encoding = tiktoken.get_encoding("cl100k_base")
                    token_count = sum(len(encoding.encode(m["content"])) for m in messages)
                    print(f"[CONTEXT TOKENS]: {token_count}")
                except ImportError:
                    print("[CONTEXT TOKENS]: tiktoken library not installed, token count unavailable")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS
                )
                
                assistant_message = response.choices[0].message.content
                print(f"\nAssistant: {assistant_message}")
                self.context.append({"role": "assistant", "content": assistant_message})

                # Parse and execute the action
                action = self._parse_action(assistant_message)
                if not action:
                    self.logger.warning("Failed to parse action, asking for clarification")
                    self.context.append({"role": "user", "content": "I couldn't understand your action. Please provide a valid action in the format: Action: {\"name\": \"tool_name\", \"arguments\": {\"query\": \"your query\"}}."})
                    continue
                
                # Execute the action
                self.logger.info(f"Executing action: {action.get('name')}")
                result = self._execute_action(action)
                
                # Handle checkpoint (previously final answer)
                if result.get("is_checkpoint"):
                    self.logger.info("Checkpoint reached")
                    final_answer = result["result"]
                    # Save the query and its context to the session
                    self.session_manager.add_query_to_session(
                        query=query,
                        context=self.context,
                        used_tools=list(self.used_tools),
                        final_answer=final_answer
                    )
                    
                    # Add the checkpoint answer to the context
                    self.context.append({
                        "role": "assistant",
                        "content": f"Here's what I've found so far:\n\n{final_answer}\n\nWould you like me to explore any specific aspect further or do you have any questions about this?"
                    })
                    
                    return final_answer
                    
                # Format observation with "Observation:" prefix to match examples in prompts.py
                observation = f"Observation: {result['result']}"
                print(f"\nObservation: {observation}")
                self.context.append({"role": "user", "content": observation})
                self.logger.info("Added observation to context")
                
            except Exception as e:
                self.logger.error(f"Error during iteration {iteration}: {e}")
                error_msg = f"Error: {str(e)}"
                # Save the failed query attempt but maintain context
                self.session_manager.add_query_to_session(
                    query=query,
                    context=self.context,
                    used_tools=list(self.used_tools),
                    final_answer=error_msg
                )
                return error_msg
        
        # If we reach the maximum number of iterations, save and return the last response
        self.logger.warning(f"Maximum iterations ({config.MAX_ITERATIONS}) reached without final answer")
        max_iter_msg = "Maximum iterations reached without a final answer."
        self.session_manager.add_query_to_session(
            query=query,
            context=self.context,
            used_tools=list(self.used_tools),
            final_answer=max_iter_msg
        )
        return max_iter_msg

    def get_current_session_summary(self):
        """Get a summary of the current session"""
        if not self.session_manager.current_session:
            return None
        return self.session_manager.get_session_summary()

    def list_available_sessions(self):
        """List all available research sessions"""
        return self.session_manager.list_sessions() 

    def start_new_query(self, query):
        """Initialize a new query for the agent"""
        system_prompt = REACT_PROMPT.system_prompt.replace("{tools}", self.tools_description)
        
        if not self.session_manager.current_session:
            self.session_manager.create_session(query)
        self.used_tools = set()
        
        initial_message = f"""{query}

IMPORTANT INSTRUCTIONS:
You have to approach research like a human researcher collaborating with you:

1. You have to first reflect on your question to understand what you're asking and plan your approach.
2. You have main research tools:
   - search_rag: For searching internal documents and research papers
   - search_web: For searching public information on the internet
   - search_arxiv: For searching academic papers on Arxiv.org
   - search_semantic_scholar: For searching papers on Semantic Scholar
   - read_paper: For downloading and reading academic papers
   - ask_user: Ask the user (supervisor) for feedback, clarification, or scope (don't use it unless you really need to)

3. For technical questions like "How can I quantify paraffin content in crude oil?", you have to check both internal resources and public information, asking clarifying questions when needed.

4. For factual questions like sports results, you have to primarily use web search and provide direct answers when available.

5. For company-specific questions like financial results, you have to prioritize internal documents while confirming with me if you need more context.

6. You have to think critically throughout the process - planning, analyzing, reconsidering approaches and ensuring you're addressing the needs effectively.

**ALWAYS CALL AN ACTION, don't forget about it.**

Remember: Your answers are checkpoints in an ongoing conversation. The user may provide feedback or ask follow-up questions.
"""
        
        if self.context:
            self.context.append({
                "role": "system",
                "content": "\n=== New Research Question ===\n"
            })
            self.used_tools = set()
        
        self.context.append({"role": "user", "content": initial_message})
        self.original_query = query
        self.current_iteration = 0
        self.pending_user_query = None

    def provide_user_response(self, response):
        """Provide user response to a pending ask_user query"""
        if not self.pending_user_query:
            return
        formatted = f"Observation: User response to '{self.pending_user_query}': {response}"
        self.context.append({"role": "user", "content": formatted})
        self.pending_user_query = None

    def step(self):
        """Perform one iteration of the ReAct loop, capturing output for GUI"""
        import sys
        from io import StringIO
        
        output_capture = StringIO()
        original_stdout = sys.stdout
        sys.stdout = output_capture
        
        try:
            if self.pending_user_query:
                sys.stdout = original_stdout
                return {"type": "waiting_user", "output": ""}
            
            self.current_iteration += 1
            print(f"\nIteration {self.current_iteration}----------------------------------")
            
            # Compute token count
            messages = [{"role": "system", "content": REACT_PROMPT.system_prompt.replace("{tools}", self.tools_description)}] + self.context
            try:
                import tiktoken
                try:
                    encoding = tiktoken.encoding_for_model(self.model)
                except Exception:
                    encoding = tiktoken.get_encoding("cl100k_base")
                token_count = sum(len(encoding.encode(m["content"])) for m in messages if m["content"] is not None)
                print(f"[CONTEXT TOKENS]: {token_count}")
            except ImportError:
                print("[CONTEXT TOKENS]: tiktoken library not installed, token count unavailable")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS
            )
            
            assistant_message = response.choices[0].message.content
            print(f"\nAssistant: {assistant_message}")
            self.context.append({"role": "assistant", "content": assistant_message})
            
            action = self._parse_action(assistant_message)
            if not action:
                print("Failed to parse action from response.")
                sys.stdout = original_stdout
                return {"type": "error", "output": output_capture.getvalue()}
            
            result = self._execute_action(action)
            
            if result.get("is_ask_user"):
                self.pending_user_query = result["query"]
                print(f"\n[User Input Required] {self.pending_user_query}")
                sys.stdout = original_stdout
                return {"type": "ask_user", "query": result["query"], "output": output_capture.getvalue()}
            
            if result.get("is_checkpoint"):
                final_answer = result["result"]
                self.session_manager.add_query_to_session(
                    query=self.original_query,
                    context=self.context,
                    used_tools=list(self.used_tools),
                    final_answer=final_answer
                )
                self.context.append({
                    "role": "assistant",
                    "content": f"Here's what I've found so far:\n\n{final_answer}\n\nWould you like me to explore any specific aspect further or do you have any questions about this?"
                })
                print("\n" + "="*80)
                print("RESULT".center(80))
                print("="*80)
                print(final_answer)
                print("="*80)
                sys.stdout = original_stdout
                return {"type": "checkpoint", "answer": final_answer, "output": output_capture.getvalue()}
            
            observation = f"Observation: {result['result']}"
            print(f"\n{observation}")
            self.context.append({"role": "user", "content": observation})
            
            sys.stdout = original_stdout
            return {"type": "observation", "output": output_capture.getvalue()}
        
        except Exception as e:
            print(f"Error in step: {str(e)}")
            sys.stdout = original_stdout
            return {"type": "error", "output": output_capture.getvalue()}
        
        finally:
            sys.stdout = original_stdout 