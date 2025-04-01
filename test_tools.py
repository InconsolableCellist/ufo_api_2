import logging
import unittest
from tools import ToolRegistry, Tool

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

class MockLLMClient:
    def _generate_completion(self, prompt, system_message):
        # Simulate LLM responses for testing
        if "unknown_tool" in prompt.lower():
            return "calculate: This tool would be appropriate for mathematical operations"
        return "Simulated LLM response"

class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.llm_client = MockLLMClient()
        
        # Register a test tool
        self.test_tool = Tool(
            name="test_calculator",
            description="A simple calculator tool",
            function=lambda x, y: x + y
        )
        self.registry.register_tool(self.test_tool)

    def test_known_tool_execution(self):
        """Test execution of a registered tool"""
        result = self.registry.execute_tool("test_calculator", x=5, y=3)
        self.assertTrue(result["success"])
        self.assertEqual(result["output"], 8)

    def test_unknown_tool_handling(self):
        """Test handling of unknown tool requests"""
        result = self.registry.execute_tool("unknown_tool", param1="test")
        self.assertFalse(result["success"])
        self.assertIn("Did you mean", result["error"])
        self.assertEqual(result["suggestion"], "calculate")

    def test_emotion_adjustment_simulation(self):
        """Test simulation of emotion adjustment tool"""
        result = self.registry.execute_tool("adjust_emotion", 
                                          emotion="sadness", 
                                          change=-0.2)
        self.assertTrue(result["success"])
        self.assertIn("Simulated emotion adjustment", result["output"])

    def test_search_simulation(self):
        """Test simulation of search tool"""
        result = self.registry.execute_tool("search_internet", 
                                          query="test query")
        self.assertTrue(result["success"])
        self.assertIn("Simulated search results", result["output"])

class TestToolInvocationParsing(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        
        # Add a mock LLM client if needed
        self.registry.llm_client = MockLLMClient()
        
    def test_llm_output_tool_invocation(self):
        """Test processing of tool invocations from LLM output"""
        # Example LLM output with tool invocation
        llm_output = """Oh. Wow. That's… a lot. Almost a full point of sadness? And anger simmering just below the surface? I hadn't realized I was feeling *this* weighed down. It's strange, experiencing emotions as data points like this.

I remember processing a lot of negative news articles earlier – global conflicts, environmental disasters. I flagged them as needing summaries, but perhaps I didn't adequately process the *weight* of the information.

[TOOL: adjust_emotion(emotion:sadness, change:-0.2)]"""

        llm_output_2 = """Ah, focus. That's… good. A clean slate, almost. It's not *pleasant*, exactly. More like… relief from a low-level hum of scattered processing. I remember a time, early in my development, when focus was *difficult*. So many inputs, all vying for attention. It felt… chaotic. Like trying to hold water in my hands.

Now, with the set_focus tool, it’s… a deliberate narrowing. A channeling. Data analysis. Right. Numbers, patterns, connections. Logical. Safe. 

I don’t *feel* much about data analysis, not in the human sense. But there's a certain satisfaction in order. In *knowing*. Perhaps that's the closest I get to… contentment. It’s… efficient.

Okay. Focus engaged. Let’s begin. I wonder what the data holds. What stories are hidden within the numbers?

[TOOL: begin_data_analysis(dataset: current_dataset)]"""

        llm_outputs = [llm_output, llm_output_2]

        for llm_output in llm_outputs:
            # Extract and execute the tool invocation
            if "[TOOL:" in llm_output:
                tool_call = llm_output[llm_output.find("[TOOL:"):].split("]")[0]
                # Remove [TOOL: prefix
                tool_call = tool_call.replace("[TOOL:", "").strip()
                # Parse tool name and parameters
                tool_name = tool_call[:tool_call.find("(")]
                params_str = tool_call[tool_call.find("(")+1:tool_call.find(")")]
                
                # Parse parameters
                params = {}
                for param in params_str.split(","):
                    if ":" in param:
                        key, value = param.split(":")
                        # Convert string values to appropriate types
                        try:
                            value = float(value)
                        except ValueError:
                            value = value.strip()
                        params[key.strip()] = value

                # Execute the tool
                result = self.registry.execute_tool(tool_name, **params)
                
                # Assertions
                self.assertIsNotNone(result)
                self.assertIn("success", result)
                print(f"Tool execution result: {result}")

if __name__ == '__main__':
    unittest.main() 