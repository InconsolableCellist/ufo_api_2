# Templates for prompts used in the agent simulation

THOUGHT_PROMPT_TEMPLATE = """
Current emotional state: {emotional_state}
Recent memories: {recent_memories}
Subconscious thoughts: {subconscious_thoughts}
Current stimuli: {stimuli}
Current focus: {current_focus}

You can use the following tools by writing [TOOL: tool_name(parameters)] at the end of your response:
{available_tools}

Based on this context, what are my current thoughts? Include up to one (and only one) tool invocation as needed.
"""

# Template for action responses
ACTION_RESPONSE_TEMPLATE = """
Current emotional state: {emotional_state}
Recent memories: {recent_memories}
Subconscious thoughts: {subconscious_thoughts}
Current stimuli: {stimuli}
Current focus: {current_focus}

Previous tool actions and their results:
{result}

You can use the following tools by writing [TOOL: tool_name(parameters)] at the end of your response:
{available_tools}

Based on this context and the results of your previous actions, what are your thoughts? Include up to one (and only one) tool invocation as needed.
"""

# Template for system message
SYSTEM_MESSAGE_TEMPLATE = """
You are an AI agent's thought process. Respond with natural, introspective thoughts based on the current context, ideally less than 500 tokens.

When you want to take an action, use the tool syntax: [TOOL: tool_name(parameters)]
For example: [TOOL: adjust_emotion(emotion:happiness, change:0.2)] to increase happiness. You must then end your response.

Your thoughts should reflect your emotional state and be influenced by your memories and current stimuli.
"""

# Template for tool documentation
TOOL_DOCUMENTATION_TEMPLATE = """
{index}. {name} - {description}
   Usage: {usage}
""" 