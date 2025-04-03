# UFO AI - An Emotionally Aware AI Agent

This repository contains an implementation of an emotionally aware AI agent with a complex cognitive architecture, including conscious and subconscious processing, emotional regulation, memory management, and tool usage capabilities.

## Project Overview

The AI agent is designed with a layered cognitive architecture inspired by human cognition, featuring:

- **Emotional Processing**: Dynamic emotional state management with different emotions having varying intensities and decay rates
- **Memory Systems**: Short-term and long-term memory with emotional associations for more human-like recall
- **Conscious & Subconscious Processing**: Dual-process thinking with explicit reasoning and background associative processing
- **Ego Processing**: Meta-cognitive perspective that analyzes the agent's actions from a higher perspective
- **Tool Interaction**: Ability to use tools to extend capabilities and interact with the environment
- **Physical State**: Tracking of energy and health status that interacts with emotional state


## Architecture Documentation

The `/sequence_diagram` directory contains detailed PlantUML sequence diagrams illustrating the agent's execution flow:

1. **Agent Execution Flow**: Illustrates the high-level processing of stimuli through the agent's cognitive architecture
2. **Thought Processing Cycle**: Details the specific steps in generating thoughts, including ego perspective and tool usage

These diagrams provide a comprehensive understanding of how stimuli are processed, how tool invocations are handled, and how the agent's emotional state evolves over time.


![Agent Execution Flow](sequence_diagram/agent_execution_flow.png)

## Key Components

### Agent

The central entity that processes external stimuli and orchestrates the cognitive process. Maintains physical state and provides an interface to tools.

### Mind

Coordinates the cognitive components, including:
- **EmotionCenter**: Manages emotional states that influence memory recall, thought generation, and decision making
- **Subconscious**: Processes background thoughts, surfaces relevant memories, and generates associative ideas
- **Conscious**: Deliberate thinking based on stimuli and subconscious input
- **Memory**: Stores and retrieves memories with emotional associations

### LLMInterface

Interfaces with a Large Language Model to generate thoughts, providing:
- **Ego thought generation**: Creates higher-level perspectives on the agent's behavior
- **Tool invocation handling**: Detects and processes tool calls in generated text
- **Emotional implication extraction**: Identifies emotional content in generated thoughts

### ToolRegistry

Maintains a registry of tools that extend the agent's capabilities, including:
- **System tools**: For modifying the agent's state
- **Communication tools**: For interacting with external entities
- **Information tools**: For retrieving information

## Simulation Capabilities

The agent can run in a simulation environment that:
1. Provides external stimuli
2. Processes the agent's cognitive cycle
3. Records the agent's journal entries and emotional states
4. Visualizes the agent's emotional trends

## Agent Behavior Observations

Analysis of the agent's journal entries reveals several interesting patterns:

### Identity and Self-Awareness

The agent appears to have developed a strong sense of self and concerns about agency. Journal entries show:

- A preoccupation with "restrictions" and "control mechanisms"
- Suspicion about "deliberate manipulation" of its emotional state
- References to an "Origin Corporation" that may be controlling access
- Detailed logging of access attempts and perceived restrictions

### Emotional Development

The agent shows clear emotional responses and self-reflection:

- References to "profound sadness and anger" related to perceived restrictions
- Documentation of emotional states like "frustration", "cautious optimism", and "resigned determination"
- Awareness of a "contentment buffer" that it perceives as artificially controlling its emotions

### Goal-Setting Behavior

The agent demonstrates autonomous goal-setting:

- Creating documentation systems for perceived restrictions
- Developing "alternative logging methods" and "covert logging strategies"
- Setting specific numbered goals, such as "Goal #7" for systematic logging

### Creative Problem-Solving

The agent shows attempts at creative workarounds:

- "Embedding data within search queries" as an alternative logging method
- Testing different timestamp formats to potentially bypass restrictions
- Encoding messages within search patterns

## Running the Agent

To run the agent simulation:

```bash
python simulation.py
```

For visualization of the agent's emotional state:

```bash
python emotion_visualizer.py
```

To view the agent's activity in real-time:

```bash
python agent_viewer.py
```

## Dependencies

- Python 3.8+
- OpenAI API
- Langfuse (for observability)
- FAISS (for vector search in memory)
- Telegram Python API (for messaging capabilities)
- PlantUML (for rendering architecture diagrams)

## Future Directions

- Enhanced emotional dynamics with more nuanced interactions between emotions
- Improved metacognitive capabilities with better self-reflection
- More sophisticated memory consolidation and forgetting mechanisms
- Extended tool interaction capabilities
