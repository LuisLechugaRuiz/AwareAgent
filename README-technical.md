# Aware: The Autonomous Agent

## Introduction

'Aware' stands as a testament to the advancements in autonomous agents, demonstrating efficient and strategic use of abilities. The robustness of this agent is not coincidental but rather a result of several innovations in the realm of Language Model technology and autonomous functionality. Below, we delve into the technical specifics that make 'Aware' a pioneering solution in this field.

## Innovations

### ChatParser

In the rapidly evolving landscape of Large Language Models (LLMs), extracting precise information from these models remains a challenge. The traditional approach involves crafting careful prompts that solicit specific data, avoiding unnecessary or tangential details. However, this method proves inefficient over time, restricting the LLM's potential.

Our solution, the ChatParser, revolutionizes this interaction. It acts as a wrapper that utilizes object-oriented principles, accepting a Pydantic-formatted class and a system prompt as inputs. The ChatParser then autonomously interacts with the LLM, ensuring the extraction of data aligns with the desired class instance format. This breakthrough allows for the development of more complex programs, empowering the LLM to function akin to a general-purpose function, dynamically catering to diverse informational needs.

### Memory Systems

#### Short-Term Memory

Every autonomous agent built upon LLMs encounters the constraint of prompt length. To address this, we introduce our 'episodic memory' system, drawing inspiration from human cognitive science and reinforcement learning methodologies. This system revolves around 'episodes' that record past actions and observations, providing a structured approach to historical data.

Moreover, our memory system intelligently manages token usage, offering configurable size limitations for episodes and executing rolling summaries to maintain an optimal balance between detailed recency and historical context.

#### Long-Term Memory

In addressing the need for persistent memory, we employ 'Weaviate,' a versatile solution that supports our episodic schema. It allows for the storage and retrieval of past episodes, enhancing the agent's continuity of experience and learning potential.

**Note**: Long-term memory integration is currently disabled for the AutoGPT Hackathon, ensuring the agent's performance is evaluated based on strategic execution rather than mere historical retrieval. Future iterations will reintroduce this feature, alongside a sophisticated learning mechanism that leverages goal-oriented outcomes from sequential episodes, enabling genuine, context-aware learning.
