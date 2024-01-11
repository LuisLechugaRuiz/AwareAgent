# Aware: AutoGPT Hackathon Submission - Autonomous Operations

You can see the AwareAgent abilities at: https://github.com/LuisLechugaRuiz/AwareAgent/tree/master/autogpts/AwareAgent/forge/sdk/abilities

**Note:** For detailed insights into the technical implementation, including our ChatParser technology and intelligent token management, please refer to our [Technical Implementation Details](./README-technical.md). These essential elements are integral to leveraging large language models and managing resources effectively.

At the heart of our submission for the AutoGPT Hackathon is Aware, an autonomous agent powered by advanced language models and designed to independently navigate tasks with minimal human intervention. Central to its functionality is the ability to manage and adapt its objectives, an innovative step forward in autonomous technology.

## Autonomous Goals Management

The cornerstone of Aware is its capacity for autonomous goal management, a testament to the agent's advanced design. This system enables the agent to create, pursue, and update its objectives based on real-time feedback from its environment and actions.

### Goal-Centric Architecture

Embedded within the agent is a structured representation of a 'Goal,' exemplified in its programming as follows:

```python
class Goal(LoggableBaseModel):
    description: str = Field("The description of the goal.")
    ability: str = Field(
        "The name of the ability (only name, without arguments) that should be used to achieve the goal should be one of the available capabilities, is very important that you verify that the goal can be achieved using this ability."
    )
    validation_condition: str = Field(
        "Explicit criteria acting as the benchmark for goal completion, essential for assessing the outcome's alignment with desired objectives. It serves as a conclusive checkpoint for the current goal and a foundational prerequisite for subsequent objectives."
    )
    status: str = Field(
        "Should be one of the following: NOT_STARTED, IN_PROGRESS, SUCCEEDED, FAILED."
    )
```

This architecture is fundamental for the agent's autonomous decision-making process, enabling it to:

- Assess outcomes against its set validation conditions.
- Dynamically adjust its strategies based on previous experiences and outcomes.
- Continuously learn and refine its approach towards goal attainment.

## Task Execution Framework

Aware operates through a systematic two-stage process, designed for efficient navigation through its task list and optimal use of its abilities.

### Planning Stage

During this initial phase, the agent actively:

- Manages its list of objectives, ensuring clarity, relevance, and attainability.
- Assigns the appropriate 'ability' to each goal, a critical step requiring an understanding of the task demands and the agent's capabilities.

### Execution Stage

Here, the agent transitions into active task mode, where it:

- Commences the execution of its immediate goal, applying the designated ability with precision.

By maintaining a clear separation between the planning and execution stages, Aware promotes a focused, organized approach to task management. This structure ensures thorough preparation for each task and allows for real-time adjustments, contributing to the agent's adaptability and success rate.

## Abilities

In the context of the hackathon, we've equipped Aware with several distinct abilities, each designed for optimal interaction and information processing.

## Search

The "Search" ability signifies a breakthrough in autonomous information retrieval and analysis, showcasing a seamless blend of LLMs, vector search, and web scraping.

### Architecture and Pipeline

1. **Query Formation and Validation Criteria**: The agent autonomously formulates multiple search queries based on the target information needed. Accompanying these queries is a validation criterion, ensuring that the retrieved data aligns precisely with the requirements.
2. **Google Search Integration**: Leveraging an automated system, the agent performs searches on Google for each formulated query, collecting the URLs of the most relevant results.
3. **Advanced Web Scraping**: Each link undergoes an extraction process where the web content is scraped using Selenium. The data is then parsed into manageable textual segments, employing an intelligent token counter algorithm for efficiency.
4. **Vector Database Storage with Weaviate**: The textual segments are indexed into Weaviate, a vector search engine, enabling nuanced, context-aware retrieval of information.
5. **Custom Prompt for LLMs and Data Validation**: The most pertinent segments are compiled into a single custom prompt. The LLM processes this prompt, extracts the essential data, and cross-verifies it against the validation criteria to ensure accuracy and relevance.

This elaborate pipeline transforms the agent into an efficient researcher, capable of sifting through vast information with precision, replicating a more enhanced, automated version of a manual research process.

## Data

Under the "Data" ability, we introduce an innovative approach, delegating intricate data manipulation tasks to a specialized agent known as "pandasai." This entity utilizes LLM's prowess to perform data operations typically executed with Pandas in Python.

### Key Functionalities

- **process_csv_and_save**: This function empowers the agent to carry out complex computations or transformations on a CSV file's data. It comprehends the task, applies the necessary Pandas operations, and stores the result in a new CSV file, all autonomously.
- **get_insights_from_csv**: Here, the agent probes the CSV data to answer specific questions, drawing insights directly from the dataset. It's capable of understanding diverse queries and responding in natural language, facilitating effortless interpretation of the data.

We've upgraded "pandasai" to intelligently display entire CSV files when needed, enhancing its analysis accuracy for complex tasks.

## Coding

Aware's programming skills are rooted in two key functions, enabling the system to create and refine code seamlessly.

### Core Functions

- **create_code**: Initiates and continuously refines code through a systematic improvement process.
- **modify_code**: Adjusts existing code by identifying and enhancing specific segments based on set parameters.

These methods rely on detailed code requirements and strict validation criteria to ensure precision and effectiveness.

### The Improvement Loop

Aware employs an "Improvement Loop," a cyclical refinement process:

1. **Test Creation**: Constructs tests from the validation criteria.
2. **Test Execution**: Runs the tests, analyzing results for discrepancies.
3. **Error Analysis and Improvement Proposal**: An LLM diagnoses failures, determining whether the issue lies in the code or test, and suggests improvements.
4. **Implementation of Enhancements**: Applies suggested refinements to the code or test.

This loop repeats until the code passes the tests or reaches the maximum iteration limit.

## Conclusion

Aware stands as our innovative contribution to the AutoGPT Hackathon, showcasing an autonomous agent's leap towards self-directed functionality.

## Highlights

- **Autonomy**: Aware excels in self-managing goals, adapting in real-time based on outcomes and environmental feedback.
- **Task Management**: It tactically handles tasks, balancing between strategic planning and decisive execution.
- **Abilities**: From deep digital research to nuanced data handling and agile coding, Aware harnesses diverse functionalities, demonstrating proficiency across various domains.

In essence, Aware encapsulates the future of autonomous systems, bringing theoretical AI autonomy into tangible reality.
