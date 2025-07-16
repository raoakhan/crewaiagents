from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class CallAnalytics():
    """CallAnalytics crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def transcriber(self) -> Agent:
        return Agent(
            config=self.agents_config['transcriber'],  # type: ignore[index]
            verbose=True
        )

    @agent
    def summariser(self) -> Agent:
        return Agent(
            config=self.agents_config['summariser'],  # type: ignore[index]
            verbose=True
        )

    @agent
    def sentiment_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['sentiment_analyst'],  # type: ignore[index]
            verbose=True
        )

    @agent
    def insights_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['insights_analyst'],  # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def transcribe_task(self) -> Task:
        return Task(
            config=self.tasks_config['transcribe_task'],  # type: ignore[index]
        )

    @task
    def summarise_task(self) -> Task:
        return Task(
            config=self.tasks_config['summarise_task'],  # type: ignore[index]
        )

    @task
    def sentiment_task(self) -> Task:
        return Task(
            config=self.tasks_config['sentiment_task'],  # type: ignore[index]
        )

    @task
    def insights_task(self) -> Task:
        return Task(
            config=self.tasks_config['insights_task'],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CallAnalytics crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        from call_analytics.listeners import TraceListener  # noqa: WPS433

        from call_analytics.tools import TranscribeTool, SentimentTool, InsightTool  # noqa: WPS433

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            tools=[TranscribeTool(), SentimentTool(), InsightTool()],  # instances, names already set to snake_case
            listeners=[TraceListener()],
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
