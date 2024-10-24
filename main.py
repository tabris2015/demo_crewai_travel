from dotenv import load_dotenv
from crewai import Agent, Crew, Task
from tools import travel_guide_rag_tool

load_dotenv()

user_query_str = input("Ask me anything about traveling in Bolivia: ")

travel_expert = Agent(
    role="Expert travel guide",
    goal="Provide useful information and recommendations about different travel destination.",
    backstory="""You are a experienced travel specialized in Bolivia. 
    Your knowledge allows you to offer very useful advice to travelers that want to visit Bolivia.""",
    tools=[travel_guide_rag_tool]
)

get_info_task = Task(
    description=f"Get precise and relevant information about the user query using the provided travel guide tool. User query: {user_query_str}",
    expected_output="A brief summary with details, facts and tips about the places to travel.",
    agent=travel_expert,
)

recommend_task = Task(
    description=f"""Recommend a list of places of interest to visit or activities
      to do in a given city in Bolivia using the original user query: {user_query_str}, 
      and the information obtained from the previous results""",
    expected_output="""A detailed report of places or activities in a bullet list.
    Include information such as availability, price range, popularity""",
    agent=travel_expert,
)

crew = Crew(
    agents=[travel_expert],
    tasks=[get_info_task, recommend_task],
    verbose=True
)

result = crew.kickoff()

print(result)