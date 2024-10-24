from crewai_tools import LlamaIndexTool
from rags import TravelGuideRAG
from prompts import travel_guide_qa_tpl

travel_guide_description = """
The Travel Guide Query Engine is an AI-powered tool that provides up-to-date travel advice and insights from a curated travel guidebook. Currently, it is based on the book Lonely Planet's: Bolivia, offering rich details to help plan your trip efficiently.

Capabilities include:
- Detailed highlights and itineraries to personalize your travel experience.
- Insider tips on saving time, avoiding crowds, and navigating like a local.
- Essential information on operating hours, websites, transit, and prices.
- Honest reviews across all budgets, covering food, sightseeing, shopping, and hidden gems.
- Cultural insights for a deeper understanding of local history, art, cuisine, and politics.
- Coverage of key destinations, such as La Paz, Lake Titicaca, Salar de Uyuni, and the Amazon Basin.

Simply input your travel questions or ask for recommendations in plain text, and the tool will provide accurate, context-rich responses to guide your journey.
Note:
    DO use this tool for recommendations, and general information retrieval. For ACTIONS use the specific 
    tool, flight, bus, hotel or restaurant.
"""

travel_guide_rag_tool = LlamaIndexTool.from_query_engine(
    TravelGuideRAG(
        store_path="travel_guide_store", 
        qa_prompt_tpl=travel_guide_qa_tpl
    ).get_query_engine(),
    name="Bolivia Travel guide",
    description=travel_guide_description
)