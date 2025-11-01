from agents.data_agent import EarthEngineDataAgent

if __name__ == "__main__":
    agent = EarthEngineDataAgent()
    result = agent.process_city_data(
        city_name="Austin, Texas",
        data_types=["land_cover", "tree_cover", "ndvi"],
        year=2020
    )
    print(result)
