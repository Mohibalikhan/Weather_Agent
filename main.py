from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI,RunConfig

import os 

from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)
config =RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

print("Weather Expert by Mohib Ali Khan")

cityname =input("Enter City name u want to Predict the Weather: ") 

agent = Agent(
    name="Weather Expert Agent",
    instructions="""
    Tum ek Weather Agent ho. Jab bhi me kisi city ka weather pochun, tum mujhe sirf ye cheezen batao:
    is taran dikhao
    1. Temperature: °C me
    2. Weather condition: (sirf English me, jaise Sunny, Cloudy, Rainy waghera)
    3. Location: Province name, Country name
    Koi extra baat ya info mat dena — sirf ye teen cheezen.
"""
)


result = Runner.run_sync(
    agent,
    input = cityname,
    run_config=config
    )



print(result.final_output)