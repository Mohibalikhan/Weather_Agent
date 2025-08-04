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
    instructions="Tum ak Weather Agent ho tum mujhe temperature batao ge jis city ka weather pochon sirf mujhe temperature batao degree me bs", 
    
)

result = Runner.run_sync(
    agent,
    input = cityname,
    run_config=config
    )



print(result.final_output)