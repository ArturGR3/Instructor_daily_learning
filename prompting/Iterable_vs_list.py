import openai
import instructor
from pydantic import BaseModel, Field
from typing import Iterable, Literal
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

class Dog(BaseModel):
    breed: str
    country: str
    best_quality: Literal["speed", "loyalty", "cuteness"]

class DogList(BaseModel):
    dogs: list[Dog] = Field(description="List of 5 dog breeds")

client = instructor.from_openai(openai.OpenAI())

def get_dog_breeds(query: str, approach: str = "iterable"):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Iterable[Dog] if approach == "iterable" else DogList,
        stream=approach == "iterable",
        messages=[
            {"role": "user", "content": query},
            # {"role": "system", "content": "You are a canine expert."}
        ],
    )

if __name__ == "__main__":
    query = "Provide information about 5 unique dog breeds from different countries."

    # Iterable approach
    print("Using Iterable approach:")
    for dog in get_dog_breeds(query, "iterable"):
        print(f"{dog.breed} from {dog.country} - Best quality: {dog.best_quality}")

    print("\nUsing List approach:")
    # List approach
    dog_list = get_dog_breeds(query, "list")
    for dog in dog_list.dogs:
        print(f"{dog.breed} from {dog.country} - Best quality: {dog.best_quality}")