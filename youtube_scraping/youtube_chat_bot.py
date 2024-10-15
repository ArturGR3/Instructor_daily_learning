from youtube_transcript_api import YouTubeTranscriptApi
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
import re
import instructor
import openai
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from typing import List

# Load environment variables
load_dotenv(find_dotenv(usecwd=True))

# Initialize OpenAI client with instructor
client = instructor.from_openai(openai.OpenAI())

console = Console()

def extract_video_id(url: str) -> str | None:
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return None

def get_transcript(video_id: str) -> List[dict]:
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        console.print(f"[bold red]Error fetching transcript: {str(e)}[/bold red]")
        return []

class Topic(BaseModel):
    title: str = Field(description="A concise title for the topic")
    start_time: float = Field(description="The start time of the topic in seconds")

class VideoTopics(BaseModel):
    topics: List[Topic] = Field(max_items=10, description="Up to 10 main topics from the video")

def generate_topics(transcript: List[dict]) -> VideoTopics:
    transcript_text = " ".join([segment['text'] for segment in transcript])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that analyzes YouTube video transcripts and identifies the main topics discussed in the video."
            },
            {
                "role": "user",
                "content": f"Based on the following transcript, identify up to 10 main topics discussed in the video. For each topic, provide a concise title and the approximate start time in seconds.\n\nTranscript: {transcript_text}"
            }
        ],
        response_model=VideoTopics
    )
    
    return response

def display_topics(topics: VideoTopics):
    table = Table(title="Main Topics")
    table.add_column("Number", style="cyan", justify="right")
    table.add_column("Topic", style="magenta")
    table.add_column("Start Time", justify="right", style="green")
    
    for i, topic in enumerate(topics.topics, 1):
        table.add_row(str(i), topic.title, f"{topic.start_time:.0f}s")
    
    console.print(table)

def main():
    console.print("[bold]YouTube Chat Bot[/bold]")
    
    url = Prompt.ask("Enter a YouTube URL")
    video_id = extract_video_id(url)
    
    if not video_id:
        console.print("[bold red]Invalid YouTube URL[/bold red]")
        return
    
    with console.status("[bold green]Fetching transcript..."):
        transcript = get_transcript(video_id)
    
    if not transcript:
        return
    
    with console.status("[bold green]Generating topics..."):
        topics = generate_topics(transcript)
    
    display_topics(topics)
    
    topic_number = Prompt.ask("Enter the number of the topic you're interested in", choices=[str(i) for i in range(1, len(topics.topics) + 1)])
    selected_topic = topics.topics[int(topic_number) - 1]
    
    video_url = f"https://www.youtube.com/watch?v={video_id}&t={int(selected_topic.start_time)}s"
    console.print(f"\n[bold green]Here's the link to the video at the selected topic:[/bold green]")
    console.print(f"[link={video_url}]{video_url}[/link]")

if __name__ == "__main__":
    main()
