from youtube_transcript_api import YouTubeTranscriptApi
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
import re
import instructor
import openai
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict, Generator, Iterable, Tuple
import csv
import json
from rich.live import Live
from rich.status import Status
import time

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

def save_transcript_to_csv(video_id: str, transcript: List[Dict[str, any]]):
    filename = f"transcript_{video_id}.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['start', 'duration', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for segment in transcript:
            writer.writerow(segment)
    
    console.print(f"[bold green]Transcript saved to {filename}[/bold green]")
    
class TranscriptSegment(BaseModel):
    source_id: int
    start: float
    text: str

def get_transcript(source: str) -> Tuple[Generator[TranscriptSegment, None, None], str]:
    """
    Fetches the transcript from a YouTube video URL or loads it from a CSV file,
    and yields TranscriptSegment objects.
    
    :param source: YouTube video URL or path to a CSV file
    :yield: TranscriptSegment objects
    """
    if source.endswith('.csv'):
        with open(source, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            data = list(reader)
            video_id = source.split("_")[1].split(".")[0]
    else:
        video_id = extract_video_id(source)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        data = YouTubeTranscriptApi.get_transcript(video_id)
        save_transcript_to_csv(video_id, data)
    transcript_segments = []
    for index, segment in enumerate(data):
        transcript_segments.append(TranscriptSegment(
            source_id=index,
            start=float(segment['start']),
            text=segment['text']
        ))
    return transcript_segments, video_id

class Topic(BaseModel):
    topic_id: int = Field(description="The id of the topic")
    title: str = Field(description="A concise title for the topic")
    summary: str = Field(description="A short summary of the topic")
    keywords: List[str] = Field(description="Key words or phrases related to this topic")
    start_time: float = Field(description="The start time of the topic in seconds")
    end_time: float = Field(description="The end time of the topic in seconds")

class Topics(BaseModel):
    topics: List[Topic] = Field(description="The list of topics")

def generate_topics(segments: Iterable[TranscriptSegment]) -> Iterable[Topics]:
    
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an AI assistant that analyzes YouTube video transcripts and identifies the main topics discussed in the video.
                You are given a sequence of YouTube transcript segments and your job is to return up to 5 main topics discussed in the video. 
                For each topic, provide a concise title, a short summary, relevant keywords, and the approximate start and end times in seconds.
                Note that this is a transcript and so there might be spelling errors. Note that and correct any spellings. Use the context to 
                make sure you're spelling things correctly."""
            },
            {
                "role": "user",
                "content": f"""Based on the following transcript, identify up to 5 main topics discussed in the video. 
                For each topic, provide a concise title, a short summary, relevant keywords, and the approximate start and end times in seconds.
                \n\nTranscript: {segments}"""
            }
        ],
        response_model=instructor.Partial[Topics],
        stream=True
    )    
    
# transcript, video_id = get_transcript('transcript_OzNuAg2bx6k.csv')
# topics = generate_topics(transcript)

# for partial_topics in topics:
#     print(json.dumps(partial_topics.model_dump(), indent=4))  
    

def display_topics(transcript):
    console = Console()
    topics_generator = generate_topics(transcript)
    topics_list = []
    added_topic_ids = set()  # To keep track of topics we've already added

    console.print("[bold green]Generating topics...[/bold green]")

    table = Table(title="Main Topics", show_lines=True)
    table.add_column("Number", style="cyan", justify="right")
    table.add_column("Topic", style="white")
    table.add_column("Summary", style="green")
    table.add_column("Keywords", style="yellow")
    table.add_column("Time Range", justify="right", style="white")

    with Live(table, console=console, refresh_per_second=4) as live:
        for partial_topics in topics_generator:
            for topic in partial_topics.topics or []:
                if (topic.topic_id not in added_topic_ids and
                    all([topic.topic_id, topic.title, topic.summary, topic.keywords, 
                         topic.start_time is not None, topic.end_time is not None])):
                    
                    summary = topic.summary or ""
                    keywords = topic.keywords or []
                    start_time = topic.start_time if topic.start_time is not None else 0
                    end_time = topic.end_time if topic.end_time is not None else 0
                    
                    table.add_row(
                        str(topic.topic_id),
                        topic.title,
                        summary[:70] + "..." if len(summary) > 70 else summary,
                        ", ".join(keywords[:5]) + ("..." if len(keywords) > 5 else ""),
                        f"{start_time:.0f}s - {end_time:.0f}s"
                    )
                    topics_list.append(topic)
                    added_topic_ids.add(topic.topic_id)
                    
                    live.update(table)
            #         time.sleep(0.5)  # Pause briefly to make the addition visible

            # console.print("[bold green]Updating topics...[/bold green]")
            # time.sleep(1)  # Pause between batches of topics

    if not topics_list:
        console.print("No topics were generated.")

    return topics_list

# topics_list = display_topics(transcript)

class Answer(BaseModel):
    question: str = Field(description="The question that the user asked")
    answer: str = Field(description="The answer to the user question")
    start_time: float = Field(description="The start time of the video that you used to answer the question")
    end_time: float = Field(description="The end time of the video that you used to answer the question")

def answer_question(transcript: List[TranscriptSegment], topics: List[Topic], question: str) -> Generator[Answer, None, None]:
    """
    This function uses an LLM to determine which topics are relevant to a given question.
    It then extracts the relevant parts of the transcript based on the start and end times of the topics.
    """     
    # convert dict to markdown topic id, title, summary, keywords, start_time, end_time
    topics_md = "\n".join([f"Topic id: {topic.topic_id}, Title: {topic.title}, Summary: {topic.summary}, Keywords: {topic.keywords}, Start time: {topic.start_time}, End time: {topic.end_time}" for topic in topics])
    
    answer_generator = client.chat.completions.create_iterable(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "system",
                "content": """You are an AI assistant that uses the youtube video transcript and it main topics to answer user questions.
                Please provide the user question and the answer with the start and end times of the video that you used to answer the question.
                """
            },
            {
                "role": "assistant",
                "content": f"""[Transcript from the video]:\n{transcript}
                \n[Topics from the transcript]:\n{topics_md}"""
            },
            {
                "role": "user",
                "content": f"""Please provide the user question and the answer with the start and end times of the video that you used to answer the question.
                \n[Question]:\n {question}"""
            }
        ],
        response_model=Answer,
        stream=True
    )
    
    # Print the relevant topics numbers, titles and start and end times 
    console.print("[bold green]Answer:[/bold green]")
    for partial_answer in answer_generator:
        console.print(f"[bold green]Question:[/bold green] {partial_answer.question}")
        console.print(f"[bold green]Answer:[/bold green] {partial_answer.answer}")
        console.print(f"[bold green]Start time:[/bold green] {partial_answer.start_time}")
        console.print(f"[bold green]End time:[/bold green] {partial_answer.end_time}")
        return partial_answer        

# answer = answer_question(transcript, topics_list, "What is the main topic of the video?")
    
def main():
    console = Console()
    console.print("[bold]YouTube Chat Bot[/bold]")
    
    input_value = Prompt.ask("Enter a CSV filename or YouTube URL")
    
    with console.status("[bold green]Fetching transcript..."):
        transcript, video_id = get_transcript(input_value)
        
    if not transcript:
        console.print("[bold red]Invalid input[/bold red]")
        return
    
    with console.status("[bold green]Generating topics...") as status:
        topics = display_topics(transcript)
    
    if not topics:
        console.print("[bold red]Failed to generate topics[/bold red]")
        return
    
    while True:
        action = Prompt.ask("What would you like to do?", choices=["question", "exit"])
        
        if action == "exit":
            break
        elif action == "question":
            question = Prompt.ask("What's your question about the video?")
            with console.status("[bold green]Analyzing question and generating answer..."):
                answer= answer_question(transcript, topics, question)
            
            video_url = f"https://www.youtube.com/watch?v={video_id}&t={int(answer.start_time)}s"
            console.print(f"\n[bold green]Here's the link to the video at the relevant part:[/bold green]")
            console.print(f"[link={video_url}]{video_url}[/link]")

if __name__ == "__main__":
    main()


# transcript_OzNuAg2bx6k.csv



