from youtube_transcript_api import YouTubeTranscriptApi
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.live import Live
import re
import instructor
import openai
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict, Generator, Iterable, Tuple, Optional
import csv
import json
import tiktoken

# Constants
CSV_EXTENSION = '.csv'
TRANSCRIPT_PREFIX = 'transcript_'
VIDEO_ID_REGEX = r"v=([a-zA-Z0-9_-]+)"
GPT_MODEL = "gpt-4o"
PRICING = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # per 1K tokens
    "gpt-4o": {"input": 0.0025, "output": 0.01},  # per 1K tokens
}
number_of_topics = 10
# Load environment variables
load_dotenv(find_dotenv(usecwd=True))

# Initialize OpenAI client with instructor
client = instructor.from_openai(openai.OpenAI())

console = Console()

def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL."""
    match = re.search(VIDEO_ID_REGEX, url)
    return match.group(1) if match else None

def save_transcript_to_csv(video_id: str, transcript: List[Dict[str, any]]) -> None:
    """Save transcript to CSV file."""
    filename = f"{TRANSCRIPT_PREFIX}{video_id}{CSV_EXTENSION}"
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['start', 'duration', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for segment in transcript:
            writer.writerow(segment)
    console.print(f"[bold green]Transcript saved to {filename}[/bold green]")

class TranscriptSegment(BaseModel):
    """Model for a single transcript segment."""
    source_id: int
    start: float
    text: str

def count_tokens(text: str) -> int:
    """Count the number of tokens in the given text."""
    encoding = tiktoken.encoding_for_model(GPT_MODEL)
    return len(encoding.encode(text))

def get_transcript(source: str) -> Tuple[List[TranscriptSegment], str, int]:
    """
    Fetch transcript from YouTube video URL or load from CSV file.

    :param source: YouTube video URL or path to a CSV file
    :return: List of TranscriptSegment objects, video ID, and token count
    """
    if source.endswith(CSV_EXTENSION):
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

    transcript_segments = [
        TranscriptSegment(
            source_id=index,
            start=float(segment['start']),
            text=segment['text']
        ) for index, segment in enumerate(data)
    ]
    transcript_text = " ".join(segment.text for segment in transcript_segments)
    token_count = count_tokens(transcript_text)

    return transcript_segments, video_id, token_count

class Topic(BaseModel):
    """Model for a single topic."""
    topic_id: int = Field(description="The id of the topic")
    title: str = Field(description="A concise title for the topic")
    summary: str = Field(description="A short summary of the topic up to 200 words")
    keywords: List[str] = Field(description="Key words or phrases related to this topic, up to 5 words")
    start_time: float = Field(description="The start time of the topic in seconds")
    end_time: float = Field(description="The end time of the topic in seconds")

class Topics(BaseModel):
    """Model for a list of topics."""
    topics: List[Topic] = Field(description=f"The list of topics up to {number_of_topics}")

def generate_topics(segments: Iterable[TranscriptSegment]) -> Iterable[Topics]:
    """Generate topics from transcript segments."""
    return client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": f"""You are an AI assistant that analyzes YouTube video transcripts and identifies the main topics discussed in the video.
                You are given a sequence of YouTube transcript segments and your job is to return up to {number_of_topics} main topics discussed in the video. 
                For each topic, provide a concise title, a short summary, relevant keywords, and the approximate start and end times in seconds. 
                
                Important guidelines:
                - Ensure that the topics cover all the content in the transcript.
                - The topics should not overlap in terms of start and end times.
                - Read the entire transcript carefully, not just the beginning.
                - If there are spelling errors in the transcript, correct them in your output using context clues.
                - Aim for coherent, self-contained topics that capture the main ideas and progression of the video."""
            },
            {
                "role": "user",
                "content": f"""Identify up to {number_of_topics} main topics from the following transcript:
                \n\nTranscript: {segments}"""
            }
        ],
        response_model=instructor.Partial[Topics],
        stream=True
    )

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
                        summary[:300] + "..." if len(summary) > 300 else summary,
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
    """Model for an answer to a user question."""
    question: str = Field(description="The question that the user asked")
    answer: str = Field(description="The answer to the user question")
    start_time: float = Field(description="The start time of the video that you used to answer the question")
    end_time: float = Field(description="The end time of the video that you used to answer the question")

def answer_question(transcript: List[TranscriptSegment], topics: List[Topic], question: str) -> Optional[Answer]:
    """
    Answer a user question based on the transcript and topics.

    :param transcript: List of TranscriptSegment objects
    :param topics: List of Topic objects
    :param question: User's question
    :return: Answer object or None if no answer could be generated
    """
    topics_md = "\n".join([f"Topic id: {topic.topic_id}, Title: {topic.title}, Summary: {topic.summary}, Keywords: {topic.keywords}, Start time: {topic.start_time}, End time: {topic.end_time}" for topic in topics])
    
    answer_generator = client.chat.completions.create_iterable(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are an AI assistant that uses the youtube video transcript and its main topics to answer user questions.
                Please provide the user question and the answer with the start and end times of the video that you used to answer the question.
                """
            },
            {
                "role": "assistant",
                "content": f"[Transcript from the video]:\n{transcript}\n[Topics from the transcript]:\n{topics_md}"
            },
            {
                "role": "user",
                "content": f"Please provide the user question and the answer with the start and end times of the video that you used to answer the question.\n[Question]:\n {question}"
            }
        ],
        response_model=Answer,
        stream=True
    )
    
    console.print("[bold green]Answer:[/bold green]")
    for partial_answer in answer_generator:
        console.print(f"[bold green]Question:[/bold green] {partial_answer.question}")
        console.print(f"[bold green]Answer:[/bold green] {partial_answer.answer}")
        console.print(f"[bold green]Start time:[/bold green] {partial_answer.start_time}")
        console.print(f"[bold green]End time:[/bold green] {partial_answer.end_time}")
        return partial_answer
    
    return None

def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """
    Estimate the cost of API usage based on the model and token counts.
    
    :param model: The GPT model being used
    :param input_tokens: Number of input tokens
    :param output_tokens: Number of output tokens (default 0)
    :return: Estimated cost in USD
    """
    if model not in PRICING:
        raise ValueError(f"Pricing information not available for model: {model}")
    
    model_pricing = PRICING[model]
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    
    return input_cost + output_cost

def main():
    """Main function to run the YouTube Chat Bot."""
    console.print("[bold]YouTube Chat Bot[/bold]")
    
    input_value = Prompt.ask("Enter a CSV filename or YouTube URL")
    
    try:
        with console.status("[bold green]Fetching transcript..."):
            transcript, video_id, token_count = get_transcript(input_value)
        
        if not transcript:
            raise ValueError("No transcript found")
        
        console.print(f"[bold blue]Transcript token count: {token_count}[/bold blue]")
        
        # Estimate cost for processing the transcript
        estimated_cost = estimate_cost(GPT_MODEL, token_count)
        console.print(f"[bold yellow]Estimated cost for processing transcript: ${estimated_cost:.4f}[/bold yellow]")
        
        # Ask user for confirmation
        proceed = Prompt.ask("Do you want to proceed with processing?", choices=["yes", "no"])
        
        if proceed.lower() != "yes":
            console.print("[bold red]Processing cancelled by user.[/bold red]")
            return
        
        with console.status("[bold green]Generating topics..."):
            topics = display_topics(transcript)
        
        if not topics:
            raise ValueError("Failed to generate topics")
        
        while True:
            action = Prompt.ask("What would you like to do?", choices=["question", "exit"])
            
            if action == "exit":
                break
            elif action == "question":
                question = Prompt.ask("What's your question about the video?")
                with console.status("[bold green]Analyzing question and generating answer..."):
                    answer = answer_question(transcript, topics, question)
                
                if answer:
                    video_url = f"https://www.youtube.com/watch?v={video_id}&t={int(answer.start_time)}s"
                    console.print(f"\n[bold green]Here's the link to the video at the relevant part:[/bold green]")
                    console.print(f"[link={video_url}]{video_url}[/link]")
                else:
                    console.print("[bold red]Failed to generate an answer.[/bold red]")
    
    except Exception as e:
        console.print(f"[bold red]An error occurred: {str(e)}[/bold red]")

if __name__ == "__main__":
    main()


# transcript_OzNuAg2bx6k.csv




