import argparse
import json
import time
import os
import re
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Import your AI agent libraries
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai


def download_youtube_video(url):
    """
    Download the YouTube video from the given URL using yt-dlp.
    Returns the path to the downloaded video file.
    """
    try:
        print(f"Downloading video from {url} using yt-dlp...")
        temp_dir = tempfile.mkdtemp()
        # The output template will save the file with its id and proper extension (e.g., mp4)
        output_template = os.path.join(temp_dir, "%(id)s.%(ext)s")
        command = ["yt-dlp", "-f", "best[ext=mp4]", "-o", output_template, url]
        subprocess.run(command, check=True)
        
        # Look for the downloaded MP4 file in the temporary directory
        for file_name in os.listdir(temp_dir):
            if file_name.endswith(".mp4"):
                file_path = os.path.join(temp_dir, file_name)
                print(f"Downloaded video to {file_path}")
                return file_path
        print("No MP4 file was downloaded.")
        return None
    except Exception as e:
        print(f"Error downloading video from {url}: {e}")
        return None


def parse_analysis(analysis_text):
    """
    Parses the raw analysis text from the agent.
    Standardizes the text to ensure consistent markers and then extracts the summary
    and a list of Q&A pairs (if present).
    """
    # Standardize markers: replace variations like "**Video Summary:**" with "**Summary:**"
    analysis_text = analysis_text.replace("**Video Summary:**", "**Summary:**")
    
    # Define our expected markers
    summary_marker = "**Summary:**"
    qa_marker = "**Questions and Answers:**"
    summary = ""
    qa_list = []
    
    # Extract the summary
    index_summary = analysis_text.find(summary_marker)
    if index_summary != -1:
        summary_start = index_summary + len(summary_marker)
        index_qa = analysis_text.find(qa_marker)
        if index_qa != -1:
            summary = analysis_text[summary_start:index_qa].strip()
        else:
            summary = analysis_text[summary_start:].strip()
    else:
        summary = analysis_text.strip()
    
    # Extract Q&A pairs if the Q&A marker is present
    index_qa = analysis_text.find(qa_marker)
    if index_qa != -1:
        qa_text = analysis_text[index_qa + len(qa_marker):].strip()
        # Regex pattern to capture Q&A pairs (accepts optional numbers after "Question")
        qa_pattern = r"\*\*Question(?:\s*\d+)?\:\*\*\s*(.*?)\s*\*\*Answer:\*\*\s*(.*?)\s*\*\*Context:\*\*\s*\"?(.+?)\"?(?=\n\s*\*\*|$)"
        matches = re.findall(qa_pattern, qa_text, flags=re.DOTALL)
        for question, answer, context in matches:
            qa_list.append({
                "question": question.strip(),
                "answer": answer.strip(),
                "context": context.strip()
            })
    
    return {"summary": summary, "qa": qa_list}


def process_video(video_file_path, agent):
    """
    Processes a video file using the agent:
      - Uploads the file for processing.
      - Waits until processing is complete.
      - Runs the agent with the analysis prompt.
      - Parses and returns the structured analysis.
    """
    print(f"Processing video file: {video_file_path}")
    processed_video = upload_file(video_file_path)
    # Wait until the video processing is complete
    while processed_video.state.name == "PROCESSING":
        time.sleep(1)
        processed_video = get_file(processed_video.name)
    
    analysis_prompt = (
        "Please analyze the provided video and generate a summary followed by a set of questions and answers. "
        "Each question and answer must include a context section that points to the specific part of the summary it was derived from."
    )
    
    print("Running agent analysis on video...")
    response = agent.run(analysis_prompt, videos=[processed_video])
    analysis_text = response.content
    print("Agent analysis complete.")
    print("\n--- Analysis Result ---")
    print(analysis_text)
    
    return parse_analysis(analysis_text)


def main():
    parser = argparse.ArgumentParser(
        description="Process scraped YouTube videos JSON and add video analysis output."
    )
    parser.add_argument("input_json", type=str, help="Path to input JSON file containing scraped YouTube videos.")
    args = parser.parse_args()

    # Load environment variables and configure the API key
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if API_KEY:
        genai.configure(api_key=API_KEY)
    else:
        print("GOOGLE_API_KEY not found. Please set it in your environment or .env file.")
        return

    # Read the input JSON file (which is a list of video objects)
    try:
        with open(args.input_json, "r") as f:
            videos_data = json.load(f)
    except Exception as e:
        print(f"Error reading input JSON file: {e}")
        return

    # Initialize the AI Agent with a system prompt that instructs it to generate a summary and Q&A
    system_prompt = """
You are a video summarizer AI. Your task is to analyze the provided video and perform the following steps:
1. Generate a concise summary of the video's content.
2. Based on the summary, generate a set of relevant questions and their corresponding answers.
3. For each question and answer, include a 'context' section that clearly refers to the part(s) of the summary from which the question was derived.
Ensure that your output is well-structured, clear, and provides actionable insights.
"""
    print("Initializing AI Agent...")
    agent = Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
        system_prompt=system_prompt
    )

    # Process each video in the input JSON list
    output_data = []
    for video_obj in videos_data:
        url = video_obj.get("url")
        if not url:
            print("No URL found for video object; skipping.")
            continue
        print(f"\nProcessing video: {video_obj.get('title', 'Unknown Title')} - {url}")
        video_file_path = download_youtube_video(url)
        if not video_file_path:
            print(f"Skipping video due to download error: {url}")
            video_obj["analysis"] = {"error": "Download failed"}
            output_data.append(video_obj)
            continue
        try:
            analysis = process_video(video_file_path, agent)
            # Append the structured analysis to the original video object under a new key "analysis"
            video_obj["analysis"] = analysis
        except Exception as e:
            print(f"Error processing video {url}: {e}")
            video_obj["analysis"] = {"error": str(e)}
        finally:
            # Clean up the temporary video file
            try:
                os.remove(video_file_path)
            except Exception as e:
                print(f"Error removing temporary video file {video_file_path}: {e}")
        output_data.append(video_obj)

    # Write the updated list (with original content and analysis) to output.json in the current directory
    with open("output.json", "w") as outfile:
        json.dump(output_data, outfile, indent=4)
    print("\nOutput written to output.json")


if __name__ == '__main__':
    main()