import os
import math
import openai
import yt_dlp
from pydub import AudioSegment
from dotenv import load_dotenv
import datetime
import csv

# Load environment variables
load_dotenv()

# Function to download audio from YouTube


def download_audio_from_youtube(url):
    ydl_opts = {
        # Prefer WebM audio if available
        'format': 'bestaudio[ext=webm]/bestaudio/best',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Find the downloaded file
    info_dict = ydl.extract_info(url, download=False)
    filename = ydl.prepare_filename(info_dict)
    # Ensure we get the WebM file path
    return filename


# Function to truncate audio into chunks
def truncate_audio(file_path, target_size_mb=25):
    # Get the actual file size
    file_stats = os.stat(file_path)
    file_size_bytes = file_stats.st_size  # Get the file size in bytes

    # Calculate target size in bytes
    target_size_bytes = target_size_mb * 1024 * 1024  # Convert MB to bytes

    # Calculate the number of chunks
    num_chunks = math.ceil(file_size_bytes / target_size_bytes)
    print("-------------chunk numbers-------------", num_chunks)

    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Calculate the duration for each chunk
    total_duration_ms = len(audio)  # Total duration in milliseconds
    chunk_duration_ms = total_duration_ms / num_chunks

    # Split the audio into chunks
    chunk_files = []
    for i in range(num_chunks):
        start_time = i * chunk_duration_ms
        end_time = min((i + 1) * chunk_duration_ms, total_duration_ms)
        chunk = audio[start_time:end_time]

        chunk_file = f"{os.path.splitext(file_path)[0]}_chunk{i}.webm"
        chunk.export(chunk_file, format="webm")
        chunk_files.append(chunk_file)

    return chunk_files

# Function to transcribe audio chunks using OpenAI's Whisper API


def transcribe_audio_chunks(audio_chunks, api_key):
    openai.api_key = api_key
    transcripts = []

    print("Whisper API running... Please wait.")
    total_chunks = len(audio_chunks)

    for i, chunk in enumerate(audio_chunks):
        print(f"Processing chunk {i+1} of {total_chunks}: {chunk}")
        try:
            with open(chunk, 'rb') as audio_file:
                response = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            transcripts.append(response.text)
            print(f"Successfully processed chunk {i+1} of {total_chunks}.")
        except Exception as e:
            print(f"Error processing chunk {i+1} of {total_chunks}: {e}")

    print("Whisper API processing complete.")
    return " ".join(transcripts)

# Function to generate a report using OpenAI's GPT-4o-mini
def generate_report_from_transcript(transcript_path, api_key, model="gpt-4o-mini", temperature=0.1):
    openai.api_key = api_key

    # Read the transcript from the file
    with open(transcript_path, "r") as file:
        transcript_text = file.read()

    # Define the prompt for refining the transcript
    refinement_prompt = f"""
    You are a skilled editor. Please take the following transcript and improve its grammar, punctuation, and clarity. Ensure the sentences are coherent and easy to understand while preserving the original meaning.

    Transcript:
    {transcript_text}
    """

    # Log the prompt
    print("Refining transcript with the following prompt")

    try:
        # Generate the refined transcript
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a skilled editor."},
                {"role": "user", "content": refinement_prompt}
            ],
            temperature=temperature,
            top_p=0.9,        # Ensures the output is more focused
            frequency_penalty=0.2,  # Slightly reduces repetition
            presence_penalty=0.0
        )

        refined_text = response.choices[0].message.content.strip()

        # Log the refined text
        print("Refined Transcript:\n", refined_text)
    except Exception as e:
        print(f"Error during transcript refinement: {e}")
        return None
    
    # Define the prompt for the report generation
    report_prompt = f"""
    You are an experienced financial analyst working with the industry for over 30 years. 
    Based on the following transcript from a financial video, please extract all the tickers mentioned and list out first,
    then and generate a detailed report including:
    1. Key Tickers mentioned and their opinions.
    2. Support and Resistance levels for key stocks.
    3. General Takeaways from Macro events, Sentiment.
    4. Potential Trades for the Next Week.

    Transcript:
    {refined_text}
    """

    # Log the prompt
    print("Generating report with the prompt")

    try:
        # Generate the report
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an experienced financial analyst working with the industry for over 30 years."},
                {"role": "user", "content": report_prompt}
            ],
            temperature=temperature
        )

        report_text = response.choices[0].message.content.strip()

        # Log the generated report text
        print("Generated Report:\n", report_text)
    except Exception as e:
        print(f"Error during report generation: {e}")
        return

    # Generate the prompt for ticker generation
    ticker_prompt = f"""
    Based on the report generate by gpt, please extract all the tickers mentioned and generate a list of tickers to watchlist.
    This list should be able to directly import to TradingView.
    Taking the Format the tickers for TradingView import shown below as an example and no more other words needed,
    the custom section start with ### such as ###Tech Section, and then it should followed section's name such as NASDAQ
    and lastly add the ticker name such as AAPL, this is the template example you should follow:

    ###TECH STOCKS,NASDAQ:AAPL,NASDAQ:GOOGL,NASDAQ:NVDA,NASDAQ:META,###FINANCIALS,NYSE:JPM,NYSE:GS,###CONSUMER GOODS,NYSE:PG,###OTHER STOCKS,NYSE:CVNA,NYSE:PLTR,NASDAQ:PANW,NYSE:ONON,AMEX:SOXL,NASDAQ:APPS,NYSE:UBER,NASDAQ:QQQ,NASDAQ:MSTR,NYSE:PHM,NASDAQ:AFRM,NASDAQ:Z
    Transcript:
    {report_text}
    """

    # Log the prompt for ticker generation
    print("Generating tickers with the following prompt")

    try:
        # Generate the trading view tickers
        ticker_response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an experienced financial analyst working with the industry for over 30 years"},
                {"role": "user", "content": ticker_prompt}
            ],
            temperature=temperature
        )

        tickers_text = ticker_response.choices[0].message.content.strip()

        # Log the generated tickers
        print("Generated Tickers:\n", tickers_text)
    except Exception as e:
        print(f"Error during ticker generation: {e}")
        return

    # Generate a timestamp for identification
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the report and tickers to files
    refined_filename = f"refined_{timestamp}.txt"
    report_filename = f"report_{timestamp}.txt"
    tickers_filename = f"tickers_{timestamp}.txt"

    try:
        with open(refined_filename, "w") as refine_file:
            refine_file.write(refined_text)
        with open(report_filename, "w") as report_file:
            report_file.write(report_text)

        with open(tickers_filename, "w") as tickers_file:
            tickers_file.write(tickers_text)

        print(f"Refined saved as {refined_filename}")
        print(f"Report saved as {report_filename}")
        print(f"Tickers saved as {tickers_filename}")
    except Exception as e:
        print(f"Error saving files: {e}")
    return refined_text, report_text, tickers_text

# Function to save the report as a CSV file
def save_report_as_csv(report_text, timestamp):
    # Define the CSV filename
    csv_filename = f"report_{timestamp}.csv"

    # Define the sections
    sections = [
        "Key Tickers and Opinions",
        "Support and Resistance Levels",
        "General Takeaways from Macro, Sentiment",
        "Potential Trades for the Next Week"
    ]

    # Split the report text into sections
    report_lines = report_text.split('\n')
    current_section = None
    section_data = {section: [] for section in sections}

    for line in report_lines:
        line = line.strip()
        if line in sections:
            current_section = line
        elif current_section and line:
            section_data[current_section].append(line)

    try:
        # Write the data to a CSV file
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for section, data in section_data.items():
                writer.writerow([section])
                for item in data:
                    writer.writerow([item])
                writer.writerow([])  # Add an empty row between sections

        print(f"Report saved as {csv_filename}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

def main():
    api_key = os.getenv('OPENAI_TOKEN')

    url = input("Please enter the YouTube video URL: ")

    # # Step 1: Download audio from YouTube
    audio_file = download_audio_from_youtube(url)
    print("----------------------------------------------------------------")
    print(f"Downloaded audio to {audio_file}")

    # # Step 2: Truncate the audio into smaller chunks
    chunks = truncate_audio(audio_file, target_size_mb=25)
    print(f"Audio has been split into {len(chunks)} chunks.")

    # # Generate a timestamp for identification
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # # Step 3: Transcribe each chunk using Whisper API
    full_transcript = transcribe_audio_chunks(chunks, api_key)
    print("Full Transcript:\n", full_transcript)

    # # Save the transcript to a file with a timestamp
    with open(f"transcript_{timestamp}.txt", "w") as file:
        file.write(full_transcript)

    print(f"Transcript saved as transcript_{timestamp}.txt")

    transcript_path = f"transcript_{timestamp}.txt"
    
    # transcript_path = "transcript_20240825_222102.txt"

    model = "gpt-4o-mini"
    refined_text, report_text, tickers_text = generate_report_from_transcript(transcript_path, api_key, model)

    
if __name__ == "__main__":
    main()
