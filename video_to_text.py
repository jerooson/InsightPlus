import os
import math
import openai
import yt_dlp
from pydub import AudioSegment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Function to download audio from YouTube
def download_audio_from_youtube(url):
    ydl_opts = {
        'format': 'bestaudio[ext=webm]/bestaudio/best',  # Prefer WebM audio if available
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

# Main function to orchestrate the entire process


def main():
    url = 'https://youtu.be/Wq2hyPeIHj4'
    api_key = os.getenv('OPENAI_TOKEN')

    # Step 1: Download audio from YouTube
    audio_file = download_audio_from_youtube(url)
    # audio_file = "OpenAI API Structured Outputs For Finance [jqp7WO3pCFA].webm"
    print("----------------------------------------------------------------")
    print(f"Downloaded audio to {audio_file}")

    # Step 2: Truncate the audio into smaller chunks
    chunks = truncate_audio(audio_file, target_size_mb=25)
    print(f"Audio has been split into {len(chunks)} chunks.")
    # chunks = [audio_file]
    # Step 3: Transcribe each chunk using Whisper API
    full_transcript = transcribe_audio_chunks(chunks, api_key)
    print("Full Transcript:\n", full_transcript)


if __name__ == "__main__":
    main()
