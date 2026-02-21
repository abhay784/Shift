import os
import sys
import io
import tempfile
import speech_recognition as sr
from groq import Groq
from elevenlabs.client import ElevenLabs
import pygame
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep your responses concise and conversational "
    "since they will be spoken aloud. Avoid markdown, bullet points, or long lists. "
    "Respond naturally as if having a spoken conversation."
)

EXIT_PHRASES = {"quit", "exit", "stop", "goodbye", "bye"}

ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # "George" - a natural male voice


def speak(eleven_client, text):
    print(f"Agent: {text}")
    audio = eleven_client.text_to_speech.convert(
        text=text,
        voice_id=ELEVENLABS_VOICE_ID,
        model_id="eleven_multilingual_v2",
    )
    audio_bytes = b"".join(audio)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    finally:
        pygame.mixer.music.unload()
        os.unlink(tmp_path)


def listen(recognizer, microphone):
    with microphone as source:
        print("Listening...")
        audio = recognizer.listen(source, phrase_time_limit=15)
    try:
        text = recognizer.recognize_google(audio)
        print(f"You: {text}")
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}")
        return None


def get_llm_response(client, history):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=history,
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content


def main():
    groq_key = os.getenv("GROQ_API_KEY")
    eleven_key = os.getenv("ELEVENLABS_API_KEY")

    if not groq_key:
        print("Error: GROQ_API_KEY not set.")
        print("Get a free key at https://console.groq.com")
        sys.exit(1)

    if not eleven_key:
        print("Error: ELEVENLABS_API_KEY not set.")
        print("Get a free key at https://elevenlabs.io")
        sys.exit(1)

    groq_client = Groq(api_key=groq_key)
    eleven_client = ElevenLabs(api_key=eleven_key)
    pygame.mixer.init()
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    speak(eleven_client, "Hello! I'm your voice assistant. How can I help you?")

    try:
        while True:
            user_text = listen(recognizer, microphone)

            if user_text is None:
                speak(eleven_client, "Sorry, I didn't catch that. Could you say it again?")
                continue

            if user_text.lower().strip() in EXIT_PHRASES:
                speak(eleven_client, "Goodbye!")
                break

            history.append({"role": "user", "content": user_text})

            try:
                reply = get_llm_response(groq_client, history)
            except Exception as e:
                print(f"LLM error: {e}")
                speak(eleven_client, "Sorry, I had trouble thinking of a response. Try again.")
                history.pop()
                continue

            history.append({"role": "assistant", "content": reply})
            speak(eleven_client, reply)

    except KeyboardInterrupt:
        print()
        speak(eleven_client, "Goodbye!")


if __name__ == "__main__":
    main()
