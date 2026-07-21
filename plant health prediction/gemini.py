from google import genai

client = genai.Client(api_key="AIzaSyAY3BECyor9P31cqlUs7du3mViu0jh683Q")

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Hello bhai"
)

print(response.text)