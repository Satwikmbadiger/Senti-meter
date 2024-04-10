import googleapiclient.discovery
import pandas as pd

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyD1lXAVHpBQbDXaao5C-kTrBBkDbn1tvEI"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

request = youtube.commentThreads().list(
    part="snippet",
    videoId="ij9AcZwMf2I",
    maxResults=10
)

response = request.execute()

comments = []

for item in response['items']:
    comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
    comments.append(comment_text)

df = pd.DataFrame(comments, columns=['Comment'])

print(df)
