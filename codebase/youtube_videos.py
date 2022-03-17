from __future__ import unicode_literals
import youtube_dl

ydl_opts = {}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    meta = ydl.extract_info(
        'https://www.youtube.com/watch?v=9bZkp7q19f0', download=False)

print(meta['upload_date'])
print(meta['uploader'])
print(meta['view_count'])
print(meta['like_count'])
try:
    print(meta['dislike_count'])
except :
    print("None")

print(meta['id'])
print(meta['format'])


