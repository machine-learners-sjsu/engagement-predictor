from __future__ import unicode_literals
import youtube_dl
import json


def extract_video_information(video_url):
    ydl_opts = {}

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        video_meta_info = ydl.extract_info(video_url, download=False)

    meta_information = {
        "video_id": video_meta_info["id"],
        "video_channel_name": video_meta_info["channel"],
        "video_title": video_meta_info["title"],
        "video_thumbnail": video_meta_info["thumbnail"],
        "video_artist": video_meta_info["artist"],
        "like_count": video_meta_info["like_count"],
        "description": video_meta_info["description"],
        "upload_date":  video_meta_info["upload_date"],
        "uploader": video_meta_info["uploader"],
        "video_duration": video_meta_info["duration"],
        "view_count": video_meta_info["view_count"],
        "video_rating": video_meta_info["average_rating"],
        "video_url": video_meta_info["webpage_url"],
        "video_category": video_meta_info["categories"],
        "video_tags": video_meta_info["tags"],
        "video_like_count": video_meta_info["like_count"],
        "video_creator": video_meta_info["creator"],
        "video_alt_title": video_meta_info["alt_title"],
        "video_age_limit": video_meta_info["age_limit"],
        "video_channel": video_meta_info["channel_url"],
    }

    return meta_information


print(extract_video_information('https://www.youtube.com/watch?v=YQHsXMglC9A'))
