from __future__ import unicode_literals
import youtube_dl
import json


def extract_video_information(video_url):
    ydl_opts = {}

    try:

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            video_meta_info = ydl.extract_info(video_url, download=False)

        meta_information = {
            "video_id": "",
            "video_channel_name": "",
            "video_title": "",
            "video_thumbnail": "",
            "video_artist": "",
            "like_count": "",
            "description": "",
            "upload_date": "",
            "uploader": "",
            "video_duration": "",
            "view_count": "",
            "video_rating": "",
            "video_url": "",
            "video_category": "",
            "video_tags": "",
            "video_like_count": "",
            "video_creator": "",
            "video_alt_title": "",
            "video_age_limit": "",
            "video_channel": "",
        }

        try:
            meta_information.update({"video_id": video_meta_info["id"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"video_channel_name": video_meta_info["channel"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"video_title": video_meta_info["title"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"video_thumbnail": video_meta_info["thumbnail"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"video_artist": video_meta_info["artist"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"like_count": video_meta_info["like_count"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"description": video_meta_info["description"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"upload_date": video_meta_info["upload_date"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"uploader": video_meta_info["uploader"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"video_duration": video_meta_info["video_duration"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"view_count": video_meta_info["view_count"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"video_rating": video_meta_info["video_rating"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"video_url": video_meta_info["webpage_url"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"video_category": video_meta_info["categories"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"video_tags": video_meta_info["tags"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"video_like_count": video_meta_info["like_count"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"video_creator": video_meta_info["creator"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"video_alt_title": video_meta_info["alt_title"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"video_age_limit": video_meta_info["age_limit"]})
        except Exception as e:
            print(e)

        try:
            meta_information.update({"video_channel": video_meta_info["channel_url"]})
        except Exception as e:
            print(e)

        return meta_information

    except Exception as e:
        print(e)
