from langdetect import detect

# Get language tag from langdetect API
# Input:
# track_url: track lyric path
# Return:
# success: language tag (en, es...)
# fail: None
def get_lang_tag(track_url):
    try:
        # Get lyric string
        lyric_file = open(track_url, encoding='utf-8')
        lyric_str = lyric_file.read()
        lang = detect(lyric_str)
        return lang
    except:
        return None