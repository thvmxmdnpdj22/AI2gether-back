import streamlit.components.v1 as components
import urllib.parse

def get_google_map_embed_url(place):
    base_url = "https://www.google.com/maps/embed/v1/search"
    query = urllib.parse.quote_plus(place)
    api_key = "YOUR_GOOGLE_MAPS_API_KEY"  # 반드시 환경변수로 관리 추천
    return f"{base_url}?key={api_key}&q={query}"

def show_map(place_name):
    url = get_google_map_embed_url(place_name)
    iframe = f'<iframe width="100%" height="350" frameborder="0" style="border:0" src="{url}" allowfullscreen></iframe>'
    components.html(iframe, height=370)
