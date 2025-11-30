import fitz
import pytesseract
import requests
import re

from PIL import Image
from bs4 import BeautifulSoup

def clean(t): return re.sub(r"\s+"," ", t.replace("\x00","") ).strip()

def load_pdf(path): 
    return clean("\n".join([p.get_text() for p in fitz.open(path)]))

def load_image(path): 
    return clean(pytesseract.image_to_string(Image.open(path)))

def load_web(url):
    h ={"User-Agent":"Mozilla/5.0"}
    soup=BeautifulSoup(requests.get(url,headers=h).text,"html.parser")
    for t in soup(["script","style","nav","footer","header"]): t.decompose()
    
    return clean(soup.get_text())

def chunk(text,size=1200): 
    return [text[i:i+size] for i in range(0,len(text),size)]
