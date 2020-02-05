from epub_conversion.utils import convert_epub_to_lines as CETL
from epub_conversion.utils import open_book
import pandas as pd
from bs4 import BeautifulSoup

def textfinder(name):
    # 'name' is the name of the file in question being converted
    # Spits out a text file without html markings
    book = open_book(f"FData/{name}")
    text = CETL(book)
    soup = BeautifulSoup(" ".join(text))
    return soup.get_text()