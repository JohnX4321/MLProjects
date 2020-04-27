import requests, string
from lxml import html
from googlesearch import search
from bs4 import BeautifulSoup


def chatbot_query(query, index=0):
    fallback = 'Cannot decode reply for this query.'
    res = ''
    try:
        search_res_list = list(search(query, tld="co.in", num=10, stop=3, pause=1))
        page = requests.get(search_res_list[index])
        tree = html.fromstring(page.content)
        soup = BeautifulSoup(page.content, features="lxml")
        article_txt = ''
        article = soup.findAll('p')
        for ele in article:
            article_txt += '\n' + ''.join(ele.findAll(text=True))
        article_txt = article_txt.replace('\n', '')
        first_sent = article_txt.split('.')
        first_sent = first_sent[0].split('?')[0]

        chars_wa_space = first_sent.translate({ord(c): None for c in string.whitespace})
        if len(chars_wa_space) > 0:
            res = first_sent
        else:
            res = fallback
        return res
    except:
        if len(res) == 0: res = fallback
        return res
