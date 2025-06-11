# -*- coding: utf-8 -*-
'''
#Author: Yalei Meng  yaleimeng@sina.com
#License: Created with VS Code, (C) Copyright 2025
#Description: 提供了百度、搜狗和360搜索的访问功能。
#Date: 
LastEditTime: 2025-04-15 13:30:34
#FilePath: Do not edit
'''
import requests as rq
from bs4 import BeautifulSoup as bs
from baidusearch.baidusearch import search


def html_in(page, load=None, soup=True,):
    head = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1", }
    if not load:
        r = rq.get(page, headers=head, timeout=9)
    else:
        r = rq.get(page, headers=head, timeout=9, params=load)
    # r.encoding = detect(r.content)['encoding']
    return bs(r.text, 'lxml') if soup else r.text


def search_sogou(query):
    '''测试成功。URL为跳转链接,默认9条结果'''
    alink = f'https://www.sogou.com/web?query={query}'
    soup = html_in(alink)
    # print(soup)
    piles = soup.select('div.vrwrap')
    data = []
    for pile in piles:
        if not pile.h3 or not pile.h3.a:
            continue
        
        # 更精确地提取摘要文本
        snippets_text = ""
        
        # 尝试从不同的CSS选择器中获取摘要
        snippet_selectors = [
            'div.str_info',  # 搜狗的摘要通常在这个div中
            'div.ft',
            'p.str_info',
            'div.rb'
        ]
        
        for selector in snippet_selectors:
            snippet_elem = pile.select_one(selector)
            if snippet_elem:
                snippets_text = snippet_elem.get_text(strip=True)
                break
        
        # 如果上述选择器都没找到，使用整个pile的文本但排除标题
        if not snippets_text:
            full_text = pile.get_text(strip=True)
            title_text = pile.h3.get_text(strip=True)
            # 移除标题部分，获取剩余文本作为摘要
            if title_text in full_text:
                snippets_text = full_text.replace(title_text, '', 1).strip()
            else:
                snippets_text = full_text
        
        # 清理文本：移除多余的空白字符和特殊字符
        snippets_text = snippets_text.replace('\xa0', ' ')  # 替换不间断空格
        snippets_text = snippets_text.replace('\u3000', ' ')  # 替换全角空格
        snippets_text = ' '.join(snippets_text.split())  # 规范化空白字符
        
        # 限制摘要长度，避免过长的文本
        if len(snippets_text) > 200:
            snippets_text = snippets_text[:200] + "..."
        
        one = {'title': pile.h3.text.strip(), 
               'snippets': snippets_text,
               'url': f"https://www.sogou.com{pile.h3.a['href']}", }
        data.append(one)
    return data


def search_360(query):
    '''测试成功。URL为跳转链接，默认10条结果'''
    soup = html_in(f'https://www.so.com/s?q={query}')
    piles = soup.select('li.res-list')
    data = []
    for pile in piles:
        if not pile.h3:
            continue
            
        # 更精确地提取摘要文本
        snippets_text = ""
        
        # 尝试从摘要相关的选择器中获取文本
        snippet_selectors = [
            'div.res-rich',
            'div.abs',
            'p.res-desc'
        ]
        
        for selector in snippet_selectors:
            snippet_elem = pile.select_one(selector)
            if snippet_elem:
                snippets_text = snippet_elem.get_text(strip=True)
                break
        
        # 如果没找到，使用整个pile的文本但排除标题
        if not snippets_text:
            full_text = pile.get_text(strip=True)
            title_text = pile.h3.get_text(strip=True)
            if title_text in full_text:
                snippets_text = full_text.replace(title_text, '', 1).strip()
            else:
                snippets_text = full_text
        
        # 清理文本
        snippets_text = snippets_text.replace('\xa0', ' ')
        snippets_text = snippets_text.replace('\u3000', ' ')
        snippets_text = ' '.join(snippets_text.split())
        
        # 限制长度
        if len(snippets_text) > 200:
            snippets_text = snippets_text[:200] + "..."
            
        one = {'title': pile.h3.text.strip(), 
               'snippets': snippets_text,
               'url': pile.h3.a['href']}
        data.append(one)
    return data


def search_baidu(query):
    '''测试成功。URL为跳转链接，默认10条结果'''
    results = search(query, num_results=10)
    data = []
    for pile in results:
        # 清理百度返回的摘要文本
        snippets_text = pile['abstract'].replace('\n', '')
        snippets_text = snippets_text.replace('\xa0', ' ')
        snippets_text = snippets_text.replace('\u3000', ' ')
        snippets_text = ' '.join(snippets_text.split())
        
        # 限制长度
        if len(snippets_text) > 200:
            snippets_text = snippets_text[:200] + "..."
            
        one = {'url': pile['url'], 
               'title': pile['title'].strip(), 
               'snippets': snippets_text}
        data.append(one)
    return data


if __name__ == '__main__':
    chaxun = '2025五一放假安排'
    out1 = search_baidu(chaxun)
    # out1 =search_sogou(query)
    # out1 = search_360(query)
    print(out1)
    print(len(out1))