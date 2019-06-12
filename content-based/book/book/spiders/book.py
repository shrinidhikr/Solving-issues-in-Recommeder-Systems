# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 10:51:05 2019

@author: sahil
"""

# -*-coding: utf-8 -*-
"""
Created on Sun Nov 25 10:28:29 2018

@author: mohammed
"""

import logging
import re
from scrapy.utils.log import configure_logging  
from scrapy.contrib.spiders import CrawlSpider, Rule
from book.items import NewItem
from scrapy.contrib.linkextractors import LinkExtractor
from scrapy.contrib.linkextractors import IGNORED_EXTENSIONS
from scrapy.http import Request
import urllib.parse


class Subpages(CrawlSpider):
    name = "book"
    
    allowed_domains = ["flipkart.com"]
    start_urls = [
            "https://www.flipkart.com/search?q=novels&as=on&as-show=on&otracker=AS_Query_HistoryAutoSuggest_0_6&otracker1=AS_Query_HistoryAutoSuggest_0_6&as-pos=0&as-type=HISTORY&as-backfill=on"
    ]
    
    configure_logging(install_root_handler=False)
    logging.basicConfig(
        filename='log.txt',
        format='%(levelname)s: %(message)s',
        level=logging.INFO
    )
    #which crawls the all subpages of website and its pages until it does not find <a> tag 
    rules = (
            Rule(LinkExtractor(allow=(r''),restrict_xpaths=('//a[@class="_2Xp0TH"]'),),follow=True,),
            Rule(LinkExtractor(allow=(r''),restrict_xpaths=('//div[@class="_3liAhj _1R0K0g"]//a[@class="_2cLu-l"]'),),follow=True,callback='parse_items_',),
            )
       
    def parse_items_(self, response):
        self.log('Hi, this is an item page! %s' % response.url)
      
        item = NewItem()
        trans_table = {ord(c): None for c in u'\r\n\t'}
        #item['page'] = response.url
        item['title'] =  response.xpath('//span[@class="_35KyD6"]/text()').extract()
        #item['price'] =  response.xpath('//div[@class="_1vC4OE _3qQ9m1"]/text()').extract()
        #item['rating'] =  response.xpath('//div[@class="_3ors59"]//div[@class="niH0FQ _2nc08B"]//span[@class="_2_KrJI"]//div[@class="hGSR34"]/text()').extract()
        x = str( ' '.join(s.strip().translate(trans_table) for s in response.xpath('//div[@class="_1HmYoV _35HD7C"]//div[@class="bhgxx2 col-12-12"]//div[@class="_3cpW1u"]').extract()))
        cleanr = re.compile('<.*?>')                                                                                       
        cleantext = re.sub(cleanr, '', x)
        item['Description'] = cleantext
        item['Author'] = response.xpath('//a[@class="_3la3Fn _1zZOAc oZoRPi"]//text()').extract()
        gener =  (str(response.xpath('//div[@class="_3WHvuP"]//ul//li[@class="_2-riNZ"][4]/text()').extract())).split(": ")
        item['Gener'] = gener[1] 
        yield item
    
         
         
    custom_settings = {
     'FEED_URI': 'book1.csv',
     'FEED_FORMAT': 'csv',
     'FEED_EXPORT_ENCODING': 'utf-8'
              } 




  