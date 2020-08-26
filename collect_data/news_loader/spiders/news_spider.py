import re

import scrapy
from itemloaders.processors import MapCompose, Join
from scrapy.loader import ItemLoader


class NewsSpider(scrapy.Spider):
    name = 'news_spider'
    site_url = 'https://ria.ru/'
    themes = [ 'science', 'culture','religion', 'society','world', 'economy', 'politics',]

    def create_date(self, start_year=2018, end_year=2019):
        assert start_year <= end_year
        for year in range(start_year, end_year):
            for month in range(1, 13):
                for day in range(1, 32):
                    date = "{}{:0>2d}{:0>2d}".format(year, month, day)
                    yield date + '/'


    def start_requests(self):
        for theme in self.themes:
            for date in self.create_date():
                url = self.site_url+ theme + '/' + date
                yield scrapy.Request(url = url, callback=self.parse_links, meta={'theme':theme})

    def parse_links(self, response):
        links = response.xpath('//div[@class = "list-item__content"]/a/@href').extract()

        for link in links:
            yield response.follow(link, self.parse_item)

    def parse_item(self, response):
        yield {
            'theme': response.xpath('//meta[@property="article:tag"]/@content').extract(),
            'title': response.xpath('//div[@class="article__header"]/h1[@class="article__title"]/text()').extract_first(),
            'text':response.xpath('//div[@class="article__text"]').extract(),
        }