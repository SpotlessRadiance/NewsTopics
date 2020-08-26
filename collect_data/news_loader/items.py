# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from itemloaders.processors import TakeFirst, MapCompose, Join
import re



class NewsLoaderItem(scrapy.Item):
    theme = scrapy.Field()
    title = scrapy.Field()
    text = scrapy.Field()

