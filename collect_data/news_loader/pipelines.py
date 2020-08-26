
import csv
from scrapy.exporters import CsvItemExporter
import re


def clean_text(s):
    return ' '.join(map(lambda s: re.sub(r'</?\w+ {,1}\S*>', '', s.strip()), s))
#TODO: \xa /n

class NewsLoaderPipeline:
    themes = ['Общество', 'Религия','Экономика','Культура','В мире','Политика','Наука']
    def open_spider(self, spider):
        self.file = open('news.csv', 'wb')#, encoding='utf8')
        self.exporter = CsvItemExporter(self.file, 'utf-8')
        self.exporter.start_exporting()

       # self._fields = ["theme", "title", "text"]
      #  self.file.write(';'.join(self._fields))

    def close_spider(self, spider):
        self.exporter.finish_exporting()
        self.file.close()

    def process_item(self, item, spider):
        item['theme'] = [tag for tag in item['theme'] if tag in self.themes]
        #item['title'] = item['title'][0]
        item['text'] = clean_text(item['text'])
        self.exporter.export_item(item)
        return item
