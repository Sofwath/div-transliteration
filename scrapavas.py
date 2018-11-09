
from time import sleep
from random import randint
from selenium import webdriver
from pyvirtualdisplay import Display
import pandas as pd

class AvasSpider():

	def __init__(self):
		self.url_to_crawl = "https://avas.com/"
		self.file = open("avas.csv","w") 


	def start_driver(self):
		print('starting driver...')
		self.display = Display(visible=0, size=(800, 600))
		self.display.start()
		self.driver = webdriver.Chrome("/usr/local/bin/chromedriver")
		chrome_options = webdriver.ChromeOptions()
		prefs = {"profile.managed_default_content_settings.images": 2}
		chrome_options.add_experimental_option("prefs", prefs)
		self.driver = webdriver.Chrome(chrome_options=chrome_options)
		sleep(4)

	def close_driver(self):
		print('closing driver...')
		self.display.stop()
		self.driver.quit()
		print('closed!')

	def get_page(self,url):
		print('getting page...')
		self.driver.get(url)

	def grab_list_items(self):

		
		latintitle = self.driver.title
		title = self.driver.find_elements_by_xpath("/html/body[@class='theme-default font-sans antialiased ']/div[@id='app']/div[@class='rtl container mx-auto mb-7 mt-8 px-4 md:px-0']/h1[@class='font-waheed font-normal text-5xl leading-normal mb-2 text-default']")
		try: 
			print (title[0].text)
			print (latintitle)
			scraped_info = title[0].text + "\t" + latintitle + "\n"
			#print (scraped_info)
			self.file.write(scraped_info)
			print ("...... saved")
		except:
			print ("...... err")
		
		

	def parse(self):
		self.start_driver()
		mainurl="https://avas.mv/"
		cats = [""]
		i = 0
		while i < len(cats):
		    for r in range(230,56171):
		        link = (mainurl+str(r))
		        print (link)
		        self.get_page(link)
		        all_items = self.grab_list_items()
		    i += 1
		
		self.close_driver()

Mihaaru = AvasSpider()
Mihaaru.parse()