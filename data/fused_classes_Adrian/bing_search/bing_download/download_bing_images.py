#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
wrapper to get images from the Bing API v7

Example
-------

python3 ROOT_DIR/download_bing_crt_page.py ROOT_DIR/config/download_bing_crt_page.conf

"""

import os,sys
from os.path import isfile, isdir, join
from os import listdir
from shutil import copyfile
from shutil import rmtree
import time

from configparser import ConfigParser


def get_bing_results(queries_list,api_key,per_page,meta_dir,image_dir,max_pages):
	user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36'
	
	if not os.path.isdir(meta_dir):
		os.mkdir(meta_dir)
	if not os.path.isdir(image_dir):
		os.mkdir(image_dir)
	
	queries = []
	f_queries = open(queries_list)
	
	for qline in f_queries:
		qline = qline.rstrip()
		queries.append(qline)

	f_queries.close()
	unique_urls = {}
	for pos in range(0,len(queries)):
		crtq = queries[pos]
		qparts = crtq.split(";")
		name = qparts[1].replace("'","%27").replace(" ","+")
		print('name: ', name)
		labelid = qparts[0]
		#get the images for the current ID into a dedicated subdir
		id_image_dir = os.path.join(image_dir,labelid)
		if not os.path.exists(id_image_dir):
			os.mkdir(id_image_dir)
		url_cnt = 0
		#make sure that there are no duplicate URLs
		for page in range(1, max_pages+1):
			#open the previous page to get the offset estimated by bing
			crt_page_id = os.path.join(meta_dir,labelid+"_page_"+str(page)+".json")
			if not os.path.exists(crt_page_id):
				if page == 1:
					crt_page_command = 'curl -X GET "https://westeurope.api.cognitive.microsoft.com/bing/v7.0/images/search?q=\"'+name+'\"&count='+per_page+'&offset=0 " -H "Ocp-Apim-Subscription-Key: '+api_key+' " | json_pp> '+crt_page_id
					#crt_page_command = "curl -X GET 'https://api.cognitive.microsoft.com/bing/v7.0/images/search?q=\""+name+"\"&count="+per_page+"&offset=0 ' -H 'Ocp-Apim-Subscription-Key: "+api_key+" ' | json_pp > "+crt_page_id
				else:
					#open the previous page to get the current offset - useful to remove image duplicates from the results set
					prev = page-1
					prev_path = os.path.join(meta_dir,labelid+"_page_"+str(prev)+".json")
					#crt_offset = prev * 150
					crt_offset = prev * per_page
					f_prev = open(prev_path,encoding="utf8", errors='ignore')
					for pline in f_prev:
						if "nextOffset" in pline:
							pline_parts = pline.rstrip().split(' : ')
							pline_parts[-1] = pline_parts[-1].replace(",","")
							crt_offset = int(pline_parts[-1])
					f_prev.close()
					crt_page_id = os.path.join(meta_dir,labelid+"_page_"+str(page)+".json")
					#crt_page_command = "curl -X GET 'https://api.cognitive.microsoft.com/bing/v7.0/images/search?q=\""+name+"\"&count=100&offset="+str(crt_offset)+" ' -H 'Ocp-Apim-Subscription-Key: "+api_key+" ' | json_pp > "+crt_page_id
					crt_page_command = 'curl -X GET "https://westeurope.api.cognitive.microsoft.com/bing/v7.0/images/search?q=\"'+name+'\"&count='+per_page+'&offset='+str(crt_offset)+' " -H "Ocp-Apim-Subscription-Key: '+api_key+' " | json_pp > '+crt_page_id
				os.system(crt_page_command)
				#sleep 5 second every three queries to comply with Bing API limitations for free accounts
				if pos%3 == 0:
					time.sleep(5)
			else:
				print("exists first page meta:"+crt_page_id)
			
			#open the metadata dir and get the URLs of the images
			f_meta = open(crt_page_id)
			for mline in f_meta:
				mline = mline.rstrip()
				if 'contentUrl' in mline:
					#print(mline)
					mparts = mline.split('"')
					#print(mparts[3])
					img_url = mparts[3]
					if '?' in mparts[3]:
						urparts = mparts[3].split('?')
						img_url = urparts[0]	
					img_url_lower = img_url.lower()
					if not img_url in unique_urls and not '.gif' in img_url_lower:
						out_path = os.path.join(id_image_dir,"image_"+str(url_cnt)+".jpg")
						if not os.path.exists(out_path):
							#download each image in a tmp file and then convert it to jpg
							tmp_path = os.path.join(id_image_dir,"tmp")
							url_command = "wget -q -t 10 -T 10 --no-check-certificate --user-agent=\'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36\' -O "+tmp_path+" -U IE5 \'"+img_url+"\'"
							os.system(url_command)
							#only try converting to JPG if the download was successful
							if os.path.exists(tmp_path):
								convert_command = "convert "+tmp_path+" -quality 100 "+out_path
								os.system(convert_command)
								os.remove(tmp_path) #remove the temporary image file
							else:
								print("download failed for:",img_url)
						else:
							print("exists image file:",out_path)
						unique_urls[img_url] =""
						url_cnt = url_cnt+1
			f_meta.close()
			

""" MAIN """
if __name__ == '__main__':

	#loading the configuration file
	cp = ConfigParser()
	cp.read(sys.argv[1])
	cp = cp[os.path.basename(__file__)]

	queries_list = cp['queries_list']
	api_key = cp['api_key']
	per_page = cp['per_page']
	meta_dir = cp['meta_dir'] 
	image_dir = cp['image_dir'] 	
	max_pages = int(cp['max_pages'])

	
	get_bing_results(queries_list,api_key,per_page,meta_dir,image_dir,max_pages)
