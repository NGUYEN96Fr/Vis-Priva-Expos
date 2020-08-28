#!/usr/bin/python

#IMPORTS
import sys
import os
import urllib
import re
import random
import time
import string

# ARGUMENTS:
list_file = sys.argv[1] # input file with the list of names in flat format
result_dir = sys.argv[2] # input directory that stores Flickr API metadata in XML format - one directory per POI
list_dir = sys.argv[3] # output directory that store lists of images for POIs
image_dir = sys.argv[4] # output directory to store images associated to Flickr queries
max_download = int(sys.argv[5]) # maximum number of images downloaded per POI
min_pos = int(sys.argv[6]) # min position in the list for which results are downloaded
max_pos = int(sys.argv[7]) # max position in the list for which results are downloaded

''' USAGE:
python flickr_image_download.py files/flickr_slist.txt \
	/scratch_global/vankhoa/flickr_download/flickr_metadata \
	 /scratch_global/vankhoa/flickr_download/flickr_image_lists\
	  /scratch_global/vankhoa/flickr_download/flickr_images\
	   3000 0 300
'''

#create the out dir if it does not exist


def create_poi_lists(list_file, result_dir, list_dir, min_pos, max_pos):
	'''goes through the list of POIs and create download lists for them
		images are sorted so as to maximize the number of different users that are taken into account
		this will be useful in order to create a socially shared representation of the POI 
	'''
	if(not(os.path.isdir(list_dir))):
		os.mkdir(list_dir)
	cnt = 0
	flist = open(list_file, "r")
	for line in flist:
		line_parts = line.rstrip().split("\t")
		if cnt >= min_pos and cnt < max_pos:
			local_name = line_parts[1]
			list_path = list_dir+"/"+local_name
			# create the list only if it was not previously created
			if not os.path.exists(list_path) or 1 == 1:
				f_list = open(list_path, "w")
				# go through the list of pages for the current ID
				image_dict = {} 
				user_dict = {}
				for page in range(1, 7):				
					xml_file = result_dir+"/"+local_name+"/"+str(page)
					print(xml_file)
					if os.path.exists(xml_file):
						''' dictionaries used to store images and users 
						useful since a subset of images are downloaded so as to diversify users 
						the keys of the first dictionary are created so that images from the different users are put first
						'''
						jf = open(xml_file, "r")
						for meta_line in jf:
							# select the lines with metadata
							if re.search("server", meta_line) != None:
								meta_parts = meta_line.rstrip().split('" ')
								server = ""
								user = ""
								farm = ""
								crt_id = ""
								secret = ""
								for mp in meta_parts:
									subparts = mp.split('="')
									if subparts[0] == 'server':
										server = subparts[1]	
									elif subparts[0] == 'owner':
										user = subparts[1]
									elif subparts[0] == 'farm':
										farm = subparts[1]					
									elif subparts[0] == 'secret':
										secret = subparts[1]
									if re.search( 'photo id', subparts[0]) != None:
										crt_id = subparts[1]
								url = 'https://farm'+str(farm)+'.static.flickr.com/'+str(server)+'/'+str(crt_id)+'_'+str(secret)+'.jpg';
								if not user in user_dict:
									user_dict[user] = 1
								else:
									user_dict[user] = user_dict[user] + 1 
								crt_key = user_dict[user] * 10000 + len(image_dict)
								image_dict[crt_key] = user+" "+str(crt_id)+" "+url
						jf.close()
					else:
						print("missing meta file:",xml_file)	 
				sorted_list = sorted(image_dict, key=lambda i: image_dict[i], reverse=True)
				for sl in sorted(image_dict):
					f_list.write(image_dict[sl]+"\n")
				f_list.close()
			else:
				print("exists list: ",list_path)
		cnt = cnt + 1 

def download_poi_images(list_file, list_dir, image_dir, max_download, min_pos, max_pos):
	'''
		function that downloads the images for the POIs using the lists that were previously created
	'''
	if(not(os.path.isdir(image_dir))):
		os.mkdir(image_dir)
	cnt = 0
	flist = open(list_file, "r")
	for line in flist:
		line_parts = line.rstrip().split("\t")
		local_name = line_parts[1]
		if cnt >= min_pos and cnt < max_pos:
			# create the dir for the POI
			poi_dir = image_dir+"/"+local_name
			if(not(os.path.isdir(poi_dir))):
				os.mkdir(poi_dir)
			list_path = list_dir+"/"+local_name	
			f_list = open(list_path, "r")
			image_cnt = 0
			for image_line in f_list:			
				if image_cnt < max_download:
					image_parts = image_line.rstrip().split(" ")
					image_path = poi_dir+"/"+image_parts[1]+".jpg"
					wget_command = "wget -q -t 10 -T 10 --no-check-certificate --user-agent=\'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36\' -O "+image_path+" -U IE5 \'"+image_parts[2]+"\'"
					os.system(wget_command)	
				image_cnt = image_cnt+1
			f_list.close()
		cnt = cnt + 1
 

""" MAIN """
if __name__ == '__main__':
	create_poi_lists(list_file, result_dir, list_dir, min_pos, max_pos)
	download_poi_images(list_file, list_dir, image_dir, max_download, min_pos, max_pos)
