import os.path
import requests
""" Reused from the image_detection module
"""


def download_file(filename, url):
	"""
	Download an URL to a file
	"""
	with open(filename, 'wb') as fout:
		response = requests.get(url, stream=True)
		response.raise_for_status()
		# Write response data to file
		for block in response.iter_content(4096):
			fout.write(block)

def download_if_not_exists(filename, url):
	"""
	Download a URL to a file if the file
	does not exist already.

	Returns
	-------
	True if the file was downloaded,
	False if it already existed
	"""
	if not os.path.exists(filename):
		print('Downloading', filename, '...')
		download_file(filename, url)
		return True
	print('Skip Download', filename, ', already present.')

	return False
