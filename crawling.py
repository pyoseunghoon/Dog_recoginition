from google_images_download import google_images_download

response=google_images_download.googleimagesdownload()

arguments={"keywords":"miniature pinscher puppies","limit":2000,"print_urls":True, "chromedriver":"chromedriver.exe"}
paths=response.download(arguments)
print(paths)
