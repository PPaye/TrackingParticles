from bs4 import BeautifulSoup
import requests
import os
import urllib.request
import wget
import subprocess
url = 'http://test-timing-challenge.web.cern.ch/test-Timing-Challenge/'
urlTest = 'http://test-timing-challenge.web.cern.ch/test-Timing-Challenge/Data_Test'
#urlTest = 'http://test-timing-challenge.web.cern.ch/test-Timing-Challenge/Data_Test_Unsmeared/'
urlTrain = 'http://test-timing-challenge.web.cern.ch/test-Timing-Challenge/Data_Train'
#urlTrain = 'http://test-timing-challenge.web.cern.ch/test-Timing-Challenge/Data_Train_Unsmeared/'
ext = '.txt'

#code to list possible files in URL 
def listFD(url, ext=''):
    page = requests.get(url).text
    #print(page)
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

"""
for file in listFD(url, ext):
    ind = file.rfind('-',0,len(file))
    filename = file[ind+1:]
    print(filename)
"""
def getFiles(myUrl):
    correctFilenames = []
    files = listFD(myUrl,ext)
    print("here")
    print(files)
    for file in files:
        ind = file.rfind('-',0,len(file))                                                                                                                                                                         
        filename = file[ind+1:]  
        correctFilenames.append(filename)

    for i in range(len(files)):
        if 'Test' in myUrl:
            correctName = 'DataTest/' + correctFilenames[i]
        elif 'Train' in myUrl:
            correctName = 'DataTrain/' + correctFilenames[i]
        print("correct name")
        print(correctName)
        #exists = os.path.isfile(correctName)
        if False:
            print("exists")
        else:
            print("file to download")
            print(files[i])
            #wget.download(files[i],correctName)
            wget.download(files[i])
            continue

getFiles(urlTest)
getFiles(urlTrain)
print("done")
    
    
