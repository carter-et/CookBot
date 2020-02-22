import requests
import sys
from bs4 import BeautifulSoup
import os


def extract():
    # read input
    url = read_input()
    # get current working directory
    cwd = os.getcwd()
    # get the next file number
    next_file = find_next_available_file(cwd)
    # get <body> data </body> from url
    data = connect_to_url(url)
    # write it to a file
    write_file(data, next_file)
    return


def read_input():
    url = sys.argv[1]
    return url


def connect_to_url(url):
    response = requests.get(url)
    page = str(BeautifulSoup(response.content))

    start_link = page.find("body")
    if start_link == -1: return None, 0

    return data


def find_next_available_file(cwd):
    DIR = '/ExtractOutputs'
    next_file_number = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) + 1
    return next_file_number


def write_file(data, next_file):
    #open a file for writing
    file = open("output"+next_file+".txt", "w+")

    #write the data
    for
    return


if __name__ == "__main__":
    extract()
