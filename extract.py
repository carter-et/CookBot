"""Script to grab body from website and write it into a file"""
import os
import sys
import requests
from bs4 import BeautifulSoup


def extract():
    """
    Use url from command line to scrape text from a website and write its body into a text file
    """
    # read input
    url = read_input()
    # get current working directory
    cwd = os.getcwd()
    # get the next file number
    next_file = find_next_available_file(cwd)
    # get <body> data </body> from url
    data = connect_to_url(url)
    # write it to a file
    write_file(data, next_file, name="aly")


def read_input():
    """
    Grabs input from command line

    Recipe URL should be first argument.

    Returns:
        str: url from command line
    """
    if len(sys.argv) < 2:
        print('Did you put in a url? Usage is: python extract.py <url>')
        sys.exit(1)
    elif len(sys.argv) > 2:
        print('Received too many arguments! Usage is: python extract.py <url>')
        sys.exit(1)
    url = sys.argv[1]
    return url


def connect_to_url(url):
    """
    Get response from URL GET and return the body, tags and all.

    Args:
        url: a basic html url (preferably a recipe)

    Returns:
        str: block of text from URL website body including HTML tags
    """
    response = requests.get(url)
    if not response.ok:
        print('That was a bad url! Try it in your browser first please.')
        sys.exit(2)

    soup = BeautifulSoup(response.text, 'html.parser')
    soup.prettify()
    data = soup.find("body")
    return data


def find_next_available_file(cwd):
    """Determine the next filename for writing text into

    Args:
        cwd: the current working directory

    Returns:
        int: the next number to use for filenames
    """
    directory = './ExtractOutputs'
    next_file_number = len([name for name in os.listdir(directory) \
        if os.path.isfile(os.path.join(directory, name))]) + 1
    return next_file_number


def write_file(data, next_file, name='0'):
    """Write data into a file given a file number to use as a unique identifier

    Args:
        data (str): HTML body to write into a file
        next_file (int): the number to use as a unique identifier for a file
        name (str): identifier for your machine so filenames between machines don't get duplicated
    """
    # open a file for writing
    file = open(f"./ExtractOutputs/train/{name}-output{next_file}.txt", "w+")
    open(f"./ExtractOutputs/results/{name}-output{next_file}.txt", "w+")

    #write the data
    file.write(str(data))

    #close the file
    file.close()


if __name__ == "__main__":
    extract()
