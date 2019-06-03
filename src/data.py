import scipy.io
import os
import requests
import zipfile


DATA_URL = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'
DOWNLOAD_PATH = "data"


def download_data(url=DATA_URL, download_dir=DOWNLOAD_PATH):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    r = requests.get(url, allow_redirects=True)
    open(os.path.join(download_dir, "data.zip"), 'wb').write(r.content)


def extract_data(download_dir=DOWNLOAD_PATH):
    zip_ref = zipfile.ZipFile(os.path.join(download_dir, "data.zip"), 'r')
    zip_ref.extractall(os.path.join(download_dir, "data"))
    zip_ref.close()


def read_data(download_dir=DOWNLOAD_PATH):
    return (scipy.io.loadmat(os.path.join(download_dir,
                                          "data",
                                          "matlab",
                                          "emnist-balanced.mat")),
            scipy.io.loadmat(os.path.join(download_dir,
                                          "data",
                                          "matlab",
                                          "emnist-byclass.mat")),
            scipy.io.loadmat(os.path.join(download_dir,
                                          "data",
                                          "matlab",
                                          "emnist-bymerge.mat")),
            scipy.io.loadmat(os.path.join(download_dir,
                                          "data",
                                          "matlab",
                                          "emnist-digits.mat")),
            scipy.io.loadmat(os.path.join(download_dir,
                                          "data",
                                          "matlab",
                                          "emnist-letters.mat")),
            scipy.io.loadmat(os.path.join(download_dir,
                                          "data",
                                          "matlab",
                                          "emnist-mnist.mat")),
            )


def get_data(url=DATA_URL, download_dir=DOWNLOAD_PATH):
    download_data(url=url, download_dir=download_dir)
    extract_data(download_dir=download_dir)
    read_data(download_dir=download_dir)
