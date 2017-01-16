"""
Created on Thu May 26 11:37:13 2016

@author: ilkhem

Downloads and extracts dumps from AWS S3
"""

import argparse
import gzip
import os
import tarfile

import boto3

from utils import parse_month


def download_and_extract_files(dump_dir, directory, year, month):
    '''
    downloads and extratcs the files that are in the bucket' directory' (under dumps-kaiko/markets/)
    A directory called 'directory' is created locally in hte current working directory, to which the files will
    be downloaded and extracted

    directory is the kind of dumps you want to download : ob_1, ob_10, trades

    Dumps will be downloaded to dump_dir:

    dump_dir/
        ob_1\
            btcusd\
                2016\
                    06\
                        bitfinex\
                            bitfinex_btcusd_2016_06_01.csv
                            bitfinex_btcusd_2016_06_02.csv
                            ...
                        bitstamp\
                            ...
                        ...
                    05\
                        ...
                    ...
                2015\
                    ...
                ...
            btccny\
                ...
            ...
        ob_10\
            btcusd\
                ...
            ...
        trades\
            btcusd\
                ...
            ...
        ...
    :param directory: directory of dumps to download. will also download files to the same directory in current filepath
     Can be ob_1, ob_10 or trades
    :param year: year to dl (string)
    :param month: month to dl (string)
    :return: nothing.
    '''
    if dump_dir[-1] != '/':
        dump_dir += '/'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    os.chdir(dump_dir)

    if not os.path.exists(directory + '/'):
        os.makedirs(directory + '/')
    os.chdir(directory + '/')

    print('downloading and extracting %s for %s, %s' % (directory, year, month))

    conn = boto3.client('s3')  # assumes boto.cfg setup, assumes AWS S3
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('dumps-kaiko')
    prefix = 'markets/' + directory + '/' + year + '/' + month
    for key in conn.list_objects(Bucket='dumps-kaiko', Prefix=prefix)['Contents']:
        link = key['Key']
        print('Downloading', link)
        filename = link.split('/')[-1]
        filepath = filename.split('_')[1].lower() + '/' + year + '/' + month + '/' + filename.split('_')[
            0].lower() + '/'
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        if '.tar' in filename:
            print('Extracting to', dump_dir + directory + '/' + filepath)
            bucket.download_file(link, filepath + filename)
            tar = tarfile.open(filepath + filename)
            tar.extractall(filepath)
            tar.close()
            os.remove(filepath + filename)
            for dirpath, dirnames, files in os.walk(filepath):
                for file in files:
                    try:
                        print(dirpath + '/' + file)
                        f = gzip.open(dirpath + '/' + file, 'rb')
                        newFilename = file.split(".gz")[0]
                        outF = open(filepath + newFilename, 'wb')
                        outF.write(f.read())
                        f.close()
                        outF.close()
                        os.remove(dirpath + '/' + file)
                    except:
                        print("Unable to decompress %s" % dirpath + '/' + file)
        else:
            print('file ignored')
    os.chdir('../')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='downloads and extracts kaiko data dumps from amazon S3. '
                                                 'Skeleton is: dump_dir/pair/year/month/exchange/{.csv per day}')
    parser.add_argument('dump_dir',
                        help='directory under which to store the dumps')
    parser.add_argument('bucket',
                        help='AWS bucket from which to download (ob_1, ob_10 or trades). all will download the 3 of them.')
    parser.add_argument('year',
                        help='year to be downloaded')
    parser.add_argument('month',
                        help='month to be downloaded')

    args_dict = vars(parser.parse_args())

    dump_dir = args_dict['dump_dir']
    bucket = args_dict['bucket']
    year = args_dict['year']
    month = parse_month(args_dict['month'])

    # List of allowed buckets:
    # Hardcoded, needs to be edited in order to add additional dumps
    allowed_buckets = ['trades', 'ob_1', 'ob_10']

    if bucket == 'all':
        for b in allowed_buckets:
            download_and_extract_files(dump_dir, b, year, month)
    elif bucket in allowed_buckets:
        download_and_extract_files(dump_dir, bucket, year, month)
    else:
        print("bucket needs to be one of 'trades', 'ob_1', 'ob_10', 'all' ")
