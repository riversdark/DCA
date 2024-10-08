__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/10/08 18:50:39"

"""
Download the multiple sequence alignment for a given Pfam ID
"""

import urllib3
import gzip
import sys
import argparse

parser = argparse.ArgumentParser(description = "Download the full multiple sequence alignment (MSA) in Stockholm format for a Pfam_id.")
parser.add_argument("--Pfam_id", help = "the ID of Pfam; e.g. PF00041")
args = parser.parse_args()
pfam_id = args.Pfam_id


print("Downloading the full multiple sequence alignment for Pfam: {0} ......".format(pfam_id))
http = urllib3.PoolManager()
url = f"https://www.ebi.ac.uk/interpro/wwwapi//entry/pfam/{pfam_id}/?annotation=alignment:full&download"
r = http.request('GET', url)
data = gzip.decompress(r.data)
data = data.decode()
with open("./pfam_msa/{0}_full.txt".format(pfam_id), 'w') as file_handle:
    print(data, file = file_handle)
