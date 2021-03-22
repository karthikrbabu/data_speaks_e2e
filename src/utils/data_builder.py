import csv
import os
import re
from csv import reader, writer
import random

# open file in read mode

def remove_tags():
    with open(os.path.abspath(os.path.join(os.getcwd(), "../../data/e2e-dataset/testset_w_refs.csv")), 'r') as read_obj:
        with open(os.path.abspath(os.path.join(os.getcwd(), "../../data/e2e-dataset/testset_w_refs_notags.csv")), mode='w') as write_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            headers = next(csv_reader)
            csv_writer = writer(write_obj,delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
            csv_writer.writerow(headers)
            # Iterate over each row in the csv using reader object
            print("Line Number", csv_reader.line_num)
            print("first record", csv_reader.line_num)
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                mr = row[0].split(",")
                umr = ""
                for word in mr:
                    word = re.search('\[(.+?)\]', word)
                    if word:
                        word = word.group(1).strip()
                        umr = umr + "," + word
                print("\"",umr[1:],"\"",",","\"",row[1],"\"")
                csv_writer.writerow([umr[1:],row[1]])

def random_features():
    with open(os.path.abspath(os.path.join(os.getcwd(), "../../data/e2e-dataset/testset_w_refs.csv")), 'r') as read_obj:
        with open(os.path.abspath(os.path.join(os.getcwd(), "../../data/e2e-dataset/testset_w_refs_random.csv")), mode='w') as write_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            headers = next(csv_reader)
            csv_writer = writer(write_obj, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
            csv_writer.writerow(headers)
            # Iterate over each row in the csv using reader object
            print("Line Number", csv_reader.line_num)
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                mr = row[0].split(",")
                random.shuffle(mr)
                umr = ""
                for word in mr:
                    umr = umr + "," + word
                print("\"",umr[1:],"\"",",","\"",row[1],"\"")
                csv_writer.writerow([umr[1:],row[1]])

remove_tags()
random_features()

