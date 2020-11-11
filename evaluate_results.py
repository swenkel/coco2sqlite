###############################################################################
# Convert coco annotations and predictions to a SQLite database               #
#                                                                             #
#                                                                             #
# (c) 2020 Simon Wenkel                                                       #
# Released under the Apache 2.0 license                                       #
#                                                                             #
###############################################################################



#
# import libraries
#
import argparse
import json
import sqlite3
import os
import glob
from joblib import Parallel, delayed

from tqdm import tqdm
import numpy as np

import utils

#
# functions
#
def parseARGS():
    parser = argparse.ArgumentParser()
    parser.add_argument("-DB", "--DB", \
                        default="databases/coco_annotations.sqlite3")
    parser.add_argument("-gtf", "--groundtruthfile", \
                        required=True, \
                        type=str, \
                        help="File containing groundtruth annotations")
    parser.add_argument("-gts", "--groundtruthsubset",
                         required=True,
                         type=str,
                         help="Groundtruth subset (e.g. 'val', 'test')")
    parser.add_argument("-rsf", "--resultsfolder", \
                        default="", \
                        type=str, \
                        help="Folder containing results")

    args = parser.parse_args()
    config = {}
    config["DB"] = args.DB
    config["GT_subset"] = args.groundtruthsubset
    config["GT_file"] = args.groundtruthfile
    config["Results_folder"] = args.resultsfolder
    return config



#
# main function
# 

def main():
    config = parseARGS()
    if not os.path.exists(config["DB"]):
        generate_empty_DB = True
    else:
        generate_empty_DB = False
        if not utils.is_SQLiteDB(config["DB"]):
            raise Exception("DB file exists but is not a valid SQLite file.\
                            Please provide a valid SQlite filepath.")
    db_conn = sqlite3.connect(config["DB"])
    db_curs = db_conn.cursor()
    if generate_empty_DB:
        utils.create_DB(db_conn, \
                        db_curs)
    
    
    gt_annotations = utils.load_json(config["GT_file"])
    if config["GT_file"]:
        utils.add_gt_annotations(gt_annotations, \
                                 config["GT_subset"], \
                                 db_conn,
                                 db_curs,
                                 generate_empty_DB)
    
    for result in np.sort(glob.glob(config["Results_folder"]+"*.json")):
        model = result.split("/")[-1].split(".")[0]
        utils.add_predictions_to_db(utils.load_json(result),\
                                    model,\
                                    db_curs,\
                                    db_conn)
        if not utils.check_if_model_processed(model, db_conn, db_curs):
            image_ids = utils.get_image_ids_of_pred(model,\
                                                    db_conn,\
                                                    db_curs)
            image_ids = image_ids.flatten()
            print("Calculate IoU for each prediction of", model)
            for image_id in tqdm(image_ids):
                utils.process_predictions_per_image(int(image_id),\
                                                    config["GT_subset"],\
                                                    model,\
                                                    db_conn,\
                                                    db_curs)
        else:
            print(model, "results already processed.")       
    

    db_conn.close()

if __name__ == "__main__":
    main()