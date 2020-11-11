###############################################################################
# Convert coco annotations to a SQLite database                               #
#                                                                             #
#                                                                             #
# (c) 2020 Simon Wenkel                                                       #
# Released under the Apache 2.0 license                                       #
#                                                                             #
###############################################################################



#
# import  libraries
#
import argparse
import gc
import json
import sqlite3
import time
from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
import numba as nb

from .metrics import bb_iou


#
# functions
#
def is_SQLiteDB(db_file:str)->bool:
    """
    Function to check if a file is a valid SQLite database
    Inputs:
        - file (str) : full path of the file/DB in question
    Ouputs:
        - is_SQLiteDB (bool) : file is a SQLite DB or not
    """
    with open(db_file, "rb") as file:
        header = file.read(100)
    if header[0:16] == b'SQLite format 3\000':
        is_SQLiteDB = True
    else:
        is_SQLiteDB = False
    return is_SQLiteDB



def create_DB(db_conn:sqlite3.Connection,
              db_curs:sqlite3.Cursor):
    """
    Function to generate all tables required in an empty SQLite database

    Inputs:
        - db_conn (sqlite3.connector) : database connection
        - db_curs (sqlite3.cursor) : database cursor to execute commands
    Outputs:
        - None
    """
    db_curs.execute('''CREATE TABLE images
                    (`orig_id` INTEGER,
                     `file_name` TEXT,
                     `coco_url` TEXT,
                     `height` INTEGER,
                     `WIDTH` INTEGER,
                     `date_captured` TEXT,
                     `flickr_url` TEXT,
                     `license` INTEGER,
                     `subset` TEXT)''')
    db_curs.execute('''CREATE TABLE annotations
                    (`segmentation` TEXT,
                     `area` REAL,
                     `iscrowd` INTEGER,
                     `image_id` INTEGER,
                     `bbox` TEXT,
                     `category_id` INTEGER,
                     `orig_id` INTEGER,
                     `subset` TEXT,
                     `isGT` INTEGER)''')
    db_curs.execute('''CREATE TABLE supercategories
                    (`supercategory` TEXT)''')
    db_curs.execute('''CREATE TABLE categories
                    (`category_id` INTEGER,
                     `name` TEXT,
                     `supercategory_id` INTEGER)''')
    db_curs.execute('''CREATE TABLE licenses
                    (`url` TEXT,
                     `license_id` INTEGER,
                     `name` TEXT,
                     `subset` TEXT)''')
    db_curs.execute('''CREATE TABLE predictions
                    (`image_id` INTEGER,
                     `category_id` INTEGER,
                     `bbox` TEXT,
                     `score` REAL,
                     `IoU` REAL,
                     `is_valid_class_in_img` TEXT,
                     `best_match_gt_annotation_id` INTEGER,
                     `model` TEXT,
                     `comments` TEXT )''')
    db_curs.execute('''CREATE TABLE status\
                    (`model` TEXT,
                     `subset` TEXT,
                     `processed` TEXT)''')
    db_conn.commit()
    print("DB generated.")

def check_if_images_in_db(subset:str,
                          total_count:int,
                          db_curs)->bool:
    """

    """
    if subset in db_curs.execute("SELECT DISTINCT subset\
                                  FROM images").fetchall()[0]:
        imgs_in_db = True
        # check if subset is complete, throw exception otherwise
        if db_curs.execute("SELECT COUNT(*)\
                            FROM images\
                            WHERE subset=?", \
                            [subset]).fetchall()[0][0] != total_count:
            raise Exception("Subset of images is in DB but inclomplete!")
    else:
        imgs_in_db = False
    
    return imgs_in_db

def check_if_annotations_in_db(subset:str,
                               total_count:int,
                               db_curs)->bool:
    """

    """
    if subset in np.array(db_curs.execute("SELECT DISTINCT subset\
                                            FROM annotations").fetchall()):
        annot_in_db = True
        # check if subset is complete, throw exception otherwise
        if db_curs.execute("SELECT COUNT(*)\
                            FROM annotations\
                            WHERE subset=?",\
                            [subset]).fetchall()[0][0] != total_count:
            raise Exception("Subset of annotations is in DB but inclomplete!")
    else:
        annot_in_db = False
    
    return annot_in_db



def check_if_predictions_in_db(model:str,
                               total_count:int,
                               db_curs)->bool:
    """

    """
    models = db_curs.execute("SELECT DISTINCT model\
                                  FROM predictions").fetchall()
    if len(models) != 0:
        models = np.array(models)
        if model in models:
            annot_in_db = True
            # check if subset is complete, throw exception otherwise
            if db_curs.execute("SELECT COUNT(*)\
                                FROM predictions\
                                WHERE model=?",\
                                [model]).fetchall()[0][0] != total_count:
                raise Exception(model," predictions is in DB but inclomplete!")
        else:
            annot_in_db = False
    else:
        annot_in_db = False
    
    return annot_in_db


def image_data_to_list(item:dict,
                       subset:str)->list:
        """
        Assuming the structure of each image dict:
                    `orig_id` INTEGER,
                     `file_name` TEXT,
                     `coco_url` TEXT,
                     `height` INTEGER,
                     `WIDTH` INTEGER,
                     `date_captured` TEXT,
                     `flickr_url` TEXT,
                     `license` INTEGER,
                     `subset` TEXT
        Inputs:
            - item (dict) : dict containing all data about an image
            - subset (str) : is the name of the particular subset the image\
                             in question is part of
        Outputs:
            - list_to_move (list) : list containing items as required for \
                                    insert into SQLite table
        """
        list_to_move = [item["id"], \
                        item["file_name"], \
                        item["coco_url"], \
                        item["height"], \
                        item["width"], \
                        item["date_captured"], \
                        item["flickr_url"], \
                        item["license"], \
                        subset]
        return list_to_move
    
def annotations_to_list(item:dict,
                        subset:str,
                        isGT:int)->list:
    """

    Assumed table structure for groundtruth annotations:
                `segmentation' TEXT,
                    `area' REAL,
                    `iscrowd` INTEGER,
                    `image_id` INTEGER,
                    `bbox` TEXT,
                    `category_id` INTEGER,
                    `orig_id` INTEGER,
                    `subset` TEXT
                    `isGT` INTEGER

    """

    list_to_move = [json.dumps(item["segmentation"]), \
                    item["area"], \
                    item["iscrowd"], \
                    item["image_id"], \
                    json.dumps(item["bbox"]), \
                    item["category_id"], \
                    item["id"], \
                    subset,\
                    isGT]
    return list_to_move


def add_gt_annotations(gt:dict,
                       subset:str,
                       db_conn:sqlite3.Connection,
                       db_curs:sqlite3.Cursor,
                       empty_db:bool = False):
    """
    Adding GroundTruth data to the database
    Assuming a fully coco compliant json structure
    """
    


    keys = gt.keys()

    # min. required keys are "annotations" and "images"
    if ("images" not in keys) or ("annotations" not in keys):
        raise Exception("Groundtruth data lacks images or annotations.\
                         Please provide a valid groundtruth annotation file")
    
    # check if images are already in DB
    if empty_db or not check_if_images_in_db(subset,\
                                             len(gt["images"]),\
                                             db_curs):
        items_to_insert = Parallel(n_jobs=-1, prefer="threads")(
                            delayed(image_data_to_list)(item, subset)
                                for item in tqdm(gt["images"])
                            )
        db_curs.executemany("INSERT INTO images\
                            VALUES (?,?,?,?,?,?,?,?,?)",
                            items_to_insert)
        db_conn.commit()
    else:
        print("GT images in DB already.")
    


    # check if annotations are in DB first
    if empty_db:
        items_to_insert = Parallel(n_jobs=-1, prefer="threads")(
                            delayed(annotations_to_list)(item, subset, 1)
                                for item in tqdm(gt["annotations"])
                            )
        
        db_curs.executemany("INSERT INTO annotations\
                            VALUES (?,?,?,?,?,?,?,?,?)",
                            items_to_insert)
        db_conn.commit()
    elif not check_if_annotations_in_db(subset,\
                                        len(gt["annotations"]),\
                                        db_curs):
        items_to_insert = Parallel(n_jobs=-1, prefer="threads")(
                            delayed(annotations_to_list)(item, subset, 1)
                                for item in tqdm(gt["annotations"])
                            )
        
        db_curs.executemany("INSERT INTO annotations\
                            VALUES (?,?,?,?,?,?,?,?,?)",
                            items_to_insert)
        db_conn.commit()
    else:
        print("GT annotations in DB already.")

    # licenses
    if "licenses" in keys:
        list_to_move = []
        for lic in gt["licenses"]:
            list_to_move.append([lic["url"], \
                                 lic["id"], \
                                 lic["name"], \
                                 subset])
        db_curs.executemany("INSERT INTO licenses \
                             VALUES (?,?,?,?)", list_to_move)
        db_conn.commit()
    
    # if "catgegories" in keys:
    #     for cat in gt["categories"]:
    #         a = 1

def add_predictions_to_db(predictions:list,
                          model:str,
                          db_curs:sqlite3.Cursor,
                          db_conn:sqlite3.Connection):
    """

    Assuming the following structure for the predictions table:
                     `image_id` INTEGER,
                     `category_id` INTEGER,
                     `bbox` TEXT,
                     `score` REAL,
                     `is_valid_class_in_img` TEXT,
                     `best_match_gt_annotation_id` INTEGER,
                     `model` TEXT,
                     `comments` TEXT
                     
    """
    def generate_prediction_list_(item:dict,\
                                  model:str)->list:
        """

        """
        prediction = [item["image_id"],
                      item["category_id"],
                      json.dumps(item["bbox"]),
                      item["score"],
                      "-0.1",
                      "unknown",
                      -999,
                      model,
                      "none"]
        return prediction

    if not check_if_predictions_in_db(model,\
                                      len(predictions),\
                                      db_curs):
        print("Adding", model)
        items_to_insert = Parallel(n_jobs=-1, prefer="threads")(
                            delayed(generate_prediction_list_)(item, model)
                                for item in tqdm(predictions)
                            )
        db_curs.executemany("INSERT INTO predictions\
                            VALUES (?,?,?,?,?,?,?,?,?)",
                            items_to_insert)
        db_conn.commit()
    else:
        print(model," results already in DB!")


def check_if_model_processed(model,
                             db_conn,
                             db_curs):
    """

    """
    models_procssed = db_curs.execute("SELECT DISTINCT model\
                                        FROM status").fetchall()
    if len(models_procssed) != 0:
        models_procssed = np.array(models_procssed)
        if model in models_procssed:
            is_processed = True
        else:
            is_processed = False
    else:
        is_processed = False
    return is_processed


def get_image_ids_of_pred(model,
                          db_conn,
                          db_curs):
    """

    """
    image_ids = np.array(db_curs.execute("SELECT DISTINCT image_id\
                                          FROM predictions\
                                          WHERE model=?",\
                                          [model]).fetchall())
    return image_ids
    

def process_predictions_per_image(image_id,\
                                  subset, \
                                  model, \
                                  db_conn, \
                                  db_curs):
    """

    """
    # get all valid categories first
    valid_categories = db_curs.execute("SELECT DISTINCT category_id\
                                        FROM annotations\
                                        WHERE subset=? AND image_id=?",\
                                        [subset, image_id]).fetchall()
    
    # returns an Array of tuples, so conversion to np.ndarray
    # makes it much easier to find something in it
    valid_categories = np.array(valid_categories)
    
    # get groundtruth annotations
    gt_annotations = db_curs.execute("SELECT area,bbox,category_id, orig_id\
                                      FROM annotations\
                                      WHERE subset=? AND image_id=?",\
                                      [subset, image_id]).fetchall()
    # get predictions
    pred_annotations = db_curs.execute("SELECT rowid,bbox,category_id\
                                        FROM predictions\
                                        WHERE model=? AND image_id=?",\
                                        [model, image_id]).fetchall()
    correct_class_pred = []
    incorrect_class_pred = []
    for i in range(len(pred_annotations)):
        if pred_annotations[i][2] not in valid_categories:
            # append rowid of incorrect class only
            incorrect_class_pred.append(pred_annotations[i][0])
        else:
            # append full prediction
            correct_class_pred.append(pred_annotations[i])

    # Set all the wrong predictions (classes) to False
    for rID in incorrect_class_pred:
        db_curs.execute("UPDATE predictions\
                          SET is_valid_class_in_img=?\
                          WHERE rowid=?",\
                          ["False", rID])

    # cacluate IoUs
    for prediction in correct_class_pred:
        # best prediction
        # format [orig_id, IoU]
        best_prediction = [-1, 0.0]
        for annotation in gt_annotations:
            # check if class is correct
            if prediction[2] == annotation[2]:
                iou_tmp = bb_iou(json.loads(annotation[1]),\
                                 json.loads(prediction[1]))
                if iou_tmp >= best_prediction[1]:
                    best_prediction = [annotation[3], iou_tmp]
        db_curs.execute("UPDATE predictions\
                         SET (is_valid_class_in_img,\
                         best_match_gt_annotation_id,\
                         IoU)=(?,?,?)\
                         WHERE rowid=?",\
                         ["True",\
                          best_prediction[0],\
                          best_prediction[1],\
                          prediction[0]])

    db_conn.commit()


    





        
    

    
    

                            


    

