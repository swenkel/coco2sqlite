###############################################################################
# Convert coco annotations to a SQLite database toolchain                     #
#                                                                             #
#  handling json file utils                                                   #
#                                                                             #
# (c) 2020 Simon Wenkel                                                       #
# Released under the Apache 2.0 license                                       #
#                                                                             #
###############################################################################



#
# import libraries
#
import json

#
# functions
#
def load_json(file:str) -> dict:
    """

    """
    with open(file, "r") as f:
        data = f.read()
    f.close()
    return json.loads(data)