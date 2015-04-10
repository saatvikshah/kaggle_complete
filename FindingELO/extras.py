import json


class Params:
    pass


def load_params():
    json_data = open("settings.json")
    data = json.load(json_data)
    params = Params()
    params.data_dir = data["data_dir"]
    params.cache_dir = data["cache_dir"]
    params.partitions = data["num_partitions"]
    return params

def toint(elem):
    try:
            return (int(elem))
    except:
            return 0
