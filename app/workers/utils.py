import json


def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except:
        return False