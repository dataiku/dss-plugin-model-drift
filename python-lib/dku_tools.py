# -*- coding: utf-8 -*-
import datetime
import dataiku

def process_timestamp(timestamp):
    """
    Convert the timestamp to str date
    :param timestamp:
    :return:
    """
    return str(datetime.datetime.fromtimestamp(timestamp / 1000))


def get_train_date(model_version, version_id):
    m = dataiku.Model(model_version, ignore_flow=True)
    for v in m.list_versions():
        if v.get('versionId') == version_id:
            return process_timestamp((v.get('snippet').get('trainDate')))
    return None


def set_column_description(dataset, column_description_dict):
    dataset_schema = dataset.read_schema()
    for col_info in dataset_schema:
        col_name = col_info.get('name')
        col_info['comment'] = column_description_dict.get(col_name)
    dataset.write_schema(dataset_schema)