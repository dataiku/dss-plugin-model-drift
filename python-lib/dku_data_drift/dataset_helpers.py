def get_partitioning_columns(dataset):
    """
    Dataset DSS object
    :param dataset:
    :return:
    """
    partitioning_dimensions = dataset.get_definition().get('partitioning', {}).get('dimensions', [])
    if len(partitioning_dimensions) > 0:
        partitioning_type = partitioning_dimensions[0].get('type')
        if partitioning_type == 'value':
            partitioning_columns = [col.get('name') for col in partitioning_dimensions]
            return partitioning_columns
    return None
