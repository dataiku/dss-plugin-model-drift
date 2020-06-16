def get_partitioning_columns(dataset):
    """
    Dataset DSS object
    :param dataset: 
    :return: The partitioning columns as a list of strings
    """
    partitioning_settings = dataset.get_config().get('partitioning', {})
    partitioning_dimensions = partitioning_settings.get('dimensions', [])
    is_filesystem_partition = 'filePathPattern' in dataset.get_config().get('partitioning', {})
    if len(partitioning_dimensions) > 0 and not is_filesystem_partition:
        partitioning_columns = [col.get('name') for col in partitioning_dimensions]
        return partitioning_columns
    return []

