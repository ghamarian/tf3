

def reorder_request(features, categories, defaults, list_features):
    dict_features = {}
    for f, c, d in zip(features, categories, defaults):
        dict_features[f] = {}
        dict_features[f]['category'] = c
        dict_features[f]['default'] = d
    cat_columns = [dict_features[c]['category'] for c in list_features]
    default_values = [dict_features[c]['default'] for c in list_features]
    return cat_columns, default_values


def get_target_labels(target, target_type, fs):
    # TODO labels if target type is a RANGE, BOOL, ...
    if target_type == 'categorical' or target_type == 'hash':
        return fs.cat_unique_values_dict[target]
    elif 'range' in target_type:
        return [str(a) for a in list(range(min(fs.df[target].values), max(fs.df[target].values)))]
    return None
