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


def write_features(categories, data, writer, config_file):
    for label, categories in zip(data.index, categories):
        cat = data.Category[label] if data.Category[label] != 'range' else 'int-range'
        if 'none' in cat:
            cat = 'none' + '-' + categories if 'none' not in categories else categories
            writer.add_item('COLUMN_CATEGORIES', label, cat)
    writer.write_config(config_file)


def remove_target(features, target):
    sfeatures = features.copy()
    if target <= len(features):
        sfeatures.pop(target)
    return sfeatures


def get_new_features(form, feat_defaults, target, fs_list):
    new_features = {}
    for k, v in feat_defaults.items():
        if k not in fs_list:
            new_features[k] = form[k] if k != target else feat_defaults[k]
    return new_features
