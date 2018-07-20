from forms.parameters_form import GeneralClassifierForm, GeneralRegressorForm


def define_param_form(target_type):
    if target_type == 'numerical':
        return GeneralRegressorForm()
    else:
        return GeneralClassifierForm()


def get_number_inputs(categories):
    return len([categories[i] for i in range(len(categories)) if categories[i] != 'none']) - 1,


def get_number_outputs( target, target_type, fs):
    # data =  sess.get('data')
    # number_outputs = 1 if target_type == 'numerical' else data['#Unique Values'][target_type]  # TODO fix
    target_type = categories[target]
    if target_type == 'categorical' or target_type == 'hash':
        return fs.cat_unique_values_dict[target]
    elif 'range' in target_type:
        return [str(a) for a in list(range(min(fs.df[target].values), max(fs.df[target].values)))]
    return 1

