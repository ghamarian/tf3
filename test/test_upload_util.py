
from utils import upload_util
from session import Session

# def test_create_form():
#     user_configs = {}
#     user_dataset = 'test_dataset'
#     form, url = upload_util.create_form(user_configs, user_dataset)
#     print(form, url)

#def test_existing_data():

#both methods above need app context, hooooow?


def test_generate_dataset_name():
    app_root = 'test_folder'
    username = 'test_user'
    dataset_name = 'test_dataset'

    name = upload_util.generate_dataset_name(app_root, username, dataset_name)
    assert name == dataset_name + '_1'

