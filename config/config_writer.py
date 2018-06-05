from configparser import ConfigParser
from pprint import pprint


class ConfigWriter:

    def __init__(self, form):
        self.form = form
        self.config = ConfigParser()

    def items(self):
        result = []
        for k, value in self.form.items():
            print(k, value)
            if 'csrf_token' not in k:
                section, key = k.split('-', 1)
                result.append((section.upper(), key, value))
        return result

    def populate_config(self):
        for section, key, value in self.items():
            if section not in self.config.sections():
               self.config.add_section(section)
            self.config.set(section, key, value)

    def write_config(self, path):
        with open(path, 'w') as f:
            self.populate_config()
            self.config.write(f)
