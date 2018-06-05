from configparser import ConfigParser


class ConfigWriter:

    def __init__(self, form):
        self.form = form
        self.config = ConfigParser()

    def items(self):
        result = []
        for k, value in self.form.items():
            section, key = k.split('-', 1)
            result.append((section.upper(), key, value))
        return result

    def populate_config(self):
        for section, key, value in self.items():
            if section not in self.config.sections():
               self.config.add_section(section)
            self.config.set(section, key, value)

