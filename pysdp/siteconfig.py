import os

def update_config(config):
    config['data_dir'] = os.path.join(os.path.expanduser('~'), 'Klima', 'data')
