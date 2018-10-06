import configparser, os

config = configparser.ConfigParser()
config.read(['telematics.cfg', os.path.expanduser('~/.telematics.cfg'),
             '/etc/telematics/telematics.cfg'])

