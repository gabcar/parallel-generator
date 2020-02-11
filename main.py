import src
import example
import utils.config_handler as ch

config = ch.load_config_as_dict('configs/volume.yml')
ch.print_configs(config)

g = example.VolumeToSliceGenerator(**config['data'])

import matplotlib.pyplot as plt
plt.subplots(1, 2)
plt.ion()
while True:
    batch = g(1)

    plt.clf()

    plt.subplot(1, 2, 1)
    plt.imshow(batch[0][0].squeeze(), cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(batch[0][1].squeeze())

    plt.pause(1)


