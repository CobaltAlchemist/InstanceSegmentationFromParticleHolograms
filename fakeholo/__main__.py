import os
import sys
import cv2
from tqdm.auto import tqdm
import logging
from typer import Typer, Option
from .server import genserver
import json
import pandas as pd
from .generator import SampleGenerator

fakeholo = Typer()
@fakeholo.command()
def server(
    debug: bool = Option(False, '-d', '--debug', help='Run the server in debug mode'),
    log: str = Option('output/unconfiguredserver.log', '-l', '--log', help='The file to store the server logs'),
    quiet: bool = Option(False, '-q', '--quiet', help='Log only to the file except for errors'),
    host: str = Option('0.0.0.0', '-h', '--host', help='The host to bind the server to'),
    port: int = Option(5000, '-p', '--port', help='The port to bind the server to')
):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(filename=log, level=level, format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
    if not quiet:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    genserver.run(host=host, port=port)

@fakeholo.command()
def generate(
    config: str = Option('config.json', '-c', '--config', help='The configuration file to use'),
    output: str = Option('.', '-o', '--output', help='The folder to save the generated samples to'),
    count: int = Option(1, '-n', '--count', help='The number of samples to generate')
):
    with open(config) as f:
        config = json.load(f)
    image_folder = f'{output}/images'
    labels_file = f'{output}/labels.csv'
    pd.DataFrame(columns=['filename', 'x', 'y', 'width', 'height', 'contour']).to_csv(labels_file, index=False)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    for i in tqdm(range(count), total=count):
        generator = SampleGenerator.from_dict(config)
        sample = generator.generate()
        labels = []
        filename = f'sample_{i}.png'
        for bb in sample.bounding_boxes:
            labels.append({
                'filename': filename,
                'x': bb.x,
                'y': bb.y,
                'width': bb.w,
                'height': bb.h,
                'contour': bb.contour
            })
        pd.DataFrame(labels).to_csv(labels_file, index=False, mode='a')
        image_file = f'{image_folder}/{filename}'
        tqdm.write(f'Saving {image_file}')
        cv2.imwrite(image_file, sample.image.as_arr)

if __name__ == '__main__':
    fakeholo()
