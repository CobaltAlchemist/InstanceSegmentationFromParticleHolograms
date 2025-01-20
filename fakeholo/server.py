from flask import Flask, jsonify, request, Response
from collections import deque
from threading import Thread
import time
import random
import logging
from datetime import datetime
import pickle
from multiprocessing import Semaphore
import copy

genserver = Flask(__name__)

# Assume the buffer size and batch size
BUFFER_SIZE = 256

# The buffer is implemented as a deque, which allows for efficient popping from both ends
buffer = deque(maxlen=BUFFER_SIZE)
buffer_health = Semaphore(BUFFER_SIZE)
max_queued = Semaphore(BUFFER_SIZE)

# Variable to control the generator thread
stop_thread = False

def generate_sample(config):
    from fakeholo.generator import SampleGenerator
    config['seed'] = -1
    config['torch'] = True
    output_size = config['synthesis']['output_size']
    logging.debug(config)
    return SampleGenerator.from_dict(config).generate(output_size)

# This function will generate a new sample and add it to the buffer
def generate_and_store_sample(config):
    logging.info("Starting generator thread")
    samples_generated = 0
    start_time = datetime.now()
    while True:
        if stop_thread:
            logging.info("Stopping generator thread")
            break
        try:
            logging.info("Waiting for buffer space")
            buffer_health.acquire()
            logging.debug("Generating sample")
            c_config = copy.deepcopy(config)
            sample = generate_sample(c_config)  # your provided function
            if (len(sample.bounding_boxes) == 0):
                logging.info("No bounding boxes, skipping")
                buffer_health.release()
                continue
            sample = pickle.dumps(sample)
            buffer.append(sample)
            samples_generated += 1
            # Log the buffer size every second
            if (datetime.now() - start_time).seconds > 0:
                logging.info(f"Buffer size: {len(buffer)}")
                logging.info(f"Samples generated per second: {samples_generated / (datetime.now() - start_time).seconds}")
            time.sleep(0.1)  # Sleep to prevent this function from hogging CPU
            max_queued.release()
        except Exception as e:
            print(e)
            raise e

generator_thread = None
config = {}

@genserver.route('/set_config', methods=['POST'])
def set_config():
    global generator_thread
    global stop_thread
    global buffer
    global config
    logging.info("Got set config")
        
    if config == request.json:
        logging.warning("Already configured")
        return jsonify({"message": "Configuration unchanged."}), 200

    # Signal the current generator thread to stop
    stop_thread = True

    # Wait for it to stop
    if generator_thread and generator_thread.is_alive():
        generator_thread.join()
        
    config = request.json

    # Clear the buffer
    buffer.clear()

    # Restart the generator thread with new config
    stop_thread = False
    logging.info(f"Restarting thread with config: {config}")
    generator_thread = Thread(target=generate_and_store_sample, args=(config,))
    generator_thread.start()
    while not generator_thread.is_alive():
        logging.info("Waiting for generator thread to start")
        time.sleep(0.1)

    return jsonify({"message": "Configuration updated and buffer reset."}), 200

@genserver.route('/get_sample', methods=['GET'])
def get_batch():
    try:
        if len(buffer) < BUFFER_SIZE // 2:
            return jsonify({"error": "Not enough samples in buffer, try again later"}), 400
        max_queued.acquire(block=False)
        sample = buffer[random.randint(0, len(buffer) - 1)]
        buffer_health.release()
        response = Response(sample, content_type="application/octet-stream")
        return response, 200
    except Exception as e:
        print(e)
        raise e
    
@genserver.route('/health', methods=['GET'])
def health():
    return jsonify({"message": "Server is healthy."}), 200
    
@genserver.route('/close', methods=['POST'])
def close():
    global stop_thread
    stop_thread = True
    return jsonify({"message": "Server stopped."}), 200