import random
import math
import time
from datetime import datetime, timedelta
import pymongo
from pymongo import MongoClient
import sys

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["sample_data"]
candle_collection = db["candles"]

# Define constants
start_date = datetime(2010, 1, 1, 0, 0)
end_date = datetime(2023, 8, 14, 23, 59)
baseline_price = 30000  # BTCUSDT's M futures price baseline
frequency = timedelta(hours=1)
chunk_size = 200 * 1024 * 1024  # 200 MB
current_chunk_size = 0
buffer = []


def create_candle(timestamp, open_, high, low, close, volume, reward):
  return {
      "timestamp": timestamp,
      "open": open_,
      "high": high,
      "low": low,
      "close": close,
      "volume": volume,
      "reward": reward,
  }


# Calculate the size of a single candle object
sample_candle = create_candle(start_date, 0, 0, 0, 0, 0, 0)
candle_size = sys.getsizeof(str(sample_candle))

current_timestamp = start_date
sin_wave_freq = 0.001
counter = 0

while current_timestamp <= end_date:
  open_ = baseline_price + random.uniform(-100, 100)
  high = open_ + random.uniform(0, 100)
  low = open_ - random.uniform(0, 100)
  close = random.uniform(low, high)
  volume = random.uniform(100, 10000)
  reward = math.sin(counter * sin_wave_freq)

  candle = create_candle(current_timestamp, open_, high, low, close, volume,
                         reward)
  current_chunk_size += candle_size

  if current_chunk_size >= chunk_size:
    # Insert buffer into MongoDB
    candle_collection.insert_many(buffer)
    buffer = []
    current_chunk_size = 0

  buffer.append(candle)
  current_timestamp += frequency
  counter += 1

if buffer:
  # Insert remaining buffer into MongoDB
  candle_collection.insert_many(buffer)

print(f"Generated {counter} lines of candle data.")
