import logging
import os
import sys
from datetime import datetime
from exception import CustomException  


LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging setup complete.")
    try: 
        1 / 0  # This will raise a ZeroDivisionError to test logging
    except Exception as e:
        logging.info(f"Error occurred: {e}")
        raise CustomException(e, sys)