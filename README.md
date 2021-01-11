[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About The Project

Spark Streaming app to classify mini batches of incoming digit images.


### Built With

* Python 3.8.6
* Spark 3.0.1 (Prebuilt for Apache Hadoop 3.2 and later)
* JDK 1.8.0_161

## Getting Started

### Prerequisites

Install Spark:

- For Mac OS see (local machine): https://kevinvecmanis.io/python/pyspark/install/2019/05/31/Installing-Apache-Spark.html

Download and install Python:

- https://www.python.org/

### Installation

1. Clone the repo
   ```sh
   cd <directory where to place the repo>
   git clone https://github.com/younessz/spark_streaming_img_classifier
   ```
2. Setup Python virtual environment
   ```sh
   cd <repo directory>
   # setup Python virtual environment
   python3.8 -m venv spark_app
   source spark_app/bin/activate
   # installing required Python packages
   python3 -m pip install -r requirements.txt
   ```

## Data

MNIST handwritten digits dataset (cf. streaming_app/src/modeling/data_processing.py).


## Usage

WIP - To be continued

## License

Distributed under the MIT License. See `LICENSE` for more information.


## Maintainer

Youness Zarhloul - [LinkedIn](https://www.linkedin.com/in/youness-zarhloul/)


## Acknowledgements

* Any contributor direct or indirect to the amazing tools used by the project.
