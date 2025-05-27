# K-Means

This project implements the K-Means clustering algorithm from scratch and demonstrates its application on a sample dataset (**[Mall Customer](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python/code)**). It includes both standard K-Means and K-Means++ initialization, as well as visualization of clustering results and the Elbow method for optimal cluster selection.

## Features

- Custom K-Means and K-Means++ initialization
- Visualization of clusters and centroids
- Elbow method to determine optimal number of clusters
- Silhouette Score for determinink the number of clusters
- Data preprocessing (encoding categorical variables, dropping unnecessary columns)

## Requirements

- Python 3.10+
- numpy
- pandas
- matplotlib
- seaborn

## Installation

Install the required packages using pip:

```bash
pip install -r requirements.txt
```


## Project Structure

```
K-Means/
├── main.py
├── kmean.py
├── utils.py
├── test.ipynb
├── core.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENCE

```

## Usage

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Run the main script:
    ```bash
    python main.py
    ```
```bash
python main.py
```

This will:
- Load and preprocess the data
- Perform clustering with both K-Means and K-Means++
- Plot the resulting clusters
- Use the Elbow method to help choose the best number of clusters

## File Structure

- [`kmean.py`](kmean.py): Contains the `KMeans` class some methods.
- [`main.py`](main.py): Example usage of the `KMeans` class.
- [`test.ipynb`](test.ipynb): Example usage of the `KMeans` class with the Mall dataset **[Mall Customer dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python/code)**.
- [`requirements.txt`](requirements.txt): List of required Python packages.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.