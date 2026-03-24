# MRMR

## Overview
A machine learning project focused on feature selection and ranking.

## Getting Started

### Prerequisites
- Python 3.8+
- pandas
- scikit-learn
- numpy

### Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from mrmr import MRMR

# Initialize and run feature selection
selector = MRMR()
selected_features = selector.fit_transform(X, y)
```

## Features
- Fast feature selection
- Multiple ranking algorithms
- Support for classification and regression

## Contributing
Pull requests welcome. Please open an issue first to discuss changes.

## License
MIT
