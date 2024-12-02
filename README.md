text_classification
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

## Installation

1. Clone the repository:
```bash
git clone git@github.com:nfrvnikita/service_text_classification.git
cd service_text_classification
```

2. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies using Poetry:
```bash
poetry install
```

## Running the Application

1. Activate Poetry virtual environment:
```bash
poetry shell
```

2. Run the web application:
```bash
streamlit run app.py
```

The application will be available at http://localhost:8501

## Project Features

- Text preprocessing and cleaning
- BERT-based text classification
- Web interface using Streamlit
- Data balancing and vectorization capabilities
- Visualization tools for model metrics

## Tech Stack

- Python 3.11+
- PyTorch
- Transformers (Hugging Face)
- Streamlit
- Pandas
- Scikit-learn
- Poetry for dependency management

## Contributing

1. Fork the repository
2. Create your feature branch:
```bash
git checkout -b feat/your-feature-name
```

3. Make your changes and commit them:
```bash
git add .
git commit -m "feat: Add some feature"
```

4. Push to your fork:
```bash
git push origin feat/your-feature-name
```

5. Create a Pull Request to the main repository

### Guidelines for Contributors

- Use Poetry for dependency management
- Follow PEP 8 style guide for Python code
- Add documentation for new features
- Ensure all tests pass before submitting PR
- Keep the code clean and well-documented

## License

This project is licensed under the terms of the MIT license.

## Contact

[Nikita Anufriev](nfrv.nikita@gmail.com)

[Project Link](https://github.com/nfrvnikita/service_text_classification)

