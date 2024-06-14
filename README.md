# AML Mini Challenge: Cross-Selling of Credit Cards
This repository holds the work and results of the Mini Challenge in the module Applied Machine Learning (aml) at FHNW.

## Project Structure
- **[`data/`](/data/)**: The data directory holds all `.csv` files from the [PKDD'99 Financial Data Set](https://sorry.vse.cz/~berka/challenge/pkdd1999/berka.htm).
  - [`translation_mappings.json`](/data/translation_mappings.json): The `translation_mappings.json` holds all variable mappings from Czech to English. It is utilized by the [`DataLoader`](/src/data_utils.py) and translates the variables into english at the onloading.
- **[`images/`](/images/)**: Inside the images folder the Entity Relationship Diagram of the Data Set can be found.
- **[`src/`](/src/)**: The source directory contains outsourced code such as utility functions and classes.
  - [`data_utils.py`](/src/data_utils.py): The `data_utils.py` file contains the `DataLoader` class that streamlines the process of loading the dataset into memory.
  - [`plot_utils.py`](/src/plot_utils.py): The `plot_utils.py` file holds all plotting functions that are used during exploration and analysis inside the main `notebook.ipynb`.
  - [`train_utils.py`](/src/train_utils.py): The `train_utils.py` file houses utility functions used for training like the `cross_validate` function - The centerpiece for training the explored models.
- [`notebook.html`](/notebook.html) and [`notebook.ipynb`](/notebook.ipynb): The main notebooks contain all required specifications from data preparation to model explanations.
- [`USE-OF-AI.md`](/USE-OF-AI.md): This markdown notes how assistants like ChatGPT and GitHub CoPilot were used.
- [`README.md`](/README.md): This markdown.
- [`requirements.txt`](/requirements.txt): This textfile contains all dependencies used in the project.
