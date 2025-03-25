# Chained Multi-Output Email Classification
 
This project implements a modular architecture for multi-label email classification using a chained multi-output design (Design Decision 1). In this approach, a single RandomForest model predicts a concatenated target that represents three dependent variables:

- **Type2**  

- **Type2 + Type3**  

- **Type2 + Type3 + Type4**
 
The model is trained to output a single string (e.g. `"Yes+No+Yes"`). During evaluation, the output is split into its constituent parts and evaluated in a chained manner—only if Type2 is correctly predicted does the evaluation proceed to Type3, and only if both are correct does it evaluate Type4.
 
## Project Structure
 
``` project/ 
        ├── Config.py # Configuration file (defines column names and constants) 
        ├── embeddings.py # Module for computing TF-IDF embeddings 
        ├── preprocess.py # Data loading, deduplication, noise removal, and label chaining 
        ├── data_model.py # Data encapsulation and train-test splitting 
        ├── base.py # Abstract base class for models 
        ├── main.py # Main controller that executes the pipeline 
        ├── modelling.py # (Optional) Connector for modular model invocation 
          └── model/ 
          └── randomforest.py # Modified RandomForest model implementing chained evaluation 
 ```

## Overview
 
The overall pipeline follows these steps:

1. **Data Loading & Preprocessing:**  

   - Raw CSV files (e.g., *AppGallery.csv* and *Purchasing.csv*) are loaded.

   - Data is cleaned using deduplication and noise removal.

   - Chained labels are built from the dependent variables (`y2`, `y3`, `y4`).
 
2. **Feature Extraction:**  

   - TF-IDF embeddings are generated from text fields (ticket summary and interaction content).
 
3. **Data Modeling:**  

   - The data is encapsulated in a `Data` object that performs train-test splitting while ensuring that only classes with enough samples are retained.
 
4. **Model Training & Evaluation:**  

   - A single RandomForest model (in `model/randomforest.py`) is trained on the concatenated chained target.

   - At prediction time, the output string is split into three parts and evaluated in a chained manner:

     - **Stage 1:** Check if Type2 is predicted correctly.

     - **Stage 2:** Check if the concatenation of Type2 and Type3 is correct (only if Stage 1 passes).

     - **Stage 3:** Check if the full chain (Type2+Type3+Type4) is correct (only if Stage 2 passes).

   - Detailed instance-level results and an overall chained accuracy are printed.
 
## How to Run
 
1. **Prepare the Data:**  

   Place your CSV files (`AppGallery.csv` and `Purchasing.csv`) into a folder named `data` at the project root.
 
2. **Install Dependencies:**  

   Ensure you have Python 3.9 or later installed. Then install the required Python packages. For example:

   ```bash

   pip install numpy pandas scikit-learn

(Add any additional dependencies as needed, such as stanza or transformers if you uncomment translation functions.)
 
Run the Project:

Open a terminal in the project directory and run:
 
  ```bash
  
  python main.py
```
This will execute the entire pipeline and print detailed evaluation results to the console.
 
