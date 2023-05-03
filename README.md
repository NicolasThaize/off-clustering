# off-clustering
OpenFoodFact data clustering. Entire workflow.

# App description
**/data** : Contains stored data if needed <br>
**/model_saver** : A module which saves, get and run fitted models <br>
**/predict** : Contains every prediction functions<br>
**/metrics** : Contains metrics related functions

# Setup
Requirements:
- Python >= v3.10
- [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
- pip >= v23.1.2

## 1. Install dependencies
### 1.1 Using makefile
```make install-dependencies```

### 1.2 Using command prompt
1. ```conda create --name off-clustering```<br>
2. ```conda activate off-clustering```<br>
3. ```pip install -r requirements.txt```<br>
4. ```pip install -r teaching_ml_2023/requirements.txt```

# Debug

## If module from teching_ml_2023 cannot be imported
Be sure that import path in the module begins by "."<br>
Example: 
If error is 
> ModuleNotFoundError: No module named 'data_loader'

Line<br>
```from data_loader import get_data```<br>
becomes<br>
```from .data_loader import get_data```<br>

# Sources
**teaching_ml_2023** is from [this repository](https://github.com/HerySon/teaching_ml_2023/).
