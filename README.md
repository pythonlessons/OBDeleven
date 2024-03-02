# OBDeleven

## How to run the project
- Create a virtual environment:
```bash
python -m venv venv
```
- Activate the virtual environment:
```bash
# ubuntu:
source venv/bin/activate
# windows
.\venv\Scripts\activate
```
- Install the dependencies:
```bash
pip install -r requirements.txt
```
- Update OpenAI API key in `configs.yaml` or set it as an environment variable:
```bash
export OPEN_AI_API_KEY=<your_api_key>
```
- Update the `configs.yaml` file with the required parameters or use default values.
- Update `songs.txt` with the list of songs you want to include in the RAG model vectorized database.

- Run the project with default parameters:
```bash
flask run
```
- Run the project with custom parameters:
```bash
flask run --port 5000 --host 0.0.0.0
```
The project will be running on `http://127.0.0.1:5000/`, you can play around by chating with the RAG model.

## How to test the RAG model
When you were sucessfully able to run this project, you can evaluate the RAG model using `tests.py` file. Make sure to update the `configs.yaml` file with the required parameters before running the tests. Update `questions` and `ground_truths` with the questions and their respective ground truths you want to evaluate the RAG model on.
When everything is updates, run the tests using the following command:
```bash
python tests.py
```
Test results will be saved in `test_results.csv` file.


## This project can be easily deployed to any remote server. For example:
https://flask.palletsprojects.com/en/3.0.x/tutorial/deploy/

