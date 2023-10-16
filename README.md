# IntentClassification

Create the virtual environment

```virtualenv venv```

Activate the virtual environment

```source venv/bin/activate```

Install the required dependencies into the environment

```pip install -r requirements.txt```

Run the uvicorn standard command to run the app.py

```uvicorn app:app --host 0.0.0.0 --port 8000 --reload```

Use Postman POST api to check the intent of the statment -  use the below url for post action.
```http://0.0.0.0:8000/intent```

Also can give the query using the browser

```http://0.0.0.0:8000/```