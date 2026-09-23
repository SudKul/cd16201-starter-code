# Optional Cookiecutter template for local MLflow steps

The project already includes the required cleaning scaffold. Cookiecutter is only for
adding another step. If you choose to use it, install the optional authoring tool
before creating a step:

```sh
python -m pip install cookiecutter==2.6.0
cookiecutter cookie-mlflow-step -o src
```

Provide a step name, script name, description, and comma-separated parameter names
without spaces. The generated script has an explicit TODO and accepts string
parameters. Edit the script and `MLproject` parameter types for your step. Use
project-local files for inputs and outputs, and run it with the preinstalled
Workspace environment using `--env-manager=local`.
