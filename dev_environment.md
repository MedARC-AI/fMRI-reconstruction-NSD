# Development environment on Docker

To support development without focussing too much on configuration, we have set up a Docker Compose environment that can be set up with 1 command.

When using your GPU, there's no need for CUDA or CUDNN to be installed. OS-level NVIDIA drivers are ~~CUDAnough~~ good enough as CUDA and CUDNN is inside the container.

Whilst not required, this setup gives you the exact same environment and dependencies as everyone else. This will avoid the "works on my machine" type of bugs.

## How to get started

**Todo**: The author has a Linux-first setup. If you use Windows, please help to update this readme to work for Windows, too.

**Requirements**:

 - Docker: https://docs.docker.com/engine/install
 - Docker-compose: https://docs.docker.com/compose/install

**Installation:**

- Clone this repo: 
    - `git clone https://github.com/LAION-AI/medical.git` 
    - or `git clone git@github.com:LAION-AI/medical.git` if using SSH.
- `cd` into the cloned repo.
- Run `docker-compose up -d`. It should build the image and spawn the container. This might take a while depending on your internet connection.
- Now, run `docker-compose logs --tail=100` to get the logs in your terminal. Inside, you will find a URL and a token:

    ```
    laion_medical_research_1  |     To access the server, open this file in a browser:
    laion_medical_research_1  |         file:///home/dev/.local/share/jupyter/runtime/jpserver-1-open.html
    laion_medical_research_1  |     Or copy and paste one of these URLs:
    laion_medical_research_1  |         http://hostname:8888/lab?token=a867047cb0418730f08967096a80b2c6f8acb624e01204d2
    laion_medical_research_1  |      or http://127.0.0.1:8888/lab?token=a867047cb0418730f08967096a80b2c6f8acb624e01204d2
    ```

- Copy and paste the link pointing to `http://127.0.0.1:8888` in your browser. You should now have access to all the notebooks with local inference/training.

## Keeping dependencies up to date

If you update dependencies (for example, because we need a new dependency), the best way to do it is like this:

 - Open the terminal inside the container (either in JupyterLab or `docker-compose exec laion_medical_research bash`)
 - Run `pip install <package name>` to install your dependency
 - Run `pip list --format=freeze > Docker/requirements.txt` to update the dependencies. Now you can commit and push this to the repo.
    - **Don't** use `pip freeze`! This is because [of an issue with pip](https://github.com/pypa/pip/issues/8174) corrupting the requirements.txt at the moment.
 - Others can run `docker-compose up -d --build` to update their containers with the new dependencies.