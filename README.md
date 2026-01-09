Commands to train the model:
docker build -f training/Dockerfile -t llm_training .
docker run -it --rm -v $PWD:/workspace llm_training:latest /bin/bash

Inside the container run:
python3 -m training.train