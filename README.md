Commands to train the model:
docker run -it --rm -v $PWD:/workspace llm_training:latest /bin/bash

Inside the container run:
python3 -m training.train