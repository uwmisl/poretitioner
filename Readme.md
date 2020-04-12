Poretitioner 
=============

## Description 

![](NTER_gif_1_sm.gif)


## Getting Started 


### Developers 

- Clone the project 

```
git clone https://github.com/uwmisl/poretitioner.git
```

- Navigate to the repository 

- Run `bash ./bootstrap_dev.sh`
   - This will set you up with Nix (our package manager) and all other developer dependencies needed to build and contribute to the project 
   
- 
#### Build the application 

- Run `nix-build -A app`
- `./result/bin/poretitioner`

#### Build a docker image of the application 
Docker images can only be built on Linux machines. 

- Run 

```docker_image=$(nix-build -A docker)```

- The environment variable `docker_image` now contains a path to the docker image, copy this file wherever you need it. 

- ```docker load < ${docker_image}```
