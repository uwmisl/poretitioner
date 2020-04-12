Poretitioner 
=============

Poretitioner is an application for reading reporter proteins from nanopores sensors.


## Getting Started: Users  

Here's how to use Poretitioner out of the box. 

- Download [Docker](https://www.docker.com/), if you don't have it already.

- Download the Docker image here.

- From the directory you downloaded the docker image to, run

```
docker load < docker_poretitioner.tar.gz
```

- Now you're ready to use poretitioner 

```
docker run poretitioner:latest
```


## Getting Started: Developers 

If you're interested in contributing to Poretitioner, here's how to get started:

- Clone the project 

```
git clone https://github.com/uwmisl/poretitioner.git
```

- Navigate to the repository `cd poretitioner`

- Run `bash ./bootstrap_dev.sh`
   - This will set you up with Nix (our package manager) and all other developer dependencies needed to build and contribute to the project 
   
- You're all set! 

#### Build the application 

- Run `nix-build -A app`
- `./result/bin/poretitioner`

#### Build a docker image of the application 
Docker images can only be built on Linux machines. 

- Run 

```docker_image=$(nix-build -A docker)```

- The environment variable `docker_image` now contains a path to the docker image, copy this file wherever you need it. 

```docker load < ${docker_image}```


# How it works 

The full paper, [*Multiplexed direct detection of barcoded protein reporters on a nanopore array*,](https://www.biorxiv.org/content/10.1101/837542v1), describes the application in detail. Here's a summary in GIF form:
![](NTER_gif_1_sm.gif)

