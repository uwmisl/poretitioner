FROM alpine:3.8 as intermediate

# install git
RUN apk --update add git openssh

# add credentials on build
ARG SSH_KEY
RUN mkdir /root/.ssh/

# Add the keys and set permissions
RUN echo "${SSH_KEY}" > /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa

# Ensure your domain is accepted
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts


RUN eval $(ssh-agent) && \
    ssh-add && \
    ssh-keyscan -H github.com >> /root/.ssh/known_hosts && \
    git clone git@github.com:uwmisl/poretitioner.git

WORKDIR poretitioner

# TODO: Point this out the appropriate release branch. https://github.com/uwmisl/poretitioner/issues/9
RUN git checkout origin/jdunstan/docker-image


FROM lnl7/nix:2.3.3

COPY --from=intermediate /poretitioner /usr/poretitioner
WORKDIR /usr/poretitioner

RUN nix-env --file ./nix/env.nix --install
RUN nix-build --show-trace ./default.nix

WORKDIR /usr/poretitioner/result/bin/

ENTRYPOINT [ "poretitioner" ]
