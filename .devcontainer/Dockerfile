# Dev container image: a base system and the pixi binary, nothing else. The
# repository is bind-mounted, and `pixi install` (postCreateCommand) builds the
# same locked environment as on a host.
FROM mcr.microsoft.com/devcontainers/base:noble

# The pixi version CI uses.
ARG PIXI_VERSION=v0.72.2
RUN curl -fsSL -o /usr/local/bin/pixi \
      "https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-$(uname -m)-unknown-linux-musl" \
    && chmod +x /usr/local/bin/pixi \
    && pixi --version

USER vscode
RUN echo 'eval "$(pixi completion -s bash)"' >> /home/vscode/.bashrc
