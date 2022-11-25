---
title: Makefile for docker-compose
date: 2021-04-10
author: Shanaka DeSoysa
description: Makefile for basic docker-compose commands.
---

# Makefile for docker-compose

Makefile for basic docker-compose commands.

```sh
## Project Name
##
##   Parameters
##-------------------------|-----------------------------------------------------------------
## COMPOSE_FILE            | Name of compose file (docker-compose.yml)
## BUILD_CACHE             | Docker build cache (or --no-cache)

COMPOSE_FILE := docker-compose.yml
BUILD_CACHE :=
#"--no-cache"

THIS_FILE := $(lastword $(MAKEFILE_LIST))
CMD_ARGUMENTS ?= $(cmd)

.PHONY: help build up start down destroy stop restart logs logs-web ps

##   Commands
##-------------------------|-----------------------------------------------------------------
## help                    | print docs (make help) or dry run a command (make help cmd=run)

help : Makefile
ifeq ($(CMD_ARGUMENTS),)
	@sed -n 's/^##//p' $< | less
else
	${MAKE} $(CMD_ARGUMENTS) --dry-run
endif

## build                   | docker-compose build
build:
		docker-compose -f $(COMPOSE_FILE) build $(c)

## up                      | docker-compose up
up:
		docker-compose -f $(COMPOSE_FILE) up -d $(c)

## start                   | docker-compose start
start:
		docker-compose -f $(COMPOSE_FILE) start $(c)

## down                    | docker-compose down
down:
		docker-compose -f $(COMPOSE_FILE) down $(c)

## destroy                 | docker-compose down and remove volumes
destroy:
		docker-compose -f $(COMPOSE_FILE) down -v $(c)

## stop                    | docker-compose stop
stop:
		docker-compose -f $(COMPOSE_FILE) stop $(c)

## restart                 | docker-compose stop and up
restart:
		docker-compose -f $(COMPOSE_FILE) stop $(c)
		docker-compose -f $(COMPOSE_FILE) up -d $(c)

## logs                    | docker-compose logs
logs:
		docker-compose -f $(COMPOSE_FILE) logs --tail=100 -f $(c)

## ps                      | docker-compose ps
ps:
		docker-compose -f $(COMPOSE_FILE) ps
```
