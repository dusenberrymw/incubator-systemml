#!/usr/bin/env bash
ps -ef | grep `whoami` | grep "[p]ython" | awk '{print $2}' | xargs kill -9
ps -ef | grep `whoami` | grep "[j]ava" | awk '{print $2}' | xargs kill -9
