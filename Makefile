# GNU Make on Windows. Every target is also available through manage.cmd/manage.ps1.
SHELL := cmd.exe
.SHELLFLAGS := /d /s /c
.DEFAULT_GOAL := help
PORT ?= 7876
SOURCES ?= 0
OUT ?= results/benchmarks/latest
MANIFEST ?= benchmark_manifest.json
CHECKPOINTS ?= weights
MODEL_SET ?= all
CLOAKS ?= weights/phantomseal/cloaks/mix_full
ARGS ?=
MANAGE = powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(CURDIR)/manage.ps1"

.PHONY: help setup models import-models configure doctor run stop status test smoke smoke-extra benchmark benchmark-help
# run stays attached to this terminal with live logs; stop it with Ctrl+C.
help setup models doctor run stop status test smoke smoke-extra benchmark-help:
	$(MANAGE) $@ -Port "$(PORT)" $(ARGS)

configure:
	$(MANAGE) configure -Checkpoints "$(CHECKPOINTS)" -Cloaks "$(CLOAKS)" $(ARGS)

import-models:
	$(MANAGE) import-models -Checkpoints "$(CHECKPOINTS)" -ModelSet "$(MODEL_SET)" $(ARGS)

benchmark:
	$(MANAGE) benchmark -Port "$(PORT)" -Sources "$(SOURCES)" -Manifest "$(MANIFEST)" -OutputDir "$(OUT)" $(ARGS)
