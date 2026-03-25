.DEFAULT_GOAL := help
.PHONY: help
.PHONY: context-qmd-tree
.PHONY: context-index context-get context-contracts context-modules context-classes context-functions
.PHONY: context-match context-qmd-outline context-typst-outline context-typst-includes
.PHONY: context-literature-index context-literature-search migrate-codex-memory
.PHONY: context-heavy context-uml context-uml-preview context-docstrings context-tree context-dir-tree context-dir-tree-external check-agent-memory

# Color codes
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m
RED := \033[0;31m

# Project directories
PKG_DIR := oracle_rri
SRC_DIR := oracle_rri
TEST_DIR := tests
DOCS_DIR := docs
TYPST ?= typst
TYPST_ROOT ?= docs
TYPST_PAPER ?= $(DOCS_DIR)/typst/paper/main.typ
TYPST_PAPER_PDF ?= $(DOCS_DIR)/typst/paper/main.pdf
TYPST_SLIDES_DIR ?= $(DOCS_DIR)/typst/slides
SLIDES ?= slides_4.typ
SLIDES_FILE := $(if $(filter %.typ,$(SLIDES)),$(SLIDES),$(SLIDES).typ)
SLIDES_SRC := $(if $(findstring /,$(SLIDES_FILE)),$(SLIDES_FILE),$(TYPST_SLIDES_DIR)/$(SLIDES_FILE))
SLIDES_PDF ?= $(SLIDES_SRC:.typ=.pdf)

# Python interpreter (uv-managed .venv by default)
VENV_PYTHON ?= $(CURDIR)/oracle_rri/.venv/bin/python
PYTHON_INTERPRETER ?= $(VENV_PYTHON)
FORCE_ACTIV_CONDA_ENV ?= 0  # set to 1 only if you insist on the old conda env check
CONDA_ENV_NAME ?= aria-nbv       # legacy: expected conda env name
QMD_FORMATTER := scripts/format_qmd_lists.py
CONTEXT_DIR ?= docs/_generated/context
CONTEXT_OUT ?= $(CONTEXT_DIR)/context_snapshot.md
CONTEXT_INDEX_OUT ?= $(CONTEXT_DIR)/source_index.md
CONTEXT_UML_OUT ?= $(CONTEXT_DIR)/oracle_rri_uml.mmd
CONTEXT_UML_FILTERED_OUT ?= $(CONTEXT_DIR)/oracle_rri_filtered_uml.mmd
CONTEXT_DOCSTRINGS_OUT ?= $(CONTEXT_DIR)/oracle_rri_class_docstrings.md
CONTEXT_CONTRACTS_OUT ?= $(CONTEXT_DIR)/data_contracts.md
CONTEXT_TREE_OUT ?= $(CONTEXT_DIR)/oracle_rri_tree.md
CONTEXT_MERMAID_EXCLUDE ?= data.downloader,vin.experimental,app
LITERATURE_INDEX_OUT ?= $(CONTEXT_DIR)/literature_index.md
GET_CONTEXT_MODE ?= packages
GET_CONTEXT_QUERY ?=
GET_CONTEXT_ROOT ?=
QMD_OUTLINE_ARGS ?= --compact
TYPST_OUTLINE_ARGS ?= --paper --mode outline
TYPST_INCLUDES_ARGS ?= --paper --mode includes
LITERATURE_SEARCH_QUERY ?=
MIGRATE_CODEX_MEMORY_ARGS ?=
MMDC ?= mmdc
MMD_DIR ?= external/mmdc-examples
MMD_OUT ?= $(MMD_DIR)
MMD_FORMAT ?= png
MMD_SCALE ?= 4

#  ══════════════════════════════════════════════════════════════════════
#  Agent Context helpers
#  ══════════════════════════════════════════════════════════════════════

.PHONY: _check_python
_check_python:
	@if [ ! -x "$(PYTHON_INTERPRETER)" ]; then \
		echo "$(RED)🚫 Python interpreter not found at $(PYTHON_INTERPRETER)$(NC)"; \
		echo "$(YELLOW)Run: UV_PYTHON=/home/jandu/miniforge3/envs/aria-nbv/bin/python uv sync$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Using python: $(PYTHON_INTERPRETER)$(NC)"

context-package: _check_python ## 🗺️ Summarize symbols per module (classes/functions/constants)
	@$(PYTHON_INTERPRETER) oracle_rri/scripts/get_context.py packages --root oracle_rri/oracle_rri

context-index: ## 🗺️ Regenerate docs/_generated/context/source_index.md
	@./scripts/nbv_context_index.sh "$(CONTEXT_INDEX_OUT)"

context-get: _check_python ## 🗺️ Run AST context helper (GET_CONTEXT_MODE, optional GET_CONTEXT_QUERY / GET_CONTEXT_ROOT)
	@bash -lc 'set -euo pipefail; \
		args=("$(GET_CONTEXT_MODE)"); \
		if [[ -n "$(strip $(GET_CONTEXT_QUERY))" ]]; then \
			args+=("$(GET_CONTEXT_QUERY)"); \
		fi; \
		if [[ -n "$(strip $(GET_CONTEXT_ROOT))" ]]; then \
			args+=("$(GET_CONTEXT_ROOT)"); \
		fi; \
		./scripts/nbv_get_context.sh "$${args[@]}"'

context-contracts: _check_python ## 🗺️ Show data/config contract index for oracle_rri
	@./scripts/nbv_get_context.sh contracts

context-modules: _check_python ## 🗺️ Show oracle_rri module map
	@./scripts/nbv_get_context.sh modules

context-classes: _check_python ## 🗺️ Show class summaries for oracle_rri
	@./scripts/nbv_get_context.sh classes

context-functions: _check_python ## 🗺️ Show public function summaries for oracle_rri
	@./scripts/nbv_get_context.sh functions

context-match: _check_python ## 🗺️ Search AST summaries (set GET_CONTEXT_QUERY=<term>)
	@if [ -z "$(strip $(GET_CONTEXT_QUERY))" ]; then \
		echo "$(RED)GET_CONTEXT_QUERY is required, e.g. make context-match GET_CONTEXT_QUERY=VinPrediction$(NC)"; \
		exit 2; \
	fi
	@./scripts/nbv_get_context.sh match "$(GET_CONTEXT_QUERY)"

context-qmd-outline: _check_python ## 🗺️ Outline Quarto pages (QMD_OUTLINE_ARGS='--compact' by default)
	@./scripts/nbv_qmd_outline.sh $(QMD_OUTLINE_ARGS)

context-typst-outline: _check_python ## 🗺️ Outline Typst paper/slides (TYPST_OUTLINE_ARGS='--paper --mode outline')
	@$(PYTHON_INTERPRETER) ./scripts/nbv_typst_includes.py $(TYPST_OUTLINE_ARGS)

context-typst-includes: _check_python ## 🗺️ Print Typst include edges (TYPST_INCLUDES_ARGS='--paper --mode includes')
	@$(PYTHON_INTERPRETER) ./scripts/nbv_typst_includes.py $(TYPST_INCLUDES_ARGS)

context-literature-index: ## 🗺️ Regenerate docs/_generated/context/literature_index.md
	@./scripts/nbv_literature_index.sh "$(LITERATURE_INDEX_OUT)"

context-literature-search: ## 🗺️ Search literature sources (set LITERATURE_SEARCH_QUERY=<term>)
	@if [ -z "$(strip $(LITERATURE_SEARCH_QUERY))" ]; then \
		echo "$(RED)LITERATURE_SEARCH_QUERY is required, e.g. make context-literature-search LITERATURE_SEARCH_QUERY=GenNBV$(NC)"; \
		exit 2; \
	fi
	@./scripts/nbv_literature_search.sh "$(LITERATURE_SEARCH_QUERY)"

migrate-codex-memory: _check_python ## 🗺️ Migrate legacy .codex notes into .agents/memory
	@$(PYTHON_INTERPRETER) scripts/migrate_codex_memory.py $(MIGRATE_CODEX_MEMORY_ARGS)

check-agent-memory: _check_python ## 🗺️ Validate agent memory scaffolding and debrief hygiene
	@$(PYTHON_INTERPRETER) scripts/validate_agent_memory.py

context: _check_python ## 🗺️ Refresh lightweight context artifacts (source index, literature index, data contracts)
	@bash -lc 'set -euo pipefail; \
		context_dir="$(CONTEXT_DIR)"; \
		index_out="$(CONTEXT_INDEX_OUT)"; \
		contracts_out="$(CONTEXT_CONTRACTS_OUT)"; \
		lit_index_out="$(LITERATURE_INDEX_OUT)"; \
		mkdir -p "$$context_dir"; \
		mkdir -p "$$(dirname "$$index_out")"; \
		scripts/nbv_context_index.sh "$$index_out" >/dev/null; \
		scripts/nbv_literature_index.sh "$$lit_index_out" >/dev/null; \
		{ \
			echo "# Data Contracts (oracle_rri)"; \
			echo ""; \
			$(PYTHON_INTERPRETER) oracle_rri/scripts/get_context.py contracts --root oracle_rri/oracle_rri \
				| sed "1{/^# Data Contracts$$/d;}"; \
		} > "$$contracts_out"; \
		echo "Wrote: $$index_out"; \
		echo "Wrote: $$lit_index_out"; \
		echo "Wrote: $$contracts_out"'
	@echo "$(GREEN)Refreshed lightweight context artifacts in $(CONTEXT_DIR)$(NC)"
	@echo "$(BLUE)Heavy fallback: make context-heavy$(NC)"
	@echo "$(BLUE)Tip: rg -n \"<pattern>\" $(CONTEXT_INDEX_OUT)$(NC)"

context-uml: _check_python ## 🗺️ Generate oracle_rri UML artifacts without printing them
	@bash -lc 'set -euo pipefail; \
		context_dir="$(CONTEXT_DIR)"; \
		uml_out="$(CONTEXT_UML_OUT)"; \
		uml_filtered_out="$(CONTEXT_UML_FILTERED_OUT)"; \
		mkdir -p "$$context_dir"; \
		mermaid_tmp="$$(mktemp)"; \
		mermaid_filtered="$$(mktemp)"; \
		$(PYTHON_INTERPRETER) -m syrenka classdiagram oracle_rri/oracle_rri > "$$mermaid_tmp"; \
		exclude_list="$(CONTEXT_MERMAID_EXCLUDE)"; \
		if [[ -z "$$exclude_list" ]]; then \
			cp "$$mermaid_tmp" "$$mermaid_filtered"; \
		else \
			$(PYTHON_INTERPRETER) scripts/filter_mermaid.py \
				--input "$$mermaid_tmp" \
				--output "$$mermaid_filtered" \
				--exclude "$$exclude_list"; \
		fi; \
		cp "$$mermaid_tmp" "$$uml_out"; \
		cp "$$mermaid_filtered" "$$uml_filtered_out"; \
		rm -f "$$mermaid_tmp" "$$mermaid_filtered"; \
		echo "Wrote: $$uml_out"; \
		echo "Wrote: $$uml_filtered_out"'

context-uml-preview: _check_python ## 🗺️ Print the filtered oracle_rri UML to stdout
	@$(MAKE) --no-print-directory context-uml >/dev/null
	@echo "# Mermaid UML Diagram of the oracle_rri:"
	@echo "\`\`\`{mermaid}"
	@cat "$(CONTEXT_UML_FILTERED_OUT)"
	@echo "\`\`\`"

context-docstrings: _check_python ## 🗺️ Generate full oracle_rri class docstrings artifact
	@bash -lc 'set -euo pipefail; \
		context_dir="$(CONTEXT_DIR)"; \
		docstrings_out="$(CONTEXT_DOCSTRINGS_OUT)"; \
		mkdir -p "$$context_dir"; \
		{ \
			echo "# Class Docstrings (oracle_rri)"; \
			echo ""; \
			$(PYTHON_INTERPRETER) oracle_rri/scripts/get_context.py classes --root oracle_rri/oracle_rri --full-doc; \
		} > "$$docstrings_out"; \
		echo "Wrote: $$docstrings_out"'

context-tree: _check_python ## 🗺️ Generate oracle_rri directory tree artifact
	@bash -lc 'set -euo pipefail; \
		context_dir="$(CONTEXT_DIR)"; \
		tree_out="$(CONTEXT_TREE_OUT)"; \
		mkdir -p "$$context_dir"; \
		{ \
			echo "# Directory Tree (oracle_rri)"; \
			echo ""; \
			echo "Directory tree for oracle_rri/oracle_rri/:"; \
			if command -v tree >/dev/null 2>&1; then \
				tree oracle_rri/oracle_rri/ -I "__pycache__"; \
			else \
				find oracle_rri/oracle_rri/ -path "*/__pycache__" -prune -o -print \
					| sed "s#^oracle_rri/oracle_rri/##" \
					| sed "/^$$/d" \
					| sort; \
			fi; \
		} > "$$tree_out"; \
		echo "Wrote: $$tree_out"'

context-heavy: _check_python ## 🗺️ Generate heavyweight fallback artifacts and combined context snapshot
	@$(MAKE) --no-print-directory context
	@$(MAKE) --no-print-directory context-uml
	@$(MAKE) --no-print-directory context-docstrings
	@$(MAKE) --no-print-directory context-tree
	@bash -lc 'set -euo pipefail; \
		out="$(CONTEXT_OUT)"; \
		index_out="$(CONTEXT_INDEX_OUT)"; \
		uml_out="$(CONTEXT_UML_OUT)"; \
		docstrings_out="$(CONTEXT_DOCSTRINGS_OUT)"; \
		contracts_out="$(CONTEXT_CONTRACTS_OUT)"; \
		tree_out="$(CONTEXT_TREE_OUT)"; \
		mkdir -p "$$(dirname "$$out")"; \
		{ \
			echo "# Context Snapshot (make context-heavy)"; \
			echo ""; \
			echo "Generated: $$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")"; \
			echo ""; \
			echo "## Contents"; \
			echo "0) Source index (all context pools)"; \
			echo "1) Environment"; \
			echo "2) Data contracts (oracle_rri)"; \
			echo "3) Mermaid UML (oracle_rri)"; \
			echo "4) Class docstrings (oracle_rri)"; \
			echo "5) Directory tree (oracle_rri)"; \
			echo ""; \
			echo "## 0) Source index (all context pools)"; \
			if [[ -f "$$index_out" ]]; then \
				sed "s/^#/###/" "$$index_out"; \
			else \
				echo "(missing $$index_out)"; \
			fi; \
			echo ""; \
			echo "## 1) Environment"; \
			echo "Python: $(PYTHON_INTERPRETER)"; \
			echo "Venv: $(VENV_PYTHON)"; \
			echo "Recreate: UV_PYTHON=/home/jandu/miniforge3/envs/aria-nbv/bin/python uv sync --extra dev --extra notebook --extra pytorch3d"; \
			echo ""; \
			echo "## 2) Data contracts (oracle_rri)"; \
			if [[ -f "$$contracts_out" ]]; then \
				sed "1{/^# Data Contracts (oracle_rri)$$/d;}" "$$contracts_out"; \
			else \
				echo "(missing $$contracts_out)"; \
			fi; \
			echo ""; \
			echo "## 3) Mermaid UML (oracle_rri)"; \
			echo "\`\`\`{mermaid}"; \
			cat "$$uml_out"; \
			echo "\`\`\`"; \
			echo ""; \
			echo "## 4) Class docstrings (oracle_rri)"; \
			if [[ -f "$$docstrings_out" ]]; then \
				sed "1{/^# Class Docstrings (oracle_rri)$$/d;}" "$$docstrings_out"; \
			else \
				echo "(missing $$docstrings_out)"; \
			fi; \
			echo ""; \
			echo "## 5) Directory tree (oracle_rri)"; \
			if [[ -f "$$tree_out" ]]; then \
				sed "1{/^# Directory Tree (oracle_rri)$$/d;}" "$$tree_out"; \
			else \
				echo "(missing $$tree_out)"; \
			fi; \
		} > "$$out"; \
		echo "Wrote: $$out"'
	@echo "$(GREEN)Wrote heavyweight context snapshot to $(CONTEXT_OUT)$(NC)"

context-external: _check_python ## 🗺️ List classes with full docstrings
	@echo "# Mermaid UML Diagram of the external/efm3d:\n\`\`\`{mermaid}"
	@$(PYTHON_INTERPRETER) -m syrenka classdiagram external/efm3d/efm3d
	@echo "\`\`\`\n---\n"
	@$(PYTHON_INTERPRETER) oracle_rri/scripts/get_context.py classes --root external/efm3d/efm3d --full-doc

	@echo "\n\n"

	echo "# Mermaid UML Diagram of the external/ATEK:\n\`\`\`{mermaid}"
	@$(PYTHON_INTERPRETER) -m syrenka classdiagram external/ATEK/atek
	echo "\`\`\`\n---\n"
	@$(PYTHON_INTERPRETER) oracle_rri/scripts/get_context.py classes --root external/ATEK/atek --full-doc

context-dir-tree: _check_python ## 🗺️ Print directory tree for `oracle_rri/oracle_rri/` (ignore __pycache__)
	@$(MAKE) --no-print-directory _context_dir_tree_print

_context_dir_tree_print:
	@echo "Directory tree for oracle_rri/oracle_rri/:"
	@bash -lc 'tree oracle_rri/oracle_rri/ -I "__pycache__"'

context-dir-tree-external: _check_python ## 🗺️ Print directory tree for `external/efm3d/efm3d` (ignore __pycache__)
	@echo "Directory tree for external/efm3d/efm3d/:"
	@bash -lc 'tree external/efm3d/efm3d/ -I "__pycache__"'
	@echo "\n\n"

	@echo "Directory tree for external/ATEK/atek/:"
	@bash -lc 'tree external/ATEK/atek/ -I "__pycache__"'

context-qmd-tree: ## 🗺️ Print docs/ .qmd structure (ignore __pycache__)
	@echo "Directory tree for docs (.qmd only):"
	@bash -lc 'tree docs -P "*.qmd" -I "__pycache__"'

#  ═══════════════════════════════════════════════════════════════════════
#  📊 Diagrams
#  ═══════════════════════════════════════════════════════════════════════

.PHONY: vin-v2-arch
VIN_V2_ARCH_ARGS ?=
vin-v2-arch: _check_python ## 📊 Generate VIN v2 architecture DOT/SVG/PNG (+ VDX via --drawio)
	@$(PYTHON_INTERPRETER) oracle_rri/scripts/generate_vin_v2_arch.py $(VIN_V2_ARCH_ARGS)

.PHONY: mmdc-render
mmdc-render: ## 📊 Render all .mmd files in a folder (MMD_DIR=..., MMD_OUT=..., MMD_FORMAT=png|svg, MMD_SCALE=4)
	@bash -lc 'set -euo pipefail; \
		in_dir="$(MMD_DIR)"; \
		out_dir="$(MMD_OUT)"; \
		fmt="$(MMD_FORMAT)"; \
		scale="$(MMD_SCALE)"; \
		mkdir -p "$$out_dir"; \
		for f in "$$in_dir"/*.mmd; do \
			[ -e "$$f" ] || continue; \
			base="$$(basename "$$f" .mmd)"; \
			out="$$out_dir/$$base.$$fmt"; \
			if [[ "$$fmt" == "svg" ]]; then \
				$(MMDC) -i "$$f" -o "$$out"; \
			else \
				$(MMDC) -i "$$f" -o "$$out" -s "$$scale"; \
			fi; \
		done'

#  ═══════════════════════════════════════════════════════════════════════
#  📚 Documentation hygiene
#  ═══════════════════════════════════════════════════════════════════════

.PHONY: docs-lint
docs-lint: _check_python ## Format QMD lists, then run Quarto checks
	@echo "$(BLUE)Formatting QMD list spacing…$(NC)"
	@$(PYTHON_INTERPRETER) $(QMD_FORMATTER) $(DOCS_DIR)
	@echo "$(BLUE)Running Quarto check…$(NC)"
	@quarto check

.PHONY: quarto-docs
quarto-docs: ## Render Quarto docs (docs/uml_diagrams)
	@quarto render docs/uml_diagrams

#  ═══════════════════════════════════════════════════════════════════════
#  🧾 Typst builds
#  ═══════════════════════════════════════════════════════════════════════

.PHONY: typst-paper typst-slide
typst-paper: ## Compile the Typst paper (docs/typst/paper/main.typ)
	@$(TYPST) compile --root $(TYPST_ROOT) $(TYPST_PAPER) $(TYPST_PAPER_PDF)

typst-slide: ## Compile a Typst slide deck (make typst-slide SLIDES=slides_4.typ)
	@$(TYPST) compile --root $(TYPST_ROOT) $(SLIDES_SRC) $(SLIDES_PDF)

#  ═══════════════════════════════════════════════════════════════════════
#  ℹ️  Help
#  ═══════════════════════════════════════════════════════════════════════

help: ## Show this help message
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)               NBV Project - Makefile Commands             $(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
		@echo "$(YELLOW)Usage:$(NC) make <target>"
	@awk 'BEGIN {FS = ":.*?## "; section=""} \
		/^#  ═+$$/ {next} \
		/^#  [📦🔍🧪📚🔧🗺️]/ {if (section) print ""; section=$$0; gsub(/^#  /, "", section); print "$(YELLOW)" section "$(NC)"; next} \
		/^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-18s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
