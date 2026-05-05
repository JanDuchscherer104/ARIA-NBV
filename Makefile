.DEFAULT_GOAL := help
.PHONY: help
.PHONY: context-qmd-tree qmd-frontmatter-check
.PHONY: context-index context-get context-contracts context-modules context-classes context-functions
.PHONY: context-match context-qmd-outline context-typst-outline context-typst-includes
.PHONY: context-literature-index context-literature-search migrate-codex-memory
.PHONY: context-heavy context-uml context-uml-preview context-docstrings context-tree context-dir-tree context-dir-tree-external check-agent-memory
.PHONY: memory-mine agents-db glossary kg-up kg-down kg-capabilities kg-ollama-check kg-search kg-query kg-brief kg-route kg-claim-check kg-consolidate kg-related kg-show-paper kg-sync kg-materialize kg-index-code kg-ingest-docs kg-ingest-docs-smoke kg-enrich kg-ingest-papers kg-export-neo4j kg-semantic-enrich kg-refresh-light kg-refresh-code kg-refresh-lit kg-refresh-full
.PHONY: lrz-probe lrz-resources lrz-resources-gpu lrz-resources-cpu lrz-jobs lrz-dss-init lrz-container-shell lrz-sbatch-cpu lrz-sbatch-single-gpu lrz-sbatch-multigpu

# Color codes
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m
RED := \033[0;31m

# Project directories
PKG_DIR := aria_nbv
SRC_DIR := aria_nbv
TEST_DIR := tests
DOCS_DIR := docs
TYPST ?= typst
TYPST_ROOT ?= docs
TYPST_PAPER ?= $(DOCS_DIR)/typst/seminar_paper/main.typ
TYPST_PAPER_PDF ?= $(DOCS_DIR)/typst/seminar_paper/main.pdf
TYPST_THESIS ?= $(DOCS_DIR)/typst/thesis/main.typ
TYPST_THESIS_PDF ?= $(DOCS_DIR)/typst/thesis/main.pdf
TYPST_PROPOSAL ?= $(DOCS_DIR)/typst/thesis/proposal.typ
TYPST_PROPOSAL_PDF ?= $(DOCS_DIR)/typst/thesis/proposal.pdf
TYPST_SLIDES_DIR ?= $(DOCS_DIR)/typst/seminar_slides
SLIDES ?= slides_4.typ
SLIDES_FILE := $(if $(filter %.typ,$(SLIDES)),$(SLIDES),$(SLIDES).typ)
SLIDES_SRC := $(if $(findstring /,$(SLIDES_FILE)),$(SLIDES_FILE),$(TYPST_SLIDES_DIR)/$(SLIDES_FILE))
SLIDES_PDF ?= $(SLIDES_SRC:.typ=.pdf)

# Python interpreter (uv-managed .venv by default)
VENV_PYTHON ?= $(CURDIR)/aria_nbv/.venv/bin/python
PYTHON_INTERPRETER ?= $(VENV_PYTHON)
FORCE_ACTIV_CONDA_ENV ?= 0  # set to 1 only if you insist on the old conda env check
CONDA_ENV_NAME ?= aria-nbv       # legacy: expected conda env name
QMD_FORMATTER := scripts/format_qmd_lists.py
CONTEXT_DIR ?= docs/_generated/context
CONTEXT_OUT ?= $(CONTEXT_DIR)/context_snapshot.md
CONTEXT_INDEX_OUT ?= $(CONTEXT_DIR)/source_index.md
CONTEXT_UML_OUT ?= $(CONTEXT_DIR)/aria_nbv_uml.mmd
CONTEXT_UML_FILTERED_OUT ?= $(CONTEXT_DIR)/aria_nbv_filtered_uml.mmd
CONTEXT_DOCSTRINGS_OUT ?= $(CONTEXT_DIR)/aria_nbv_class_docstrings.md
CONTEXT_CONTRACTS_OUT ?= $(CONTEXT_DIR)/data_contracts.md
CONTEXT_TREE_OUT ?= $(CONTEXT_DIR)/aria_nbv_tree.md
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
LRZ_SKILL_DIR ?= .agents/skills/lrz-ai-systems
LRZ_SCRIPTS_DIR ?= $(LRZ_SKILL_DIR)/scripts
LRZ_RESOURCES_ARGS ?= summary
LRZ_CMD ?=
LITKG_MANIFEST ?= .agents/external/litkg-rs/Cargo.toml
LITKG_CONFIG ?= .configs/litkg.toml
LITKG_REPO_ROOT ?= .
LITKG_PROFILE ?= thesis-coding
KG_QUERY ?=
KG_TOPIC ?=
KG_TASK ?=
KG_CLAIM ?=
KG_RELATED_PATH ?=
KG_PAPER ?=
KG_LIMIT ?= 24
KG_FORMAT ?= text
KG_DOC_PATHS ?=

#  ══════════════════════════════════════════════════════════════════════
#  Agent Context helpers
#  ══════════════════════════════════════════════════════════════════════

.PHONY: _check_python
_check_python:
	@if ! { [ -x "$(PYTHON_INTERPRETER)" ] || command -v "$(PYTHON_INTERPRETER)" >/dev/null 2>&1; }; then \
		echo "$(RED)🚫 Python interpreter not found at $(PYTHON_INTERPRETER)$(NC)"; \
		echo "$(YELLOW)Run: cd aria_nbv && uv sync --all-extras$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Using python: $(PYTHON_INTERPRETER)$(NC)"

context-package: _check_python ## 🗺️ Summarize symbols per module (classes/functions/constants)
	@$(PYTHON_INTERPRETER) aria_nbv/scripts/get_context.py packages --root aria_nbv/aria_nbv

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

context-contracts: _check_python ## 🗺️ Show data/config contract index for aria_nbv
	@./scripts/nbv_get_context.sh contracts

context-modules: _check_python ## 🗺️ Show aria_nbv module map
	@./scripts/nbv_get_context.sh modules

context-classes: _check_python ## 🗺️ Show class summaries for aria_nbv
	@./scripts/nbv_get_context.sh classes

context-functions: _check_python ## 🗺️ Show public function summaries for aria_nbv
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

agents-db: _check_python ## 🧠 Inspect or maintain .agents/issues,todos,refactors,resolved (set AGENTS_ARGS='validate')
	@$(PYTHON_INTERPRETER) scripts/agents_db.py $(AGENTS_ARGS)

glossary: _check_python ## 📖 Build shared Quarto/Typst/KG glossary artifacts
	@$(PYTHON_INTERPRETER) scripts/glossary_build.py all

qmd-frontmatter-check: _check_python ## 📖 Validate taxonomy frontmatter for rendered Quarto content
	@$(PYTHON_INTERPRETER) scripts/validate_qmd_frontmatter.py docs/contents

memory-mine: _check_python ## 🧠 Mine current repo state (docs, code, history) into repo-local MemPalace
	@echo "$(BLUE)Mining project into MemPalace...$(NC)"
	@mkdir -p .artifacts/mempalace/palace
	@$(PYTHON_INTERPRETER) -m mempalace --palace .artifacts/mempalace/palace mine .
	@$(PYTHON_INTERPRETER) -m mempalace --palace .artifacts/mempalace/palace mine .agents/memory --mode convos
	@echo "$(GREEN)MemPalace mining complete.$(NC)"

kg-up: ## 📚 Start the optional litkg Neo4j runtime
	@.agents/external/litkg-rs/scripts/kg/up.sh

kg-down: ## 📚 Stop the optional litkg Neo4j runtime
	@.agents/external/litkg-rs/scripts/kg/down.sh

kg-capabilities: ## 📚 Show litkg backend/source readiness (set KG_FORMAT=json for machine output)
	@cargo run --manifest-path "$(LITKG_MANIFEST)" -p litkg-cli -- capabilities \
		--config "$(LITKG_CONFIG)" \
		--repo-root "$(LITKG_REPO_ROOT)" \
		--format "$(KG_FORMAT)"

kg-ollama-check: ## 📚 Validate Mac/remote Ollama model endpoint for litkg runtime refresh
	@python3 .agents/external/litkg-rs/scripts/kg/ollama_http.py check \
		--config "$(LITKG_CONFIG)"

kg-search: ## 📚 Search litkg-indexed code/docs/memory/backlog/literature (set KG_QUERY='...')
	@if [ -z "$(strip $(KG_QUERY))" ]; then \
		echo "$(RED)KG_QUERY is required, e.g. make kg-search KG_QUERY='entity-aware RRI'$(NC)"; \
		exit 2; \
	fi
	@cargo run --manifest-path "$(LITKG_MANIFEST)" -p litkg-cli -- kg find \
		--config "$(LITKG_CONFIG)" \
		--repo-root "$(LITKG_REPO_ROOT)" \
		--limit "$(KG_LIMIT)" \
		--format "$(KG_FORMAT)" \
		"$(KG_QUERY)"

kg-query: ## 📚 Build a litkg context pack for a project question (set KG_QUERY='...')
	@if [ -z "$(strip $(KG_QUERY))" ]; then \
		echo "$(RED)KG_QUERY is required, e.g. make kg-query KG_QUERY='current RRI contract'$(NC)"; \
		exit 2; \
	fi
	@cargo run --manifest-path "$(LITKG_MANIFEST)" -p litkg-cli -- context-pack \
		--config "$(LITKG_CONFIG)" \
		--repo-root "$(LITKG_REPO_ROOT)" \
		--task "$(KG_QUERY)" \
		--profile "$(LITKG_PROFILE)" \
		--format "$(KG_FORMAT)"

kg-brief: ## 📚 Build a litkg brief for a topic (set KG_TOPIC='...')
	@if [ -z "$(strip $(KG_TOPIC))" ]; then \
		echo "$(RED)KG_TOPIC is required, e.g. make kg-brief KG_TOPIC='entity-aware RRI'$(NC)"; \
		exit 2; \
	fi
	@cargo run --manifest-path "$(LITKG_MANIFEST)" -p litkg-cli -- context-pack \
		--config "$(LITKG_CONFIG)" \
		--repo-root "$(LITKG_REPO_ROOT)" \
		--task "brief: $(KG_TOPIC)" \
		--profile "$(LITKG_PROFILE)" \
		--format "$(KG_FORMAT)"

kg-route: ## 📚 Route a broad task through litkg evidence and backlog (set KG_TASK='...')
	@if [ -z "$(strip $(KG_TASK))" ]; then \
		echo "$(RED)KG_TASK is required, e.g. make kg-route KG_TASK='debug candidate pose frame mismatch'$(NC)"; \
		exit 2; \
	fi
	@cargo run --manifest-path "$(LITKG_MANIFEST)" -p litkg-cli -- context-pack \
		--config "$(LITKG_CONFIG)" \
		--repo-root "$(LITKG_REPO_ROOT)" \
		--task "$(KG_TASK)" \
		--profile "$(LITKG_PROFILE)" \
		--format "$(KG_FORMAT)"

kg-claim-check: ## 📚 Claim-check against litkg context (set KG_CLAIM='...')
	@if [ -z "$(strip $(KG_CLAIM))" ]; then \
		echo "$(RED)KG_CLAIM is required, e.g. make kg-claim-check KG_CLAIM='ARIA-NBV is an end-to-end policy'$(NC)"; \
		exit 2; \
	fi
	@cargo run --manifest-path "$(LITKG_MANIFEST)" -p litkg-cli -- context-pack \
		--config "$(LITKG_CONFIG)" \
		--repo-root "$(LITKG_REPO_ROOT)" \
		--task "claim-check: $(KG_CLAIM)" \
		--profile "$(LITKG_PROFILE)" \
		--format "$(KG_FORMAT)"

kg-consolidate: ## 📚 Propose memory/backlog consolidation updates without editing files
	@cargo run --manifest-path "$(LITKG_MANIFEST)" -p litkg-cli -- kg consolidate \
		--config "$(LITKG_CONFIG)" \
		--repo-root "$(LITKG_REPO_ROOT)" \
		--format "$(KG_FORMAT)"

kg-related: ## 📚 Find litkg context related to a path or symbol (set KG_RELATED_PATH='...')
	@if [ -z "$(strip $(KG_RELATED_PATH))" ]; then \
		echo "$(RED)KG_RELATED_PATH is required, e.g. make kg-related KG_RELATED_PATH='aria_nbv/aria_nbv/rri_metrics/oracle_rri.py'$(NC)"; \
		exit 2; \
	fi
	@cargo run --manifest-path "$(LITKG_MANIFEST)" -p litkg-cli -- kg find \
		--config "$(LITKG_CONFIG)" \
		--repo-root "$(LITKG_REPO_ROOT)" \
		--limit "$(KG_LIMIT)" \
		--format "$(KG_FORMAT)" \
		"$(KG_RELATED_PATH)"

kg-show-paper: ## 📚 Show one registered paper (set KG_PAPER='VIN-NBV-frahm2025')
	@if [ -z "$(strip $(KG_PAPER))" ]; then \
		echo "$(RED)KG_PAPER is required, e.g. make kg-show-paper KG_PAPER='VIN-NBV-frahm2025'$(NC)"; \
		exit 2; \
	fi
	@cargo run --manifest-path "$(LITKG_MANIFEST)" -p litkg-cli -- lit show \
		--config "$(LITKG_CONFIG)" \
		--paper "$(KG_PAPER)" \
		--format "$(KG_FORMAT)"

kg-sync: ## 📚 Sync literature registry from docs/references.bib
	@./scripts/kg/ingest_papers.sh sync

kg-materialize: ## 📚 Materialize literature into Markdown for agent consumption
	@echo "$(BLUE)Materializing literature...$(NC)"
	@./scripts/kg/ingest_papers.sh materialize
	@echo "$(GREEN)Literature materialized to .agents/kg/generated/literature/.$(NC)"

kg-index-code: ## 🏗️ Index aria_nbv code into Neo4j
	@./scripts/kg/index_code.sh

kg-ingest-docs: ## 📝 Ingest docs/ into Neo4j/Graphiti
	@./scripts/kg/ingest_docs.sh $(KG_DOC_PATHS)

kg-ingest-docs-smoke: ## 📝 Smoke-ingest one small doc into Neo4j/Graphiti
	@GRAPHITI_DOC_CHAR_LIMIT=1200 $(MAKE) kg-ingest-docs KG_DOC_PATHS=AGENTS.md

kg-enrich: ## 📚 Refresh litkg runtime embeddings and code↔doc links
	@KG_OLLAMA_CONFIG="$(LITKG_CONFIG)" \
		KG_CODE_REPO_ROOT="$(CURDIR)" \
		KG_CODE_PATH_PREFIX="$(KG_SRC_DIR)" \
		python3 .agents/external/litkg-rs/scripts/kg/enrich_embeddings.py

kg-ingest-papers: ## 📚 Full literature pipeline (sync, download, parse, materialize)
	@./scripts/kg/ingest_papers.sh

kg-export-neo4j: ## 📚 Export literature and memory nodes to a Neo4j import bundle
	@./scripts/kg/ingest_papers.sh export-neo4j

kg-semantic-enrich: ## 📚 Enrich literature registry with Semantic Scholar metadata
	@./scripts/kg/ingest_papers.sh semantic-enrich

kg-refresh-light: context kg-capabilities ## 📚 Refresh lightweight generated context and inspect litkg readiness

kg-refresh-code: kg-index-code ## 📚 Refresh litkg code-symbol index

kg-refresh-lit: kg-sync kg-materialize kg-export-neo4j ## 📚 Refresh literature registry/materialization/export; enrich with S2 when keyed
	@if [ -n "$$SEMANTIC_SCHOLAR_API_KEY" ]; then \
		$(MAKE) kg-semantic-enrich; \
	else \
		echo "$(YELLOW)Skipping Semantic Scholar enrichment; SEMANTIC_SCHOLAR_API_KEY is not set.$(NC)"; \
	fi

kg-refresh-full: kg-refresh-light kg-refresh-code kg-refresh-lit ## 📚 Refresh lightweight context, code index, and literature artifacts

#  ═══════════════════════════════════════════════════════════════════════
#  🔧 LRZ AI Systems operator helpers
#  ═══════════════════════════════════════════════════════════════════════

lrz-probe: ## 🔧 Inspect LRZ login/allocation, DSS containers, partitions, jobs, and GPU visibility
	@$(LRZ_SCRIPTS_DIR)/lrz-probe.sh

lrz-resources: ## 🔧 One-shot Slurm resource query (LRZ_RESOURCES_ARGS='summary' or 'partition lrz-v100x2')
	@$(LRZ_SCRIPTS_DIR)/lrz-resources.sh $(LRZ_RESOURCES_ARGS)

lrz-resources-gpu: ## 🔧 One-shot LRZ GPU partition summary
	@$(LRZ_SCRIPTS_DIR)/lrz-resources.sh gpu

lrz-resources-cpu: ## 🔧 One-shot LRZ CPU partition summary
	@$(LRZ_SCRIPTS_DIR)/lrz-resources.sh cpu

lrz-jobs: ## 🔧 Show current user's LRZ Slurm jobs once
	@$(LRZ_SCRIPTS_DIR)/lrz-resources.sh mine

lrz-dss-init: ## 🔧 Initialize ARIA DSS layout (requires ARIA_DSS=/dss/.../aria-nbv)
	@if [ -z "$(strip $(ARIA_DSS))" ]; then \
		echo "$(RED)ARIA_DSS is required, e.g. make lrz-dss-init ARIA_DSS=/dss/.../aria-nbv$(NC)"; \
		exit 2; \
	fi
	@$(LRZ_SCRIPTS_DIR)/lrz-dss-init.sh "$(ARIA_DSS)"

lrz-container-shell: ## 🔧 Launch Pyxis container shell inside an LRZ Slurm allocation (requires ARIA_DSS)
	@if [ -z "$(strip $(ARIA_DSS))" ]; then \
		echo "$(RED)ARIA_DSS is required, e.g. make lrz-container-shell ARIA_DSS=/dss/.../aria-nbv$(NC)"; \
		exit 2; \
	fi
	@ARIA_DSS="$(ARIA_DSS)" ARIA_REPO="$(ARIA_REPO)" LRZ_CONTAINER_IMAGE="$(LRZ_CONTAINER_IMAGE)" \
		$(LRZ_SCRIPTS_DIR)/lrz-container-shell.sh

lrz-sbatch-cpu: ## 🔧 Submit LRZ CPU batch job (requires ARIA_DSS and LRZ_CMD)
	@if [ -z "$(strip $(ARIA_DSS))" ]; then \
		echo "$(RED)ARIA_DSS is required, e.g. make lrz-sbatch-cpu ARIA_DSS=/dss/.../aria-nbv LRZ_CMD='uv run pytest ...'$(NC)"; \
		exit 2; \
	fi
	@if [ -z '$(strip $(LRZ_CMD))' ]; then \
		echo "$(RED)LRZ_CMD is required, e.g. make lrz-sbatch-cpu ARIA_DSS=/dss/.../aria-nbv LRZ_CMD='uv run pytest ...'$(NC)"; \
		exit 2; \
	fi
	@ARIA_DSS="$(ARIA_DSS)" ARIA_REPO="$(ARIA_REPO)" LRZ_TIME="$(LRZ_TIME)" LRZ_CPUS="$(LRZ_CPUS)" LRZ_MEM="$(LRZ_MEM)" \
		$(LRZ_SCRIPTS_DIR)/lrz-sbatch-cpu.sh '$(LRZ_CMD)'

lrz-sbatch-single-gpu: ## 🔧 Submit LRZ single-GPU batch job (requires ARIA_DSS and LRZ_CMD)
	@if [ -z "$(strip $(ARIA_DSS))" ]; then \
		echo "$(RED)ARIA_DSS is required, e.g. make lrz-sbatch-single-gpu ARIA_DSS=/dss/.../aria-nbv LRZ_CMD='uv run python -c \"print(1)\"'$(NC)"; \
		exit 2; \
	fi
	@if [ -z '$(strip $(LRZ_CMD))' ]; then \
		echo "$(RED)LRZ_CMD is required, e.g. make lrz-sbatch-single-gpu ARIA_DSS=/dss/.../aria-nbv LRZ_CMD='uv run python -c \"print(1)\"'$(NC)"; \
		exit 2; \
	fi
	@ARIA_DSS="$(ARIA_DSS)" ARIA_REPO="$(ARIA_REPO)" LRZ_PARTITION="$(LRZ_PARTITION)" LRZ_GPUS="$(LRZ_GPUS)" LRZ_TIME="$(LRZ_TIME)" LRZ_CPUS="$(LRZ_CPUS)" LRZ_MEM="$(LRZ_MEM)" LRZ_CONTAINER_IMAGE="$(LRZ_CONTAINER_IMAGE)" \
		$(LRZ_SCRIPTS_DIR)/lrz-sbatch-single-gpu.sh '$(LRZ_CMD)'

lrz-sbatch-multigpu: ## 🔧 Submit LRZ multi-GPU torchrun batch job (requires ARIA_DSS and LRZ_CMD)
	@if [ -z "$(strip $(ARIA_DSS))" ]; then \
		echo "$(RED)ARIA_DSS is required, e.g. make lrz-sbatch-multigpu ARIA_DSS=/dss/.../aria-nbv LRZ_GPUS=2 LRZ_CMD='<TRAIN_MODULE_OR_SCRIPT> <ARGS>'$(NC)"; \
		exit 2; \
	fi
	@if [ -z '$(strip $(LRZ_CMD))' ]; then \
		echo "$(RED)LRZ_CMD is required, e.g. make lrz-sbatch-multigpu ARIA_DSS=/dss/.../aria-nbv LRZ_GPUS=2 LRZ_CMD='<TRAIN_MODULE_OR_SCRIPT> <ARGS>'$(NC)"; \
		exit 2; \
	fi
	@ARIA_DSS="$(ARIA_DSS)" ARIA_REPO="$(ARIA_REPO)" LRZ_PARTITION="$(LRZ_PARTITION)" LRZ_GPUS="$(LRZ_GPUS)" LRZ_NODES="$(LRZ_NODES)" LRZ_TIME="$(LRZ_TIME)" LRZ_CPUS="$(LRZ_CPUS)" LRZ_MEM="$(LRZ_MEM)" LRZ_CONTAINER_IMAGE="$(LRZ_CONTAINER_IMAGE)" \
		$(LRZ_SCRIPTS_DIR)/lrz-sbatch-multigpu.sh '$(LRZ_CMD)'

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
			echo "# Data Contracts (aria_nbv)"; \
			echo ""; \
			$(PYTHON_INTERPRETER) aria_nbv/scripts/get_context.py contracts --root aria_nbv/aria_nbv \
				| sed "1{/^# Data Contracts$$/d;}"; \
		} > "$$contracts_out"; \
		echo "Wrote: $$index_out"; \
		echo "Wrote: $$lit_index_out"; \
		echo "Wrote: $$contracts_out"'
	@echo "$(GREEN)Refreshed lightweight context artifacts in $(CONTEXT_DIR)$(NC)"
	@echo "$(BLUE)Heavy fallback: make context-heavy$(NC)"
	@echo "$(BLUE)Tip: rg -n \"<pattern>\" $(CONTEXT_INDEX_OUT)$(NC)"

context-uml: _check_python ## 🗺️ Generate aria_nbv UML artifacts without printing them
	@bash -lc 'set -euo pipefail; \
		context_dir="$(CONTEXT_DIR)"; \
		uml_out="$(CONTEXT_UML_OUT)"; \
		uml_filtered_out="$(CONTEXT_UML_FILTERED_OUT)"; \
		mkdir -p "$$context_dir"; \
		mermaid_tmp="$$(mktemp)"; \
		mermaid_filtered="$$(mktemp)"; \
		$(PYTHON_INTERPRETER) -m syrenka classdiagram aria_nbv/aria_nbv > "$$mermaid_tmp"; \
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

context-uml-preview: _check_python ## 🗺️ Print the filtered aria_nbv UML to stdout
	@$(MAKE) --no-print-directory context-uml >/dev/null
	@echo "# Mermaid UML Diagram of the aria_nbv:"
	@echo "\`\`\`{mermaid}"
	@cat "$(CONTEXT_UML_FILTERED_OUT)"
	@echo "\`\`\`"

context-docstrings: _check_python ## 🗺️ Generate full aria_nbv class docstrings artifact
	@bash -lc 'set -euo pipefail; \
		context_dir="$(CONTEXT_DIR)"; \
		docstrings_out="$(CONTEXT_DOCSTRINGS_OUT)"; \
		mkdir -p "$$context_dir"; \
		{ \
			echo "# Class Docstrings (aria_nbv)"; \
			echo ""; \
			$(PYTHON_INTERPRETER) aria_nbv/scripts/get_context.py classes --root aria_nbv/aria_nbv --full-doc; \
		} > "$$docstrings_out"; \
		echo "Wrote: $$docstrings_out"'

context-tree: _check_python ## 🗺️ Generate aria_nbv directory tree artifact
	@bash -lc 'set -euo pipefail; \
		context_dir="$(CONTEXT_DIR)"; \
		tree_out="$(CONTEXT_TREE_OUT)"; \
		mkdir -p "$$context_dir"; \
		{ \
			echo "# Directory Tree (aria_nbv)"; \
			echo ""; \
			echo "Directory tree for aria_nbv/aria_nbv/:"; \
			if command -v tree >/dev/null 2>&1; then \
				tree aria_nbv/aria_nbv/ -I "__pycache__"; \
			else \
				find aria_nbv/aria_nbv/ -path "*/__pycache__" -prune -o -print \
					| sed "s#^aria_nbv/aria_nbv/##" \
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
			echo "2) Data contracts (aria_nbv)"; \
			echo "3) Mermaid UML (aria_nbv)"; \
			echo "4) Class docstrings (aria_nbv)"; \
			echo "5) Directory tree (aria_nbv)"; \
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
			echo "## 2) Data contracts (aria_nbv)"; \
			if [[ -f "$$contracts_out" ]]; then \
				sed "1{/^# Data Contracts (aria_nbv)$$/d;}" "$$contracts_out"; \
			else \
				echo "(missing $$contracts_out)"; \
			fi; \
			echo ""; \
			echo "## 3) Mermaid UML (aria_nbv)"; \
			echo "\`\`\`{mermaid}"; \
			cat "$$uml_out"; \
			echo "\`\`\`"; \
			echo ""; \
			echo "## 4) Class docstrings (aria_nbv)"; \
			if [[ -f "$$docstrings_out" ]]; then \
				sed "1{/^# Class Docstrings (aria_nbv)$$/d;}" "$$docstrings_out"; \
			else \
				echo "(missing $$docstrings_out)"; \
			fi; \
			echo ""; \
			echo "## 5) Directory tree (aria_nbv)"; \
			if [[ -f "$$tree_out" ]]; then \
				sed "1{/^# Directory Tree (aria_nbv)$$/d;}" "$$tree_out"; \
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
	@$(PYTHON_INTERPRETER) aria_nbv/scripts/get_context.py classes --root external/efm3d/efm3d --full-doc

	@echo "\n\n"

	echo "# Mermaid UML Diagram of the external/ATEK:\n\`\`\`{mermaid}"
	@$(PYTHON_INTERPRETER) -m syrenka classdiagram external/ATEK/atek
	echo "\`\`\`\n---\n"
	@$(PYTHON_INTERPRETER) aria_nbv/scripts/get_context.py classes --root external/ATEK/atek --full-doc

context-dir-tree: _check_python ## 🗺️ Print directory tree for `aria_nbv/aria_nbv/` (ignore __pycache__)
	@$(MAKE) --no-print-directory _context_dir_tree_print

_context_dir_tree_print:
	@echo "Directory tree for aria_nbv/aria_nbv/:"
	@bash -lc 'tree aria_nbv/aria_nbv/ -I "__pycache__"'

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
	@$(PYTHON_INTERPRETER) aria_nbv/scripts/generate_vin_v2_arch.py $(VIN_V2_ARCH_ARGS)

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

.PHONY: api-docs docs-lint
api-docs: ## Generate API reference pages via Quartodoc (hard alias failures fail, warnings are non-blocking)
	@./scripts/quarto_generate_api_docs.sh

.PHONY: docs-lint
docs-lint: _check_python ## Format QMD lists, then run Quarto checks
	@echo "$(BLUE)Formatting QMD list spacing…$(NC)"
	@$(PYTHON_INTERPRETER) $(QMD_FORMATTER) $(DOCS_DIR)
	@echo "$(BLUE)Running Quarto check…$(NC)"
	@quarto check

.PHONY: quarto-docs quarto-preview
quarto-docs: ## Render the Quarto website into docs/_site
	@cd docs && quarto render .

quarto-preview: ## Preview the Quarto website locally
	@cd docs && quarto preview

#  ═══════════════════════════════════════════════════════════════════════
#  🧾 Typst builds
#  ═══════════════════════════════════════════════════════════════════════

.PHONY: typst-paper typst-slide thesis-pdf thesis-watch proposal-pdf proposal-watch
typst-paper: ## Compile the Typst paper (docs/typst/seminar_paper/main.typ)
	@$(TYPST) compile --root $(TYPST_ROOT) $(TYPST_PAPER) $(TYPST_PAPER_PDF)

typst-slide: ## Compile a Typst slide deck (make typst-slide SLIDES=slides_4.typ or SLIDES=docs/typst/thesis_slides/slides_thesis_outlook.typ)
	@$(TYPST) compile --root $(TYPST_ROOT) $(SLIDES_SRC) $(SLIDES_PDF)

thesis-pdf: ## Compile the Typst thesis (docs/typst/thesis/main.typ)
	@$(TYPST) compile --root $(TYPST_ROOT) $(TYPST_THESIS) $(TYPST_THESIS_PDF)

thesis-watch: ## Watch and recompile the Typst thesis
	@$(TYPST) watch --root $(TYPST_ROOT) $(TYPST_THESIS) $(TYPST_THESIS_PDF)

proposal-pdf: ## Compile the Typst thesis proposal (docs/typst/thesis/proposal.typ)
	@$(TYPST) compile --root $(TYPST_ROOT) $(TYPST_PROPOSAL) $(TYPST_PROPOSAL_PDF)

proposal-watch: ## Watch and recompile the Typst thesis proposal
	@$(TYPST) watch --root $(TYPST_ROOT) $(TYPST_PROPOSAL) $(TYPST_PROPOSAL_PDF)

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
